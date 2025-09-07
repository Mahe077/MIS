import numpy as np
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import torch.cuda.amp as amp
import gc


class OptimizedCycleGANModel(BaseModel):
    def name(self):
        return 'OptimizedCycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.use_amp = getattr(opt, 'use_amp', True)  # Mixed precision training
        self.grad_accumulate_steps = getattr(opt, 'grad_accumulate_steps', 1)
        
        # Reduce memory by using smaller tensors
        self.input_A1 = self.Tensor(nb, opt.input_nc, size, size).half() if self.use_amp else self.Tensor(nb, opt.input_nc, size, size)
        self.input_A2 = self.Tensor(nb, opt.input_nc, size, size).half() if self.use_amp else self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size).half() if self.use_amp else self.Tensor(nb, opt.output_nc, size, size)

        # Use smaller model dimensions
        ngf_reduced = max(opt.ngf // 2, 32)  # Reduce generator features
        ndf_reduced = max(opt.ndf // 2, 32)  # Reduce discriminator features
        n_blocks_reduced = max(opt.n_blocks // 2, 4)  # Reduce ResNet blocks

        # Lightweight generators
        self.netG_A = networks.define_G(
            opt.input_nc, opt.output_nc, ngf_reduced,
            'resnet_6blocks',  # Use lighter architecture
            opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids
        ).to('cuda')

        self.netG_B = networks.define_G(
            opt.input_nc, opt.output_nc, ngf_reduced,
            'resnet_6blocks',  # Use lighter architecture
            opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids
        ).to('cuda')

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # Single discriminator instead of multiple
            self.netD_A = networks.define_D(
                opt.output_nc, ndf_reduced, opt.which_model_netD,
                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids
            ).to('cuda')
            
            self.netD_B = networks.define_D(
                opt.input_nc, ndf_reduced, opt.which_model_netD,
                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids
            ).to('cuda')

        # Load pretrained weights if continuing training
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # Smaller image pools to save memory
            pool_size_reduced = min(opt.pool_size, 25)
            self.fake_A_pool = ImagePool(pool_size_reduced)
            self.fake_B_pool = ImagePool(pool_size_reduced)
            
            # Loss functions
            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor
            ).to('cuda')
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            # Mixed precision scaler
            if self.use_amp:
                self.scaler = amp.GradScaler()
            
            # Optimizers with reduced learning rates for stability
            lr_reduced = opt.lr * 0.8
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=lr_reduced, betas=(opt.beta1, 0.999), weight_decay=1e-4
            )
            self.optimizer_D_A = torch.optim.Adam(
                self.netD_A.parameters(), lr=lr_reduced, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D_B = torch.optim.Adam(
                self.netD_B.parameters(), lr=lr_reduced, betas=(opt.beta1, 0.999)
            )
            
            self.optimizers = [self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B]
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Optimized Networks initialized -------------')
        if hasattr(self, 'netG_A'):
            networks.print_network(self.netG_A)
        if hasattr(self, 'netG_B'):
            networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A1 = input['A1']
        input_A2 = input['A2']
        input_B = input['B']
        
        # Use non_blocking for faster GPU transfers
        self.input_A1.resize_(input_A1.size()).copy_(input_A1, non_blocking=True)
        self.input_A2.resize_(input_A2.size()).copy_(input_A2, non_blocking=True)
        self.input_B.resize_(input_B.size()).copy_(input_B, non_blocking=True)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A1 = self.input_A1.squeeze(1)
        self.real_A2 = self.input_A2.squeeze(1)
        self.real_B = self.input_B.squeeze(1) if len(self.input_B.shape) > 4 else self.input_B

    def backward_D_basic(self, netD, real, fake):
        # Real
        with amp.autocast() if self.use_amp else torch.no_grad():
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, True)
            
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            
            # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5

        if self.use_amp:
            self.scaler.scale(loss_D).backward()
        else:
            loss_D.backward()
            
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(torch.cat([self.fake_A1, self.fake_A2], dim=1))
        self.loss_D_B = self.backward_D_basic(self.netD_B, 
                                             torch.cat([self.real_A1, self.real_A2], dim=1), 
                                             fake_A)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        with amp.autocast() if self.use_amp else torch.no_grad():
            # Forward cycle: A -> B -> A
            self.fake_B = self.netG_A(torch.cat([self.real_A1, self.real_A2], dim=1))
            self.rec_A1, self.rec_A2 = torch.chunk(self.netG_B(self.fake_B), 2, dim=1)
            
            # Backward cycle: B -> A -> B
            fake_concat = self.netG_B(self.real_B)
            self.fake_A1, self.fake_A2 = torch.chunk(fake_concat, 2, dim=1)
            self.rec_B = self.netG_A(fake_concat)

            # GAN losses
            pred_fake_B = self.netD_A(self.fake_B)
            self.loss_G_A = self.criterionGAN(pred_fake_B, True)
            
            pred_fake_A = self.netD_B(fake_concat)
            self.loss_G_B = self.criterionGAN(pred_fake_A, True)

            # Cycle consistency losses
            self.loss_cycle_A = (self.criterionCycle(self.rec_A1, self.real_A1) + 
                               self.criterionCycle(self.rec_A2, self.real_A2)) * lambda_A
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

            # Identity losses (optional)
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            if lambda_idt > 0:
                self.idt_A = self.netG_A(torch.cat([self.real_A1, self.real_A2], dim=1))
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                
                idt_concat = self.netG_B(self.real_B)
                idt_A1, idt_A2 = torch.chunk(idt_concat, 2, dim=1)
                self.loss_idt_B = (self.criterionIdt(idt_A1, self.real_A1) + 
                                  self.criterionIdt(idt_A2, self.real_A2)) * lambda_A * lambda_idt

            # Total generator loss
            self.loss_G = (self.loss_G_A + self.loss_G_B + 
                          self.loss_cycle_A + self.loss_cycle_B + 
                          self.loss_idt_A + self.loss_idt_B)

        if self.use_amp:
            self.scaler.scale(self.loss_G).backward()
        else:
            self.loss_G.backward()

    def optimize_parameters(self):
        # Forward
        self.forward()
        
        # Update generators
        self.optimizer_G.zero_grad()
        self.backward_G()
        
        if self.use_amp:
            self.scaler.step(self.optimizer_G)
        else:
            self.optimizer_G.step()
        
        # Update discriminator A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        
        if self.use_amp:
            self.scaler.step(self.optimizer_D_A)
        else:
            self.optimizer_D_A.step()
        
        # Update discriminator B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        
        if self.use_amp:
            self.scaler.step(self.optimizer_D_B)
            self.scaler.update()
        else:
            self.optimizer_D_B.step()
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_current_errors(self):
        ret_errors = OrderedDict([
            ('D_A', self.loss_D_A.item()),
            ('G_A', self.loss_G_A.item()),
            ('Cyc_A', self.loss_cycle_A.item()),
            ('D_B', self.loss_D_B.item()),
            ('G_B', self.loss_G_B.item()),
            ('Cyc_B', self.loss_cycle_B.item())
        ])
        
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A.item()
            ret_errors['idt_B'] = self.loss_idt_B.item()
            
        return ret_errors

    def get_current_visuals(self):
        real_A1 = util.tensor2im(self.real_A1.data)
        real_A2 = util.tensor2im(self.real_A2.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A1 = util.tensor2im(self.rec_A1.data)
        rec_A2 = util.tensor2im(self.rec_A2.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A1 = util.tensor2im(self.fake_A1.data)
        fake_A2 = util.tensor2im(self.fake_A2.data)
        rec_B = util.tensor2im(self.rec_B.data)
        
        return OrderedDict([
            ('real_A1', real_A1), ('real_A2', real_A2), ('fake_B', fake_B),
            ('rec_A1', rec_A1), ('rec_A2', rec_A2),
            ('real_B', real_B), ('fake_A1', fake_A1), ('fake_A2', fake_A2),
            ('rec_B', rec_B)
        ])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        if self.isTrain:
            self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
            self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)