import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.no_input = opt.no_input
        # store inputs in a tensor # DONE
        self.input_A1 = self.Tensor(nb, opt.input_nc, size, size)
        # store inputs in a tensor # DONE
        self.input_A2 = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        #  G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = (networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                       'resnetMM', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)).to('cuda')

        self.netG_B = (networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                       'resnetMMReverse', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)).to('cuda')

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = (networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD,
                                             opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)).to('cuda')
            self.netD_B1 = (networks.define_D(
                opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)).to('cuda')
            self.netD_B2 = (networks.define_D(
                opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)).to('cuda')

        if not self.isTrain or opt.continue_train:

            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B1, 'D_B1', which_epoch)
                self.load_network(self.netD_B2, 'D_B2', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A1_pool = ImagePool(opt.pool_size)
            self.fake_A2_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = (networks.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor)).to('cuda')
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionLatent = torch.nn.L1Loss()
            self.criterionNew = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(
            ), self.netG_B.parameters()), lr=1.5*opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B1.parameters(
            ), self.netD_B2.parameters()), lr=opt.lr*0.1, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(
                self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            # for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(
                self.optimizers[0], opt, lr=1.5))
            self.schedulers.append(networks.get_scheduler(
                self.optimizers[1], opt, lr=0.1))
            self.schedulers.append(networks.get_scheduler(
                self.optimizers[2], opt, lr=1.0))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B1)
            networks.print_network(self.netD_B2)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A1 = input['A1']
        input_A2 = input['A2']
        input_B = input['B']
        self.input_A1.resize_(input_A1.size()).copy_(input_A1).to('cuda')
        self.input_A2.resize_(input_A2.size()).copy_(input_A2).to('cuda')
        self.input_B.resize_(input_B.size()).copy_(input_B).to('cuda')
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        print("heloo", self.input_A1.device,
              self.input_A2.device, self.input_B.device)

    def forward(self):
        self.real_A1 = Variable(self.input_A1).to('cuda')
        self.real_A2 = Variable(self.input_A2).to('cuda')
        self.real_B = Variable(self.input_B).to('cuda')
        # remoce addistional frames diementon
        self.real_A1 = self.real_A1.squeeze(1)
        self.real_A2 = self.real_A2.squeeze(1)
        # print("shape input a1", self.input_A1.shape)  -> torch.Size([1, 1, 3, 256, 256])
        # print("shape real a1", self.real_A1.shape) -> torch.Size([1, 3, 256, 256])

    def test(self):
        self.real_A1 = Variable(self.input_A1, volatile=True)
        self.real_A2 = Variable(self.input_A2, volatile=True)

        # remoce addistional frames diementon
        self.real_A1 = self.real_A1.squeeze(1)
        self.real_A2 = self.real_A2.squeeze(1)
        self.fake_B, latent_fB, self.intermediate_fakeB = self.netG_A.forward(
            self.real_A1, self.real_A2)
        self.rec_A1, self.rec_A2, latent_rA, self.intermediate_realA = self.netG_B.forward(
            self.fake_B)
        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A1, self.fake_A2, latent_fA, self.intermediate_fakeA = self.netG_B.forward(
            self.real_B)
        self.rec_B, latent_rB, self.intermediate_realB = self.netG_A.forward(
            self.fake_A1, self.fake_A2)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A1 = self.fake_A1_pool.query(self.fake_A1)
        fake_A2 = self.fake_A2_pool.query(self.fake_A2)
        self.loss_D_B = 0.5*(self.backward_D_basic(self.netD_B1, self.real_A1,
                             fake_A1)+self.backward_D_basic(self.netD_B2, self.real_A2, fake_A2))

    def l1_loss(self, input, target):
        return torch.sum(torch.abs(input - target)) / input.data.nelement()

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_latent = self.opt.lambda_latent
        lambda_new = self.opt.lambda_intermediate
        self.loss_idt_A = 0
        self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B, latent_fB, self.intermediate_fakeB = self.netG_A.forward(
            self.real_A1, self.real_A2)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)

        self.fake_A1, self.fake_A2, latent_fA, self.intermediate_fakeA = self.netG_B.forward(
            self.real_B)
        pred_fake1 = self.netD_B1.forward(self.fake_A1)
        pred_fake2 = self.netD_B2.forward(self.fake_A2)
        self.loss_G_B = (self.criterionGAN(pred_fake1, True) +
                         self.criterionGAN(pred_fake2, True))

        # Cycle consistency loss
        self.rec_A1, self.rec_A2, latent_rA, self.intermediate_realA = self.netG_B.forward(
            self.fake_B)
        self.loss_cycle_A = (self.criterionCycle(self.rec_A1, self.real_A1)
                             * lambda_A+self.criterionCycle(self.rec_A2, self.real_A2) * lambda_A)

        self.rec_B, latent_rB, self.intermediate_realB = self.netG_A.forward(
            self.fake_A1, self.fake_A2)
        self.loss_cycle_B = self.criterionCycle(
            self.rec_B, self.real_B) * lambda_B

        # Latent consistency loss
        self.latent_loss = lambda_latent * \
            self.l1_loss(latent_fB, latent_rA)+lambda_latent * \
            self.l1_loss(latent_fA, latent_rB)

        # Intermediate consistency loss
        self.new_loss = lambda_new*self.criterionNew(self.intermediate_fakeB, self.intermediate_realA) + \
            lambda_new * \
            self.criterionNew(self.intermediate_fakeA, self.intermediate_realB)
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
            self.loss_idt_A + self.loss_idt_B + self.latent_loss + self.new_loss
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        D_A = self.loss_D_A.data.item()
        G_A = self.loss_G_A.data.item()
        Cyc_A = self.loss_cycle_A.data.item()
        D_B = self.loss_D_B.data.item()
        G_B = self.loss_G_B.data.item()
        Cyc_B = self.loss_cycle_B.data.item()
        new = self.new_loss.item()
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data.item()
            idt_B = self.loss_idt_B.data.item()
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B), ('new_loss', new)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('new_loss', new)])

    def get_current_visuals(self):
        real_A1 = util.tensor2im(self.real_A1.data)
        real_A2 = util.tensor2im(self.real_A2.data)
        print(self.real_A1.data.shape, self.real_A2.data.shape)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A1 = util.tensor2im(self.rec_A1.data)
        rec_A2 = util.tensor2im(self.rec_A2.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A1 = util.tensor2im(self.fake_A1.data)
        fake_A2 = util.tensor2im(self.fake_A2.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            real_A = util.tensor2im(self.real_A.data)
            fake_A = util.tensor2im(self.fake_A.data)
            rec_A = util.tensor2im(self.rec_A.data)
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A1', real_A1), ('real_A2', real_A2), ('fake_B', fake_B), ('rec_A1', rec_A1), ('rec_A2', rec_A2),
                                ('real_B', real_B), ('fake_A1', fake_A1),  ('fake_A2', fake_A2), ('rec_B', rec_B)])

    def save(self, label):
        self.save_network(self.netD_B2, 'D_B2', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B1, 'D_B1', label, self.gpu_ids)
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
