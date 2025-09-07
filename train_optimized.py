import time
import torch
import torch.cuda.amp as amp
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import os
import sys
import gc
from torch.profiler import profile, record_function, ProfilerActivity


def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0


def optimize_batch_size(model, opt):
    """Dynamically find optimal batch size"""
    original_batch_size = opt.batchSize
    test_batch_size = 1
    
    while test_batch_size <= 16:  # Max reasonable batch size
        try:
            # Create dummy data
            dummy_A1 = torch.randn(test_batch_size, 3, opt.fineSize, opt.fineSize).cuda()
            dummy_A2 = torch.randn(test_batch_size, 3, opt.fineSize, opt.fineSize).cuda()
            dummy_B = torch.randn(test_batch_size, 3, opt.fineSize, opt.fineSize).cuda()
            
            # Test forward pass
            with torch.no_grad():
                model.set_input({
                    'A1': dummy_A1.unsqueeze(1),
                    'A2': dummy_A2.unsqueeze(1), 
                    'B': dummy_B,
                    'A_paths': ['test'] * test_batch_size,
                    'B_paths': ['test'] * test_batch_size
                })
                model.forward()
                
            test_batch_size *= 2
            print(f"Batch size {test_batch_size // 2} works. GPU Memory: {get_memory_usage():.2f}GB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimal_size = max(test_batch_size // 2, 1)
                print(f"Optimal batch size found: {optimal_size}")
                return optimal_size
            else:
                raise e
        finally:
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    return test_batch_size // 2


def train_with_gradient_accumulation(model, data_loader, opt, visualizer):
    """Training loop with gradient accumulation and mixed precision"""
    total_steps = 0
    accumulation_steps = getattr(opt, 'gradient_accumulation_steps', 2)
    
    # Enable mixed precision
    use_amp = getattr(opt, 'use_amp', True)
    scaler = amp.GradScaler() if use_amp else None
    
    # Memory monitoring
    max_memory = 0
    
    print(f"Starting training with gradient accumulation (steps: {accumulation_steps})")
    print(f"Mixed precision: {use_amp}")
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        
        # Reset gradients at the start of each epoch
        model.optimizer_G.zero_grad()
        if hasattr(model, 'optimizer_D_A'):
            model.optimizer_D_A.zero_grad()
        if hasattr(model, 'optimizer_D_B'):
            model.optimizer_D_B.zero_grad()
        
        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            
            # Set input data
            model.set_input(data)
            
            # Forward pass with mixed precision
            if use_amp:
                with amp.autocast():
                    model.forward()
                    # Calculate losses but don't backward yet for accumulation
                    lambda_idt = model.opt.identity
                    lambda_A = model.opt.lambda_A
                    lambda_B = model.opt.lambda_B
                    
                    # Generator forward
                    model.fake_B = model.netG_A(torch.cat([model.real_A1, model.real_A2], dim=1))
                    model.rec_A1, model.rec_A2 = torch.chunk(model.netG_B(model.fake_B), 2, dim=1)
                    
                    fake_concat = model.netG_B(model.real_B)
                    model.fake_A1, model.fake_A2 = torch.chunk(fake_concat, 2, dim=1)
                    model.rec_B = model.netG_A(fake_concat)
                    
                    # Calculate generator losses
                    pred_fake_B = model.netD_A(model.fake_B)
                    loss_G_A = model.criterionGAN(pred_fake_B, True)
                    
                    pred_fake_A = model.netD_B(fake_concat)
                    loss_G_B = model.criterionGAN(pred_fake_A, True)
                    
                    loss_cycle_A = (model.criterionCycle(model.rec_A1, model.real_A1) + 
                                   model.criterionCycle(model.rec_A2, model.real_A2)) * lambda_A
                    loss_cycle_B = model.criterionCycle(model.rec_B, model.real_B) * lambda_B
                    
                    loss_G = (loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B) / accumulation_steps
                    
                # Backward pass for generator
                scaler.scale(loss_G).backward()
                
                # Discriminator losses
                fake_B_pooled = model.fake_B_pool.query(model.fake_B)
                pred_real_B = model.netD_A(model.real_B)
                pred_fake_B = model.netD_A(fake_B_pooled.detach())
                loss_D_A = (model.criterionGAN(pred_real_B, True) + 
                           model.criterionGAN(pred_fake_B, False)) * 0.5 / accumulation_steps
                
                scaler.scale(loss_D_A).backward()
                
                fake_A_pooled = model.fake_A_pool.query(fake_concat)
                pred_real_A = model.netD_B(torch.cat([model.real_A1, model.real_A2], dim=1))
                pred_fake_A = model.netD_B(fake_A_pooled.detach())
                loss_D_B = (model.criterionGAN(pred_real_A, True) + 
                           model.criterionGAN(pred_fake_A, False)) * 0.5 / accumulation_steps
                
                scaler.scale(loss_D_B).backward()
                
            else:
                # Regular precision training with accumulation
                model.forward()
                model.backward_G()
                model.backward_D_A() 
                model.backward_D_B()
            
            # Update parameters every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                if use_amp and scaler is not None:
                    scaler.step(model.optimizer_G)
                    if hasattr(model, 'optimizer_D_A'):
                        scaler.step(model.optimizer_D_A)
                    if hasattr(model, 'optimizer_D_B'):
                        scaler.step(model.optimizer_D_B)
                    scaler.update()
                else:
                    model.optimizer_G.step()
                    if hasattr(model, 'optimizer_D_A'):
                        model.optimizer_D_A.step()
                    if hasattr(model, 'optimizer_D_B'):
                        model.optimizer_D_B.step()
                
                # Zero gradients after update
                model.optimizer_G.zero_grad()
                if hasattr(model, 'optimizer_D_A'):
                    model.optimizer_D_A.zero_grad()
                if hasattr(model, 'optimizer_D_B'):
                    model.optimizer_D_B.zero_grad()
            
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            
            # Memory monitoring
            current_memory = get_memory_usage()
            max_memory = max(max_memory, current_memory)
            
            # Display results
            if total_steps % opt.display_freq == 0:
                try:
                    visuals = model.get_current_visuals()
                    visualizer.display_current_results(visuals, epoch)
                except Exception as e:
                    print(f"Warning: Could not display visuals: {e}")
            
            # Print progress
            if i % 50 == 0:  # Print every 50 iterations
                try:
                    errors = model.get_current_errors()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    message = f'(epoch: {epoch}, iters: {i}, time: {t:.3f}, mem: {current_memory:.1f}GB) '
                    for k, v in errors.items():
                        message += f'{k}: {v:.3f} '
                    print(message)
                except Exception as e:
                    print(f"Epoch {epoch}, Iter {i}, Memory: {current_memory:.1f}GB")
            
            # Save latest model
            if total_steps % opt.save_latest_freq == 0:
                print(f'Saving latest model (epoch {epoch}, total_steps {total_steps})')
                model.save('latest')
                
                # Memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
        
        # End of epoch
        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving model at end of epoch {epoch}, iters {total_steps}')
            model.save('latest')
            model.save(epoch)
        
        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t '
              f'Time Taken: {time.time() - epoch_start_time:.0f} sec \t '
              f'Max Memory: {max_memory:.1f}GB')
        
        # Update learning rate
        model.update_learning_rate()
        
        # Reset max memory for next epoch
        max_memory = 0


def main():
    try:
        opt = TrainOptions().parse()
        
        # Add optimization options
        opt.use_amp = True  # Enable mixed precision
        opt.gradient_accumulation_steps = 2  # Reduce memory usage
        
        # Reduce batch size for memory efficiency
        original_batch_size = opt.batchSize
        opt.batchSize = min(opt.batchSize, 4)  # Cap at 4 for low-resource machines
        
        print(f"Optimizations applied:")
        print(f"  - Mixed precision: {opt.use_amp}")
        print(f"  - Gradient accumulation: {opt.gradient_accumulation_steps}")
        print(f"  - Batch size: {original_batch_size} -> {opt.batchSize}")
        
        # Create data loader
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print(f'#training images = {dataset_size}')
        
        # Create model
        model = create_model(opt)
        print("Model created successfully")
        
        # Optimize batch size dynamically
        print("Finding optimal batch size...")
        optimal_batch_size = optimize_batch_size(model, opt)
        if optimal_batch_size != opt.batchSize:
            print(f"Adjusting batch size from {opt.batchSize} to {optimal_batch_size}")
            opt.batchSize = optimal_batch_size
            # Recreate data loader with new batch size
            data_loader = CreateDataLoader(opt)
            dataset = data_loader.load_data()
        
        # Create visualizer
        visualizer = Visualizer(opt)
        
        # Log optimization settings
        loss_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(loss_log_name, "a") as log_file:
            log_file.write(f'================ Optimization Settings ================\n')
            log_file.write(f'Mixed Precision: {opt.use_amp}\n')
            log_file.write(f'Gradient Accumulation Steps: {opt.gradient_accumulation_steps}\n')
            log_file.write(f'Batch Size: {opt.batchSize}\n')
            log_file.write(f'GPU Memory at Start: {get_memory_usage():.2f}GB\n')
            log_file.write(f'======================================================\n')
        
        model_start_time = time.time()
        
        # Start optimized training
        train_with_gradient_accumulation(model, dataset, opt, visualizer)
        
        # Final logging
        with open(loss_log_name, "a") as log_file:
            total_time = time.time() - model_start_time
            log_file.write(f'================ Training Completed ================\n')
            log_file.write(f'Total Time: {total_time:.1f} seconds ({total_time/3600:.1f} hours)\n')
            log_file.write(f'Final GPU Memory: {get_memory_usage():.2f}GB\n')
            now = time.strftime("%c")
            log_file.write(f'Completed at: {now}\n')
            log_file.write(f'===================================================\n')
        
        print(f"Training completed successfully!")
        print(f"Total training time: {(time.time() - model_start_time)/3600:.1f} hours")
        sys.exit(0)
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "="*50)
            print("GPU OUT OF MEMORY ERROR")
            print("="*50)
            print("Try these solutions:")
            print("1. Reduce batch size further (current: {})".format(getattr(opt, 'batchSize', 'unknown')))
            print("2. Reduce image resolution (fineSize)")
            print("3. Use CPU training (add --gpu_ids -1)")
            print("4. Enable gradient checkpointing")
            print("5. Increase gradient accumulation steps")
            print("="*50)
            
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            sys.exit(1)
        else:
            raise e
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        sys.exit(1)


if __name__ == '__main__':
    main()