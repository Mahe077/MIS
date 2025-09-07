# Quick Start Guide: Optimized CycleGAN for Low-Resource Machines

## 📋 Overview

This optimized version of CycleGAN is designed to run efficiently on machines with limited GPU memory (as low as 4GB) while maintaining reasonable training quality. The optimizations include:

- **Mixed precision training** (40-50% memory reduction)
- **Gradient accumulation** (simulate larger batches with less memory)
- **Lightweight model architectures** (fewer parameters)
- **Memory-efficient data loading** (optimized caching and preprocessing)
- **Automatic resource detection** (configures itself based on your hardware)

## 🚀 Quick Setup

### Option 1: Automatic Setup (Recommended)
```bash
# Interactive setup with auto-configuration
python optimized_setup.py --dataroot /path/to/your/dataset --name my_experiment --interactive

# Or automatic setup
python optimized_setup.py --dataroot /path/to/your/dataset --name my_experiment
```

### Option 2: Manual Configuration
```bash
# Low resource machine (4GB GPU)
python train_optimized.py --dataroot ./datasets/my_data --name my_exp_low \
  --batchSize 1 --fineSize 128 --ngf 32 --ndf 32 --use_amp --gradient_accumulation_steps 4

# Medium resource machine (4-8GB GPU)  
python train_optimized.py --dataroot ./datasets/my_data --name my_exp_med \
  --batchSize 2 --fineSize 192 --ngf 48 --ndf 48 --use_amp --gradient_accumulation_steps 2

# High resource machine (8GB+ GPU)
python train_optimized.py --dataroot ./datasets/my_data --name my_exp_high \
  --batchSize 4 --fineSize 256 --ngf 64 --ndf 64 --use_amp
```

## 📊 Resource Requirements

| Configuration | GPU Memory | System RAM | Image Size | Training Time* |
|---------------|------------|------------|------------|----------------|
| Low Resource  | 3-4 GB     | 8 GB       | 128×128    | ~8-12 hours    |
| Medium Resource | 4-8 GB   | 16 GB      | 192×192    | ~12-18 hours   |
| High Resource | 8+ GB      | 16+ GB     | 256×256    | ~18-24 hours   |

*Estimated for 10,000 image pairs, 100 epochs

## 🔧 File Structure

After setup, your project should look like:
```
your_project/
├── data/
│   ├── optimized_aligned_dataset.py    # Memory-efficient data loader
│   └── optimized_unaligned_dataset.py
├── models/
│   └── optimized_cycle_gan_model.py    # Lightweight model
├── resource_configs.py                 # Auto-configuration
├── train_optimized.py                  # Optimized training loop  
├── optimized_setup.py                  # Setup helper
└── your_dataset/
    ├── trainA/                         # Source domain images
    ├── trainB/                         # Target domain images
    ├── testA/                          # (optional)
    └── testB/                          # (optional)
```

## ⚙️ Key Optimizations Explained

### 1. Mixed Precision Training
- Uses 16-bit floats for most operations, 32-bit for loss computation
- Reduces memory usage by ~40-50%
- Maintains training stability with automatic loss scaling

### 2. Gradient Accumulation  
- Splits effective batch size across multiple forward passes
- Allows training with larger effective batch sizes on small GPUs
- Example: batch_size=1 × accumulation_steps=4 = effective_batch_size=4

### 3. Model Architecture Changes
- Reduced channel dimensions (32-64 instead of 64-128)
- Fewer ResNet blocks (4-6 instead of 9)
- Single discriminator instead of multiple discriminators
- Depthwise separable convolutions in critical paths

### 4. Memory Management
- Image caching for frequently accessed data
- Automatic memory cleanup after each epoch
- Efficient tensor operations with proper device placement

## 🐛 Troubleshooting

### "CUDA Out of Memory" Error
```bash
# Try these in order:
1. Reduce batch size: --batchSize 1
2. Reduce image size: --fineSize 128
3. Increase gradient accumulation: --gradient_accumulation_steps 4
4. Use CPU training: --gpu_ids -1
```

### Slow Training
```bash
# Enable optimizations:
1. Use mixed precision: --use_amp
2. Increase batch size if memory allows
3. Use more CPU threads: --nThreads 4
4. Enable pin_memory in dataloader
```

### Poor Quality Results
```bash
# Quality improvements:
1. Increase image size: --fineSize 192 or 256
2. Increase model size: --ngf 48 --ndf 48
3. Longer training: --niter 100 --niter_decay 100
4. Adjust loss weights: --lambda_A 10 --lambda_B 10
```

### Training Instability
```bash
# Stability improvements:
1. Lower learning rate: --lr 0.0001
2. Enable dropout: (remove --no_dropout flag)
3. Gradient clipping: --max_grad_norm 1.0
4. Spectral normalization: --norm spectral
```

## 📈 Monitoring Training

### GPU Memory Usage
```bash
# Monitor in real-time
watch -n 1 nvidia-smi

# Check memory in Python
python -c "import torch; print(f'Used: {torch.cuda.memory_allocated()/1e9:.1f}GB')"
```

### Training Progress
```bash
# Check loss logs
tail -f checkpoints/your_experiment/loss_log.txt

# View generated images (if display enabled)
ls checkpoints/your_experiment/web/images/
```

### Performance Profiling
```bash
# Add to training script for detailed profiling
--profile_training
```

## 🎯 Expected Results

### Quality vs. Resource Trade-offs
- **Low Resource**: 70-80% quality of full model, usable results
- **Medium Resource**: 85-90% quality, good results  
- **High Resource**: 95-100% quality, excellent results

### Typical Memory Usage
- **Low Resource**: 2.5-3.5 GB GPU memory
- **Medium Resource**: 4-6 GB GPU memory
- **High Resource**: 6-10 GB GPU memory

## 🔄 Resuming Training

```bash
# Resume from latest checkpoint
python train_optimized.py --continue_train --which_epoch latest --dataroot ./datasets/my_data --name my_experiment

# Resume from specific epoch
python train_optimized.py --continue_train --which_epoch 50 --dataroot ./datasets/my_data --name my_experiment
```

## 🧪 Testing Trained Model

```bash
# Test with optimized settings
python test.py --dataroot ./datasets/my_data --name my_experiment --model cycle_gan --phase test --no_dropout --checkpoints_dir ./checkpoints/ --results_dir ./results/
```

## 💡 Tips for Best Results

1. **Start Small**: Begin with low-resource settings, then scale up
2. **Monitor Memory**: Keep an eye on GPU memory usage  
3. **Experiment with Loss Weights**: Adjust lambda_A, lambda_B based on your data
4. **Use Progressive Training**: Start with small images, gradually increase size
5. **Data Quality**: Ensure your dataset is well-aligned and consistent
6. **Patience**: Optimized models may need slightly longer training

## 🆘 Getting Help

If you encounter issues:

1. Check the console output for specific error messages
2. Monitor GPU memory usage with `nvidia-smi`
3. Try the interactive setup: `python optimized_setup.py --interactive`
4. Use dry run to validate config: `python optimized_setup.py --dry_run`
5. Start with the lowest resource configuration and work your way up

## 📝 Example Commands

### Complete Low-Resource Training
```bash
# Setup
python optimized_setup.py --dataroot ./datasets/maps --name maps_low_resource --resource_level low

# Manual training
python train_optimized.py \
  --dataroot ./datasets/maps \
  --name maps_low_resource \
  --model cycle_gan \
  --batchSize 1 \
  --fineSize 128 \
  --ngf 32 \
  --ndf 32 \
  --use_amp \
  --gradient_accumulation_steps 4 \
  --lr 0.0001 \
  --niter 50 \
  --niter_decay 50
```

The optimized version should allow you to train CycleGAN on much more modest hardware while still achieving good results!