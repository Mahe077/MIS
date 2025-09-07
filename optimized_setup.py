#!/usr/bin/env python3
"""
Optimized CycleGAN Training Setup
This script helps you set up and run training with optimal settings for your hardware.
"""

import argparse
import os
import sys
import torch
import subprocess
import time
from resource_configs import (
    apply_resource_config, monitor_memory_usage, 
    estimate_training_time, validate_config, 
    auto_detect_resources, cleanup_memory
)


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'torch', 'torchvision', 'PIL', 'numpy', 'psutil'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    print("✓ All dependencies are installed")
    return True


def check_gpu_setup():
    """Check GPU availability and setup"""
    if not torch.cuda.is_available():
        print("⚠ CUDA is not available. Training will use CPU (very slow)")
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            return False
    else:
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        
        print(f"✓ GPU setup detected:")
        print(f"  - Device count: {gpu_count}")
        print(f"  - Current device: {current_device}")
        print(f"  - GPU name: {gpu_name}")
        print(f"  - GPU memory: {gpu_memory:.1f} GB")
    
    return True


def setup_directories(opt):
    """Create necessary directories"""
    dirs_to_create = [
        opt.checkpoints_dir,
        os.path.join(opt.checkpoints_dir, opt.name),
        opt.results_dir if hasattr(opt, 'results_dir') else None
    ]
    
    for dir_path in dirs_to_create:
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")


def count_dataset_images(dataroot, phase='train'):
    """Count images in dataset"""
    train_dir = os.path.join(dataroot, phase)
    if not os.path.exists(train_dir):
        print(f"Warning: Dataset directory {train_dir} does not exist")
        return 0
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    count = 0
    
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                count += 1
    
    print(f"Found {count} images in {train_dir}")
    return count


def create_training_script(opt):
    """Create optimized training script"""
    script_content = f"""#!/bin/bash

# Optimized CycleGAN Training Script
# Generated automatically with resource optimization

echo "Starting optimized CycleGAN training..."
echo "Configuration: Resource level auto-detected"
echo "Batch size: {opt.batchSize}"
echo "Image size: {opt.fineSize}x{opt.fineSize}"
echo "Mixed precision: {getattr(opt, 'use_amp', True)}"

# Set environment variables for optimization
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Monitor GPU memory before training
nvidia-smi

# Run training with optimizations
python train_optimized.py \\
    --dataroot {opt.dataroot} \\
    --name {opt.name} \\
    --model cycle_gan \\
    --batchSize {opt.batchSize} \\
    --fineSize {opt.fineSize} \\
    --loadSize {opt.loadSize} \\
    --ngf {getattr(opt, 'ngf', 64)} \\
    --ndf {getattr(opt, 'ndf', 64)} \\
    --lr {opt.lr} \\
    --niter {opt.niter} \\
    --niter_decay {opt.niter_decay} \\
    --lambda_A {opt.lambda_A} \\
    --lambda_B {opt.lambda_B} \\
    --identity {opt.identity} \\
    --pool_size {opt.pool_size} \\
    --checkpoints_dir {opt.checkpoints_dir} \\
    --gpu_ids {','.join(map(str, opt.gpu_ids))} \\
    --display_freq {opt.display_freq} \\
    --save_latest_freq {opt.save_latest_freq} \\
    --save_epoch_freq {opt.save_epoch_freq} \\
    {'--no_dropout' if opt.no_dropout else ''} \\
    {'--use_amp' if getattr(opt, 'use_amp', True) else ''} \\
    --gradient_accumulation_steps {getattr(opt, 'gradient_accumulation_steps', 1)}

echo "Training completed!"
"""
    
    script_path = "train_optimized.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    print(f"Created training script: {script_path}")
    return script_path


def interactive_setup():
    """Interactive setup for users who want guided configuration"""
    print("\n" + "="*60)
    print("INTERACTIVE CYCLEGAN OPTIMIZATION SETUP")
    print("="*60)
    
    # Basic information
    print("\n1. Dataset Configuration")
    dataroot = input("Enter path to your dataset: ").strip()
    if not os.path.exists(dataroot):
        print(f"Warning: Path {dataroot} does not exist")
    
    name = input("Enter experiment name: ").strip()
    if not name:
        name = "cyclegan_optimized_experiment"
        print(f"Using default name: {name}")
    
    # Auto-detect resources
    print("\n2. Resource Detection")
    resource_level = auto_detect_resources()
    
    print(f"\n3. Advanced Options")
    use_custom = input("Use custom settings instead of auto-detected? (y/n): ").lower() == 'y'
    
    if use_custom:
        print("Enter custom settings (press Enter for defaults):")
        batch_size = input(f"Batch size [auto]: ").strip()
        image_size = input(f"Image size [auto]: ").strip()
        epochs = input(f"Training epochs [auto]: ").strip()
    
    # Create mock options object for configuration
    class MockOpt:
        def __init__(self):
            self.dataroot = dataroot
            self.name = name
            self.checkpoints_dir = './checkpoints'
            self.results_dir = './results'
            self.gpu_ids = [0] if torch.cuda.is_available() else []
            self.display_freq = 400
            self.save_latest_freq = 5000
            self.save_epoch_freq = 5
            self.which_direction = 'AtoB'
            self.dataset_mode = 'aligned'  # Most common
            
    opt = MockOpt()
    
    # Apply resource configuration
    opt = apply_resource_config(opt, resource_level)
    
    # Override with custom settings if provided
    if use_custom:
        if batch_size.isdigit():
            opt.batchSize = int(batch_size)
        if image_size.isdigit():
            opt.fineSize = int(image_size)
            opt.loadSize = int(image_size) + 30
        if epochs.isdigit():
            total_epochs = int(epochs)
            opt.niter = total_epochs // 2
            opt.niter_decay = total_epochs - opt.niter
    
    return opt


def main():
    parser = argparse.ArgumentParser(description='Optimized CycleGAN Training Setup')
    parser.add_argument('--dataroot', required=True, help='Path to dataset')
    parser.add_argument('--name', required=True, help='Experiment name')
    parser.add_argument('--resource_level', choices=['low', 'medium', 'high'], 
                       help='Resource level (auto-detected if not specified)')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run interactive setup')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show configuration without training')
    parser.add_argument('--checkpoints_dir', default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--gpu_ids', default='0', help='GPU IDs to use')
    
    args = parser.parse_args()
    
    print("CycleGAN Optimization Setup")
    print("="*40)
    
    # Run checks
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\n2. Checking GPU setup...")
    if not check_gpu_setup():
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        opt = interactive_setup()
    else:
        # Create options from arguments
        class Options:
            def __init__(self):
                self.dataroot = args.dataroot
                self.name = args.name
                self.checkpoints_dir = args.checkpoints_dir
                self.gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x.strip()]
                self.display_freq = 400
                self.save_latest_freq = 5000
                self.save_epoch_freq = 5
                self.which_direction = 'AtoB'
        
        opt = Options()
        opt = apply_resource_config(opt, args.resource_level)
    
    print("\n3. Validating configuration...")
    validate_config(opt)
    
    print("\n4. Setting up directories...")
    setup_directories(opt)
    
    print("\n5. Analyzing dataset...")
    dataset_size = count_dataset_images(opt.dataroot)
    if dataset_size == 0:
        print("No images found in dataset. Please check the path.")
        sys.exit(1)
    
    print("\n6. Estimating training time...")
    estimate_training_time(opt, dataset_size)
    
    print("\n7. Current system status:")
    monitor_memory_usage()
    
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Configuration Summary")
        print("="*60)
        print(f"Dataset: {opt.dataroot}")
        print(f"Experiment: {opt.name}")
        print(f"Batch size: {opt.batchSize}")
        print(f"Image size: {opt.fineSize}x{opt.fineSize}")
        print(f"Model features: G={getattr(opt, 'ngf', 64)}, D={getattr(opt, 'ndf', 64)}")
        print(f"Learning rate: {opt.lr}")
        print(f"Epochs: {opt.niter + opt.niter_decay}")
        print(f"Mixed precision: {getattr(opt, 'use_amp', True)}")
        print(f"Gradient accumulation: {getattr(opt, 'gradient_accumulation_steps', 1)}")
        print("\nTo start training, run without --dry_run flag")
        return
    
    print("\n8. Creating training script...")
    script_path = create_training_script(opt)
    
    print("\n" + "="*60)
    print("SETUP COMPLETED!")
    print("="*60)
    print(f"Training script created: {script_path}")
    print(f"To start training, run: ./{script_path}")
    print("")
    print("Additional tips:")
    print("- Monitor GPU memory with: watch -n 1 nvidia-smi")
    print("- Check training progress in: " + os.path.join(opt.checkpoints_dir, opt.name))
    print("- Use Ctrl+C to stop training (model will be saved)")
    print("")
    
    # Ask if user wants to start training immediately
    if not args.interactive:
        start_now = input("Start training now? (y/n): ").lower() == 'y'
        if start_now:
            print("Starting training...")
            cleanup_memory()  # Clean memory before starting
            subprocess.run(['bash', script_path])
    
    print("Setup complete!")


if __name__ == '__main__':
    main()