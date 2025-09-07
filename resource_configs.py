# resource_configs.py
"""
Configuration templates for different resource levels
"""

import argparse


class LowResourceConfig:
    """Configuration for machines with 4GB GPU memory or less"""
    
    @staticmethod
    def apply_config(opt):
        # Memory optimizations
        opt.batchSize = 1
        opt.fineSize = 128  # Smaller images
        opt.loadSize = 144  # Proportionally smaller load size
        opt.ngf = 32       # Reduce generator features
        opt.ndf = 32       # Reduce discriminator features
        
        # Training optimizations  
        opt.use_amp = True
        opt.gradient_accumulation_steps = 4  # Simulate larger batch size
        opt.pool_size = 10  # Smaller fake image pool
        opt.nThreads = 2    # Fewer data loading threads
        
        # Loss function adjustments
        opt.lambda_A = 5.0   # Reduce cycle loss weight
        opt.lambda_B = 5.0
        opt.identity = 0.1   # Reduce identity loss
        
        # Model architecture
        opt.n_blocks = 4     # Fewer ResNet blocks
        opt.no_dropout = False  # Keep dropout for regularization
        
        # Training schedule adjustments
        opt.lr = 0.0001      # Lower learning rate for stability
        opt.niter = 50       # Fewer epochs at full LR
        opt.niter_decay = 50 # Fewer decay epochs
        
        print("Applied LOW RESOURCE configuration:")
        print(f"  - Batch size: {opt.batchSize}")
        print(f"  - Image size: {opt.fineSize}x{opt.fineSize}")
        print(f"  - Model features: G={opt.ngf}, D={opt.ndf}")
        print(f"  - Gradient accumulation: {opt.gradient_accumulation_steps}")
        print(f"  - Total epochs: {opt.niter + opt.niter_decay}")
        
        return opt


class MediumResourceConfig:
    """Configuration for machines with 4-8GB GPU memory"""
    
    @staticmethod
    def apply_config(opt):
        # Memory optimizations
        opt.batchSize = 2
        opt.fineSize = 192   # Medium-sized images
        opt.loadSize = 216
        opt.ngf = 48        # Medium model size
        opt.ndf = 48
        
        # Training optimizations
        opt.use_amp = True
        opt.gradient_accumulation_steps = 2
        opt.pool_size = 25
        opt.nThreads = 4
        
        # Loss weights (balanced)
        opt.lambda_A = 8.0
        opt.lambda_B = 8.0
        opt.identity = 0.2
        
        # Model architecture
        opt.n_blocks = 6
        opt.no_dropout = False
        
        # Training schedule
        opt.lr = 0.0002
        opt.niter = 75
        opt.niter_decay = 75
        
        print("Applied MEDIUM RESOURCE configuration:")
        print(f"  - Batch size: {opt.batchSize}")
        print(f"  - Image size: {opt.fineSize}x{opt.fineSize}")
        print(f"  - Model features: G={opt.ngf}, D={opt.ndf}")
        print(f"  - Gradient accumulation: {opt.gradient_accumulation_steps}")
        print(f"  - Total epochs: {opt.niter + opt.niter_decay}")
        
        return opt


class HighResourceConfig:
    """Configuration for machines with 8GB+ GPU memory"""
    
    @staticmethod
    def apply_config(opt):
        # Standard/high quality settings
        opt.batchSize = 4
        opt.fineSize = 256   # Full resolution
        opt.loadSize = 286
        opt.ngf = 64        # Standard model size
        opt.ndf = 64
        
        # Training optimizations (still beneficial)
        opt.use_amp = True
        opt.gradient_accumulation_steps = 1  # No accumulation needed
        opt.pool_size = 50
        opt.nThreads = 8
        
        # Standard loss weights
        opt.lambda_A = 10.0
        opt.lambda_B = 10.0
        opt.identity = 0.5
        
        # Full model
        opt.n_blocks = 9
        opt.no_dropout = True  # Can afford to disable dropout
        
        # Standard training schedule
        opt.lr = 0.0002
        opt.niter = 100
        opt.niter_decay = 100
        
        print("Applied HIGH RESOURCE configuration:")
        print(f"  - Batch size: {opt.batchSize}")
        print(f"  - Image size: {opt.fineSize}x{opt.fineSize}")
        print(f"  - Model features: G={opt.ngf}, D={opt.ndf}")
        print(f"  - Gradient accumulation: {opt.gradient_accumulation_steps}")
        print(f"  - Total epochs: {opt.niter + opt.niter_decay}")
        
        return opt


def auto_detect_resources():
    """Automatically detect available resources and suggest configuration"""
    import torch
    import psutil
    
    # Check GPU memory
    gpu_memory_gb = 0
    if torch.cuda.is_available():
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory_bytes / (1024**3)
    
    # Check system RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"Detected resources:")
    print(f"  - GPU memory: {gpu_memory_gb:.1f} GB")
    print(f"  - System RAM: {ram_gb:.1f} GB")
    
    # Suggest configuration
    if gpu_memory_gb < 4:
        print("Recommendation: LOW RESOURCE configuration")
        return "low"
    elif gpu_memory_gb < 8:
        print("Recommendation: MEDIUM RESOURCE configuration") 
        return "medium"
    else:
        print("Recommendation: HIGH RESOURCE configuration")
        return "high"


def apply_resource_config(opt, resource_level=None):
    """Apply resource configuration to options"""
    
    if resource_level is None:
        resource_level = auto_detect_resources()
    
    if resource_level.lower() == "low":
        return LowResourceConfig.apply_config(opt)
    elif resource_level.lower() == "medium":
        return MediumResourceConfig.apply_config(opt)
    elif resource_level.lower() == "high":
        return HighResourceConfig.apply_config(opt)
    else:
        print(f"Unknown resource level: {resource_level}")
        print("Using medium resource configuration as fallback")
        return MediumResourceConfig.apply_config(opt)


# Additional utility functions for monitoring and debugging

def monitor_memory_usage():
    """Monitor and print current memory usage"""
    import torch
    import psutil
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB")
        print(f"GPU Utilization: {(allocated/total)*100:.1f}%")
    
    ram_used = psutil.virtual_memory().used / (1024**3)
    ram_total = psutil.virtual_memory().total / (1024**3)
    print(f"System RAM - Used: {ram_used:.2f}GB, Total: {ram_total:.2f}GB")
    print(f"RAM Utilization: {(ram_used/ram_total)*100:.1f}%")


def cleanup_memory():
    """Clean up GPU and system memory"""
    import torch
    import gc
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleanup completed")


def estimate_training_time(opt, dataset_size):
    """Estimate training time based on configuration"""
    
    # Base time per iteration (seconds) - rough estimates
    base_times = {
        "low": 0.8,     # Smaller models train faster
        "medium": 1.2,   # Medium complexity
        "high": 2.0      # Full model complexity
    }
    
    # Determine resource level
    if opt.fineSize <= 128:
        resource_level = "low"
    elif opt.fineSize <= 192:
        resource_level = "medium"
    else:
        resource_level = "high"
    
    base_time = base_times[resource_level]
    
    # Calculate iterations per epoch
    iters_per_epoch = dataset_size // opt.batchSize
    total_epochs = opt.niter + opt.niter_decay
    total_iterations = iters_per_epoch * total_epochs
    
    # Estimate time
    estimated_seconds = total_iterations * base_time
    estimated_hours = estimated_seconds / 3600
    
    print(f"Training time estimate:")
    print(f"  - Iterations per epoch: {iters_per_epoch}")
    print(f"  - Total epochs: {total_epochs}")
    print(f"  - Total iterations: {total_iterations}")
    print(f"  - Estimated time: {estimated_hours:.1f} hours")
    
    if estimated_hours > 24:
        print(f"  - That's {estimated_hours/24:.1f} days!")
        
    return estimated_hours


# Configuration validation
def validate_config(opt):
    """Validate that the configuration makes sense"""
    issues = []
    
    # Check batch size vs gradient accumulation
    effective_batch_size = opt.batchSize * getattr(opt, 'gradient_accumulation_steps', 1)
    if effective_batch_size < 2:
        issues.append("Effective batch size is very small - may cause training instability")
    
    # Check image sizes
    if opt.fineSize > opt.loadSize:
        issues.append("fineSize should be smaller than loadSize")
    
    # Check model sizes
    if hasattr(opt, 'ngf') and opt.ngf < 16:
        issues.append("Generator features (ngf) is very small - may hurt quality")
    
    # Check learning rate
    if opt.lr > 0.01:
        issues.append("Learning rate seems too high")
    elif opt.lr < 0.00001:
        issues.append("Learning rate seems too low")
    
    if issues:
        print("Configuration validation warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration validation: All checks passed")
    
    return len(issues) == 0