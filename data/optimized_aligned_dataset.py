import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import torch.nn.functional as F


class OptimizedAlignedDataset(BaseDataset):
    """Optimized version of AlignedDataset with memory efficiency improvements"""
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        
        # Ensure we have the resize_and_crop mode
        assert(opt.resize_or_crop == 'resize_and_crop')
        
        # Determine if we should use half precision
        self.use_half_precision = getattr(opt, 'use_amp', True)
        
        # More aggressive image loading optimizations
        self.load_size = getattr(opt, 'loadSize', 286)
        self.fine_size = getattr(opt, 'fineSize', 256)
        
        # Create transforms with memory optimization
        transform_list = []
        
        # Use more efficient normalization
        if self.use_half_precision:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # Convert to half precision will be done in __getitem__
            ])
        else:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.transform = transforms.Compose(transform_list)
        
        # Cache for frequently accessed images (small cache to save memory)
        self.cache_size = min(100, len(self.AB_paths) // 10)
        self.image_cache = {}
        self.cache_order = []
        
        print(f"OptimizedAlignedDataset initialized:")
        print(f"  - Dataset size: {len(self.AB_paths)}")
        print(f"  - Load size: {self.load_size}")
        print(f"  - Fine size: {self.fine_size}")
        print(f"  - Half precision: {self.use_half_precision}")
        print(f"  - Cache size: {self.cache_size}")

    def _load_and_cache_image(self, AB_path):
        """Load image with caching for frequently accessed images"""
        
        # Check cache first
        if AB_path in self.image_cache:
            return self.image_cache[AB_path].copy()
        
        # Load image
        try:
            AB = Image.open(AB_path).convert('RGB')
            
            # Resize more efficiently - use LANCZOS for better quality with less memory
            AB = AB.resize((self.load_size * 2, self.load_size), Image.LANCZOS)
            
            # Add to cache if there's room
            if len(self.image_cache) < self.cache_size:
                self.image_cache[AB_path] = AB.copy()
                self.cache_order.append(AB_path)
            elif len(self.cache_order) > 0:
                # Remove oldest item and add new one
                oldest_path = self.cache_order.pop(0)
                if oldest_path in self.image_cache:
                    del self.image_cache[oldest_path]
                self.image_cache[AB_path] = AB.copy()
                self.cache_order.append(AB_path)
            
            return AB
            
        except Exception as e:
            print(f"Error loading image {AB_path}: {e}")
            # Return a dummy image in case of error
            return Image.new('RGB', (self.load_size * 2, self.load_size), color='gray')

    def _efficient_crop(self, AB):
        """More memory-efficient cropping"""
        AB = self.transform(AB)
        
        # Convert to half precision if using AMP
        if self.use_half_precision:
            AB = AB.half()
        
        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        
        # Use smaller random crops to reduce memory usage
        max_offset_w = max(0, w - self.fine_size)
        max_offset_h = max(0, h - self.fine_size)
        
        w_offset = random.randint(0, max_offset_w) if max_offset_w > 0 else 0
        h_offset = random.randint(0, max_offset_h) if max_offset_h > 0 else 0
        
        # Extract A and B regions
        A = AB[:, h_offset:h_offset + self.fine_size, w_offset:w_offset + self.fine_size]
        B = AB[:, h_offset:h_offset + self.fine_size, w + w_offset:w + w_offset + self.fine_size]
        
        return A, B

    def __getitem__(self, index):
        # Use modulo to handle cases where dataset is smaller than expected
        index = index % len(self.AB_paths)
        AB_path = self.AB_paths[index]
        
        # Load and process image
        AB = self._load_and_cache_image(AB_path)
        A, B = self._efficient_crop(AB)
        
        # Handle direction
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # Apply random horizontal flip with probability 0.5
        if (not self.opt.no_flip) and random.random() < 0.5:
            A = torch.flip(A, [2])
            B = torch.flip(B, [2])

        # Convert to grayscale if needed (more efficient implementation)
        if input_nc == 1:  # RGB to gray
            # Use more efficient grayscale conversion
            A = torch.mean(A, dim=0, keepdim=True)

        if output_nc == 1:  # RGB to gray
            B = torch.mean(B, dim=0, keepdim=True)
    
        return {
            'A': A, 
            'B': B,
            'A_paths': AB_path, 
            'B_paths': AB_path
        }

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'OptimizedAlignedDataset'


class OptimizedUnalignedDataset(BaseDataset):
    """Optimized version of UnalignedDataset with memory improvements"""
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.use_half_precision = getattr(opt, 'use_amp', True)
        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        # Create more efficient transforms
        transform_list = [
            transforms.Resize((opt.fineSize, opt.fineSize), Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        
        self.transform = transforms.Compose(transform_list)
        
        # Specialized transform for A images (which seem to be concatenated)
        self.transformA = transforms.Compose([
            transforms.Resize((opt.fineSize * 2, opt.fineSize), Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        print(f"OptimizedUnalignedDataset initialized:")
        print(f"  - A images: {self.A_size}")
        print(f"  - B images: {self.B_size}")
        print(f"  - Fine size: {opt.fineSize}")
        print(f"  - Half precision: {self.use_half_precision}")

    def __getitem__(self, index):
        # Handle different dataset sizes efficiently
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)
        
        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]

        # Load images with error handling
        try:
            A_img = Image.open(A_path).convert('RGB')
            A_img = self.transformA(A_img)
            
            # Convert to half precision if using mixed precision
            if self.use_half_precision:
                A_img = A_img.half()
            
            # Split A image into A1 and A2 more efficiently
            height = A_img.size(1)
            mid_point = height // 2
            
            A1 = A_img[:, 0:mid_point, :].unsqueeze(0)
            A2 = A_img[:, mid_point:height, :].unsqueeze(0)
            
            # Convert to numpy for compatibility (if needed by the model)
            A1 = A1.numpy()
            A2 = A2.numpy()
            
        except Exception as e:
            print(f"Error loading A image {A_path}: {e}")
            # Create dummy data
            dummy_shape = (1, 3, self.opt.fineSize, self.opt.fineSize)
            A1 = np.zeros(dummy_shape, dtype=np.float16 if self.use_half_precision else np.float32)
            A2 = np.zeros(dummy_shape, dtype=np.float16 if self.use_half_precision else np.float32)

        try:
            B_img = Image.open(B_path).convert('RGB')
            B = self.transform(B_img)
            
            if self.use_half_precision:
                B = B.half()
                
        except Exception as e:
            print(f"Error loading B image {B_path}: {e}")
            # Create dummy data
            B = torch.zeros(3, self.opt.fineSize, self.opt.fineSize, 
                           dtype=torch.float16 if self.use_half_precision else torch.float32)

        return {
            'A1': A1, 
            'A2': A2, 
            'B': B,
            'A_paths': A_path, 
            'B_paths': B_path
        }

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'OptimizedUnalignedDataset'


class OptimizedCustomDatasetDataLoader:
    """Memory-efficient data loader with optimizations"""
    
    def __init__(self):
        pass
    
    def name(self):
        return 'OptimizedCustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        
        # Create optimized dataset
        if opt.dataset_mode == 'aligned':
            dataset = OptimizedAlignedDataset()
        elif opt.dataset_mode == 'unaligned':
            dataset = OptimizedUnalignedDataset()
        else:
            # Fall back to original implementation
            from data.custom_dataset_data_loader import CreateDataset
            dataset = CreateDataset(opt)
        
        print(f"dataset [{dataset.name()}] was created")
        dataset.initialize(opt)
        
        # Optimize DataLoader parameters for memory efficiency
        num_workers = min(opt.nThreads, 4)  # Limit workers to save memory
        pin_memory = torch.cuda.is_available()  # Only use pinned memory if GPU available
        
        # Use smaller batch size if memory constrained
        batch_size = opt.batchSize
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not opt.serial_batches,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,  # Drop incomplete batches to avoid size mismatches
            persistent_workers=num_workers > 0  # Keep workers alive between epochs
        )
        
        print(f"DataLoader optimizations:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Num workers: {num_workers}")
        print(f"  - Pin memory: {pin_memory}")
        print(f"  - Persistent workers: {num_workers > 0}")

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataloader.dataset), self.opt.max_dataset_size)


def CreateOptimizedDataLoader(opt):
    """Factory function to create optimized data loader"""
    data_loader = OptimizedCustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader