import numpy as np
import torch
from torch.utils.data import Dataset

class BinaryFileDataset(Dataset):
    def __init__(self, file_path, dtype=np.float32, sample_shape=None):
        self.file_path = file_path
        self.dtype = dtype
        self.sample_shape = sample_shape

        # Compute the size of one sample (in bytes)
        sample_size = np.prod(sample_shape) * np.dtype(dtype).itemsize
        # Memory-map the binary file
        self.mem_map = np.memmap(file_path, dtype=dtype, mode='r')
        # Compute the total number of samples in the binary file
        self.num_samples = self.mem_map.size // np.prod(sample_shape)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = np.prod(self.sample_shape) * idx
        end_idx = start_idx + np.prod(self.sample_shape)
        # Read a chunk from the memory-mapped file
        sample_np = self.mem_map[start_idx:end_idx]
        sample_np = sample_np.reshape(self.sample_shape)
        sample_torch = torch.from_numpy(sample_np)
        return {'input': sample_torch}  # No label here; you can add it if needed
