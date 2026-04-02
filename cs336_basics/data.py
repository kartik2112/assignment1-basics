import numpy as np
import torch
from numpy.typing import NDArray
from typing import List, Tuple

def get_batch(x: NDArray, batch_size: int, context_length: int, device: str) -> Tuple[torch.tensor, torch.Tensor]:
	start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)
	window_indices = np.arange(context_length + 1)
	block_indices = start_indices[:,None] + window_indices
	batch = torch.from_numpy(x[block_indices].astype(np.int64))
	x = batch[:,:-1]
	y = batch[:,1:]
	return x.to(device), y.to(device)