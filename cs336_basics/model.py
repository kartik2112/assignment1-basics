import torch
import torch.nn as nn
import math
import einops

class Linear(nn.Module):
	def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.device | None = None):
		super().__init__()
		kwargs = {'device': device, 'dtype': dtype}
		std = math.sqrt(2 / (in_features + out_features))
		self.weight = nn.Parameter(
			torch.empty((out_features, in_features), **kwargs)
		)
		nn.init.trunc_normal_(self.weight, mean=0, std=std, a = -3 * std, b = 3 * std)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return einops.einsum(self.weight, x, '... out in, ... in -> ... out')