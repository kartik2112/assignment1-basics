import torch
import torch.nn as nn
import math
import einops

class Linear(nn.Module):
	def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
		super().__init__()
		kwargs = {'device': device, 'dtype': dtype}
		std = math.sqrt(2 / (in_features + out_features))
		self.weight = nn.Parameter(
			torch.empty((out_features, in_features), **kwargs)
		)
		nn.init.trunc_normal_(self.weight, mean=0, std=std, a = -3 * std, b = 3 * std)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return einops.einsum(self.weight, x, '... out in, ... in -> ... out')
	
class Embedding(nn.Module):
	def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
		super().__init__()
		self.embedding = nn.Parameter(
			torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
		)
		nn.init.trunc_normal_(self.embedding, mean=0, std=1, a=-3,b=3)
	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		return self.embedding[token_ids, :]

class RMSNorm(nn.Module):
	def __init__(self, 
			  d_model: int, 
			  eps: float,
			  device: torch.device | None = None,
			  dtype: torch.dtype | None = None):
		super().__init__()
		self.d_model = d_model
		self.eps = eps
		self.device = device
		self.dtype = dtype
		self.gain = nn.Parameter(
			torch.empty((d_model, ), device=self.device, dtype=self.dtype)
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		in_dtype = x.dtype
		x = x.to(torch.float32)
		# rms_norm = math.sqrt(((x*x).sum()) / self.d_model + self.eps)
		rms_norm = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
		return (x / rms_norm * self.gain).to(in_dtype)