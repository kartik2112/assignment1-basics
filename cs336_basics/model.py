import torch
import torch.nn as nn
import math
import einops
from jaxtyping import Float, Int

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

class SwiGLU(nn.Module):
	def __init__(self, 
			d_model: int,
			d_ff: int,
			device: torch.device | None = None, 
			dtype: torch.dtype | None = None):
		super().__init__()
		self.device = device
		self.dtype = dtype
		self.d_ff = d_ff
		self.w1 = nn.Parameter(
			torch.empty((self.d_ff, d_model), device=self.device, dtype=self.dtype)
		)
		self.w3 = nn.Parameter(
			torch.empty((self.d_ff, d_model), device=self.device, dtype=self.dtype)
		)
		self.w2 = nn.Parameter(
			torch.empty((d_model, self.d_ff), device=self.device, dtype=self.dtype)
		)

	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		temp1 = einops.einsum(self.w1, x, 'd_ff d_model, ... d_model -> ... d_ff')
		silu = temp1 * torch.sigmoid(temp1)
		temp2 = silu * (x @ self.w3.T)
		return einops.einsum(self.w2, temp2, 'd_model d_ff, ... d_ff -> ... d_model')
	
class RoPE(nn.Module):
	def __init__(self, 
			  theta: float,
			  d_k: int,
			  max_seq_len: int,
			  device: torch.device | None = None):
		super().__init__()

		self.theta = theta
		self.d_k = d_k
		self.max_seq_len = max_seq_len
		self.device = device
		self.register_buffer('cos_sin', 
					   pre_compute_cis(self.theta, self.d_k, self.max_seq_len),
					   persistent=False)

	def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Int[torch.Tensor, "... seq_len"]) -> torch.Tensor:
		cos_sin = self.cos_sin[x.size(-2):] if token_positions is None else self.cos_sin[token_positions]
		return apply_rotary_emb(x, cos_sin)

def pre_compute_cis(theta, max_head, max_seq_len):
	freqs = 1.0 / theta ** (torch.arange(0, max_head, 2) / max_head).float()
	seq = torch.arange(max_seq_len, device=freqs.device)
	freqs = einops.einsum(seq, freqs, 'i, j -> i j')
	complex = torch.polar(torch.ones_like(freqs), freqs)
	return torch.cat([complex.real, complex.imag], dim=-1)

def apply_rotary_emb(x, cos_sin):
	x1, x2 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
	cos, sin = torch.chunk(cos_sin, 2, dim=-1)
	out = torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
	return out.reshape(*x.shape).type_as(x)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
	x_exp = torch.exp(x - x.max(dim=dim, keepdim=True).values)
	return x_exp / x_exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
	scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(K.size(-1))
	if mask is not None:
		scores = scores.masked_fill(mask == 0, float('-inf'))
	attn_weights = softmax(scores, dim=-1)
	return torch.matmul(attn_weights, V)

class MultiheadSelfAttention(nn.Module):
	def __init__(self, d_model: int, num_heads: int):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_head = d_model // num_heads

		self.w_qkv = nn.Linear(d_model, 3*d_model)
		self.w_o = nn.Linear(self.d_model, self.d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		L = x.size(-2)
		qkv = self.w_qkv(x)
		qkv_mh = einops.rearrange(qkv, 'B L (nH d_h) -> B nH L d_h')
		Q, K, V = torch.chunk(qkv_mh, 3, dim=-1)
		mask = torch.ones((L, L)).tril()
		out = scaled_dot_product_attention(Q, K, V, mask)
		out = einops.rearrange(out, 'B n_h L d_h -> B L (n_h d_h)')
		return self.w_o(out)