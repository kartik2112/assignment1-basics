import torch
import torch.nn as nn
import math
import einops
from jaxtyping import Float, Int
from functools import lru_cache
from typing import Optional
from collections.abc import Callable, Iterable

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
		self.weight = nn.Parameter(
			torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
		)
		nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3,b=3)
	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		return self.weight[token_ids, :]

class RMSNorm(nn.Module):
	def __init__(self, 
			  d_model: int, 
			  eps: float = 1e-5,
			  device: torch.device | None = None,
			  dtype: torch.dtype | None = None):
		super().__init__()
		self.d_model = d_model
		self.eps = eps
		self.device = device
		self.dtype = dtype
		self.weight = nn.Parameter(
			torch.empty((d_model, ), device=self.device, dtype=self.dtype)
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		in_dtype = x.dtype
		x = x.to(torch.float32)
		# rms_norm = math.sqrt(((x*x).sum()) / self.d_model + self.eps)
		rms_norm = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
		return (x / rms_norm * self.weight).to(in_dtype)

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
		self.w1 = Linear(d_model, self.d_ff)
		self.w3 = Linear(d_model, self.d_ff)
		self.w2 = Linear(self.d_ff, d_model)

	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		temp1 = self.w1(x)
		silu = temp1 * torch.sigmoid(temp1)
		temp2 = silu * (self.w3(x))
		return self.w2(temp2)
	
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
		cos_sin = self.cos_sin[:x.size(-2)] if token_positions is None else self.cos_sin[token_positions]
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

		self.w_qkv = Linear(d_model, 3*d_model)
		self.w_o = Linear(self.d_model, self.d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		L = x.size(1)
		qkv = self.w_qkv(x)
		qkv_mh = einops.rearrange(qkv, 'B L (nH d_h) -> B nH L d_h', d_h=self.d_head)
		Q, K, V = torch.chunk(qkv_mh, 3, dim=1)
		mask = torch.ones((L, L), device=x.device).tril()
		# mask = einops.rearrange(mask, 'seq seq -> 1 1 seq seq')
		out = scaled_dot_product_attention(Q, K, V, mask)
		out = einops.rearrange(out, 'B n_h L d_h -> B L (n_h d_h)')
		return self.w_o(out)
	
class MultiheadSelfAttention_w_RoPE(nn.Module):
	def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_head = d_model // num_heads

		self.rope = RoPE(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len)

		self.w_qkv = Linear(d_model, 3*d_model)
		self.w_o = Linear(self.d_model, self.d_model)

	def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
		L = x.size(1)
		qkv = self.w_qkv(x)
		qkv_mh = einops.rearrange(qkv, 'B L (nH d_h) -> B nH L d_h', d_h=self.d_head)
		Q, K, V = torch.chunk(qkv_mh, 3, dim=1)
		Q = self.rope(Q, token_positions)
		K = self.rope(K, token_positions)
		mask = torch.ones((L, L), device=x.device).tril()
		# mask = einops.rearrange(mask, 'seq seq -> 1 1 seq seq')
		out = scaled_dot_product_attention(Q, K, V, mask)
		out = einops.rearrange(out, 'B n_h L d_h -> B L (n_h d_h)')
		return self.w_o(out)
	
# class TransformerBlock(nn.Module):
# 	def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
# 		self.d_model = d_model
# 		self.num_heads = num_heads
# 		self.d_ff = d_ff
# 		self.mha = MultiheadSelfAttention_w_RoPE(d_model, num_heads, theta, max_seq_len)
# 		self.ffn_w1 = Linear(d_model, d_ff)
# 		self.ffn_w2 = Linear(d_ff, d_model)
# 		self.ffn_w3 = Linear(d_model, d_ff)
# 		self.ffn = nn.Sequential(self.ffn_w1, self.ffn_w2, self.ffn_w3)
# 		self.ln1 = RMSNorm(d_model, eps=1e-5)
# 		self.ln2 = RMSNorm(d_model, eps=1e-5)

# 	def forward(self, x):
# 		norm = self.rms1(x)
# 		mha_out = self.mha(norm) + x

# 		norm2 = self.rms2(mha_out)
# 		return self.ffn(norm2) + x

@lru_cache(1)
def get_rope(d_model, theta, max_seq_len) -> RoPE:
	return RoPE(theta, d_model, max_seq_len)

class TransformerAttention(nn.Module):
	def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		assert self.d_model % self.num_heads == 0
		self.d_head = self.d_model // self.num_heads
		self.theta = theta
		self.max_seq_len = max_seq_len
		self.rope = get_rope(self.d_head, theta, max_seq_len)
		self.q_proj = Linear(d_model, d_model)
		self.k_proj = Linear(d_model, d_model)
		self.v_proj = Linear(d_model, d_model)
		self.output_proj = Linear(d_model, d_model)

	def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
		L = x.size(1)
		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)

		q = einops.rearrange(q, 'B L (nH d_h) -> B nH L d_h', d_h=self.d_head)
		k = einops.rearrange(k, 'B L (nH d_h) -> B nH L d_h', d_h=self.d_head)
		v = einops.rearrange(v, 'B L (nH d_h) -> B nH L d_h', d_h=self.d_head)

		q = self.rope(q, token_positions)
		k = self.rope(k, token_positions)

		mask = torch.ones((L, L), device=x.device).tril()
		out = scaled_dot_product_attention(q, k, v, mask)
		out = einops.rearrange(out, 'B n_h L d_h -> B L (n_h d_h)')
		return self.output_proj(out)
		

class TransformerBlock(nn.Module):
	def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, d_ff: int):
		super().__init__()
		self.d_ff = d_ff
		self.attn = TransformerAttention(d_model, num_heads, theta, max_seq_len)
		self.ln1 = RMSNorm(d_model)
		self.ln2 = RMSNorm(d_model)
		self.ffn = SwiGLU(d_model, d_ff)

	def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
		attn_op = self.attn(self.ln1(x), token_positions) + x

		return self.ffn(self.ln2(attn_op)) + attn_op
	
class TransformerLM(nn.Module):
	def __init__(self, vocab_size: int,
			context_length: int,
			d_model: int,
			num_layers: int,
			num_heads: int,
			d_ff: int,
			rope_theta: float,):
		super().__init__()
		self.token_embeddings = Embedding(vocab_size, d_model)
		self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, rope_theta, context_length, d_ff) for _ in range(num_layers)])
		self.ln_final = RMSNorm(d_model)
		self.lm_head = Linear(d_model, vocab_size)
	
	def forward(self, x_indices: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
		x = self.token_embeddings(x_indices)
		for layer in self.layers:
			x = layer(x, token_positions)
		x = self.ln_final(x)
		return self.lm_head(x)
	
def cross_entropy(o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
	"""
	Interesting implementation:
	Categorical Cross Entropy Loss = neg log likelihood
				= - \sum (y_i) * log(softmax(o_i))
				= - \sum (y_i) * log(exp(o_i) / sum_exp(o))
				= - \sum (y_i) * [log(exp(o_i)) - log(sum_exp(o))]  # log property used
				= - \sum (y_i) * log(exp(o_i)) - \sum (y_i) * log(sum_exp(o))
				# Technically only one y_i will be one, rest will be zero
				= - y_m * log(exp(o_m)) - y_m * log(sum_exp(o))
				= - 1 * o_m - 1 * log(sum_exp(o))
				
	"""
	o = o - o.max(dim=-1, keepdim=True).values
	log_probs = o - torch.logsumexp(o, dim=-1, keepdim=True)
	return -log_probs.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1).mean()

def entropy_chunked(x: torch.Tensor, chunk_size: int=128) -> torch.Tensor:
	num_chunks = math.ceil(x.size(1) / chunk_size)
	chunk_entropies = []
	for i in range(num_chunks):
		chunk_probs = x[:,i*chunk_size:(i+1)*chunk_size,:].softmax(dim=-1)
		chunk_entropies.append((-chunk_probs * chunk_probs.log()).sum(dim=-1))
	return torch.cat(chunk_entropies, dim=1)

class AdamW(torch.optim.Optimizer):
	def __init__(self, params: Iterable[torch.nn.Parameter], lr: float=1e-3, betas: tuple=(0.9, 0.999), eps: float=1e-8, weight_decay: float=1e-5):
		if lr < 0:
			raise "LR cannot be negative"
		defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
		super().__init__(params, defaults)
	
	def step(self, closure: Optional[Callable] = None):
		"""
		m <- B_1 m + (1-B_1)*g
		v <- B_2 m + (1-B_2)*g^2
		alpha_t = alpha * (sqrt(1 - B_1^t))/(1 - B_2^t)
		p = p - alpha_t * (m)/(sqrt(v) + eps)
		p = p - alpha * lambda * p
		"""
		loss = None if closure is None else closure()
		for group in self.param_groups:
			lr = group['lr']
			beta1, beta2 = group['betas']
			eps = group['eps']
			weight_decay = group['weight_decay']

			for p in group["params"]:
				if p.grad is None:
					continue
				grad = p.grad.data
				state = self.state[p]
				if len(state) == 0:
					state["step"] = 0
					state["exp_avg"] = torch.zeros_like(p)
					state["exp_avg_sq"] = torch.zeros_like(p)
				exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
				state["step"] += 1

				p.data.mul_(1 - lr * weight_decay)
				exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
				exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

				bias_correction1 = (1 - beta1 ** state["step"])
				bias_correction2 = (1 - beta2 ** state["step"])

				# p.data -= lr * (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)
				p.data -= lr * (math.sqrt(bias_correction2) / bias_correction1) * exp_avg / (exp_avg_sq.sqrt() + eps)
		return loss
	
def cosine_schedule(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int) -> float:
	if t < t_w:
		return t / t_w * alpha_max
	elif t <= t_c:
		return alpha_min + 1.0 / 2 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (alpha_max - alpha_min)
	else:
		return alpha_min

def grad_norm(parameters: Iterable[nn.Parameter]) -> float:
	total_norm = 0.0
	for p in parameters:
		if p.grad is not None:
			total_norm += p.grad.data.norm(2) ** 2
	return total_norm.sqrt()

def gradient_clipping(parameters: Iterable[nn.Parameter], l2_norm_max: float, eps=1e-6):
	grad = grad_norm(parameters)
	clip_coeff = l2_norm_max / (grad + eps)
	if clip_coeff < 1:
		for p in parameters:
			if p.grad is not None:
				p.grad.data.mul_(clip_coeff)
	return grad