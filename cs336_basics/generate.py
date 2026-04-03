import torch
import torch.nn as nn

from cs336_basics.model import TransformerLM, softmax

def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
	sorted_probs, sorted_indices = torch.sort(probs, descending=True)
	cumsum_probs = sorted_probs.cumsum(dim=-1)
	mask = cumsum_probs - sorted_probs > top_p
	sorted_probs[mask] = 0.0
	
	# Re-normalize
	sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
	next_token = torch.multinomial(sorted_probs, num_samples=1)
	return torch.gather(sorted_indices, dim=-1, index=next_token)


def generate(
		model: TransformerLM,
		idx: torch.Tensor,
		max_new_tokens: int,
		block_size: int = None,
		temperature: float = 1.0,
		top_p: float = 1.0
):
	for i in range(max_new_tokens):
		idx_cond = idx[:,-block_size:] if block_size is not None else idx
		logits = model(idx_cond)

		logits = logits[:,-1,:] # last timestep for token generation

		if temperature == 0:
			idx_next = torch.argmax(logits, dim=-1, keepdim=True)
		else:
			logits /= temperature
			probs = softmax(logits, dim=-1)
			idx_next = sample_top_p(probs, top_p)

		idx = torch.cat([idx, idx_next], dim=1)
	return idx