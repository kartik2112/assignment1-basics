import torch
import os
import torch.nn as nn
import typing

def save_checkpoint(model: nn.Module, 
					optimizer: torch.optim.Optimizer, 
					iteration: int, 
					out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
	if isinstance(optimizer, list):
		optim_states = [opt.state_dict() for opt in optimizer]
	else:
		optim_states = optimizer.state_dict()
	checkpoint = {
		'model': model.state_dict(),
		'optimizer': optim_states,
		'iteration': iteration
	}
	torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
					model: nn.Module, 
					optimizer: torch.optim.Optimizer) -> int:
	checkpoint = torch.load(src)
	if isinstance(optimizer, list):
		for opt, state in zip(optimizer, checkpoint['optimizer']):
			opt.load_state_dict(state)
	else:
		optimizer.load_state_dict(checkpoint['optimizer'])
	model.load_state_dict(checkpoint['model'])
	return checkpoint['iteration']