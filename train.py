import torch
import torch.nn
import numpy as np
from pathlib import Path
import wandb
import pandas as pd
from tqdm.auto import tqdm
from time import time

from cs336_basics.data import get_batch
from cs336_basics.model import cross_entropy, TransformerLM, AdamW, entropy_chunked, cosine_schedule, gradient_clipping
from cs336_basics.generate import generate
from tokenizers import Tokenizer

class Logger:
	def __init__(self, run_name):
		self.logger = wandb.init(project="cs336-assignment1-basics", name=run_name)

	def log_metrics(self, metrics, step):
		self.logger.log(metrics, step)
	
	def log_text(self, key, text, step):
		self.logger.log({key: wandb.Html(text)}, step=step)
	
	def log_table(self, key, table, step):
		self.logger.log({key: wandb.Table(dataframe=pd.DataFrame(table))}, step=step)

@torch.no_grad
def evaluate(model, data_val, eval_iters, batch_size, context_length, device):
	losses = []
	entropies = []
	model.eval()
	for i in tqdm(range(eval_iters)):
		x, y = get_batch(data_val, batch_size, context_length, device)
		logits = model(x)
		loss = cross_entropy(logits, y)
		losses.append(loss.item())
		entropies.append(entropy_chunked(logits).mean().item())
	model.train()
	mean_loss = np.mean(losses)
	return {
		"val/loss": mean_loss.item(),
		"val/ppl": np.exp(mean_loss).item(),
		"val/entropy": np.mean(entropies).item()
	}

def train(
		training_epochs: int,
		batch_size: int,
		vocab_size: int,
		context_length:int,
		d_model:int,
		num_layers:int,
		num_heads:int,
		d_ff:int,
		rope_theta:float,
		tokenizer_path: str,
		data_path: str,
		learning_rate_max: float,
		learning_rate_min: float,
		warmup_iters: int,
		cosine_iters: int,
		run_name: str,
		log_interval: int=10,
		eval_interval: int=50,
		eval_iters: int=200
):
	torch.manual_seed(13)
	device_str = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device_str}")
	device = torch.device(device_str)

	tokenizer = Tokenizer.from_file(tokenizer_path)

	logger = Logger(run_name)

	model = TransformerLM(
		vocab_size=vocab_size, 
		context_length=context_length,
		d_model=d_model,
		num_layers=num_layers,
		num_heads=num_heads,
		d_ff=d_ff, 
		rope_theta=rope_theta)
	model = model.to(device)
	optimizer = AdamW(model.parameters(), lr=learning_rate_min, betas=(0.9, 0.95), weight_decay=0.01)
	data_train = np.memmap(Path(data_path) / "train.bin", dtype=np.uint16, mode='r')
	data_val = np.memmap(Path(data_path) / "val.bin", dtype=np.uint16, mode='r')

	print("Starting training...")

	start = time()

	for it in range(training_epochs):
		x, y = get_batch(data_train, batch_size, context_length, device)
		lr = cosine_schedule(it, learning_rate_max, learning_rate_min, warmup_iters, cosine_iters)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		logits = model(x)
		loss = cross_entropy(logits, y)

		optimizer.zero_grad()
		loss.backward()
		grad_norm = gradient_clipping(model.parameters(), l2_norm_max=1.0)
		optimizer.step()

		if it % log_interval == 0 or it == training_epochs - 1:
			duration = time() - start
			entropy = entropy_chunked(logits)
			logger.log_metrics({
				"train/loss": loss.item(),
				"train/ppl": loss.exp().item(),
				"train/entropy": entropy.mean().item(),
				"train/lr": lr,
				"train/grad_norm": grad_norm
			}, step=it)
			tqdm.write(f"Iter {it}: Train loss: {loss.item():.4f} | LR: {lr:.6f} | {duration:.2f}s")

		if it > 0 and it % eval_interval == 0 or it == training_epochs - 1:
			val_res = evaluate(model, data_val, eval_iters, batch_size, context_length, device)
			logger.log_metrics(val_res, step=it)
			tqdm.write(f"Val loss: {val_res['val/loss']:.4f}")

	# Generation
	context = torch.zeros([1,1], dtype=torch.long, device=device)
	new_text_idxs = generate(
		model=model,
		idx=context,
		max_new_tokens=1000,
		block_size=context_length,
		temperature=0.6,
		top_p=0.95
	)
	generated_text = tokenizer.decode(new_text_idxs[0].tolist())
	tqdm.write("\n---Generated text ---")
	tqdm.write(generated_text)
	logger.log_text("Generated Text", generated_text, step=training_epochs)

if __name__ == "__main__":
	train(
		training_epochs=2000,
		batch_size=64,
		vocab_size=10000,
		context_length=256,
		d_model=512,
		num_layers=4,
		num_heads=16,
		d_ff=1344,
		rope_theta=100000.0,
		tokenizer_path=str(Path("tokenizers") / "TinyStoriesV2-GPT4-10000_tokenizer.json"),
		data_path=str(Path("data") / "tokenized_data" / "TinyStoriesV2-GPT4"),
		learning_rate_max=1e-3,
		learning_rate_min=1e-5,
		warmup_iters=100,
		cosine_iters=900,
		run_name="TinyStories_10000_lr_sweep"
	)

	# experiment 1 - LRs: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2
	lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2]
	for lr in lrs:
		train(
			training_epochs=3000,
			batch_size=64,
			vocab_size=10000,
			context_length=256,
			d_model=512,
			num_layers=4,
			num_heads=16,
			d_ff=1344,
			rope_theta=100000.0,
			tokenizer_path=str(Path("tokenizers") / "TinyStoriesV2-GPT4-10000_tokenizer.json"),
			data_path=str(Path("data") / "tokenized_data" / "TinyStoriesV2-GPT4"),
			learning_rate_max=lr,
			learning_rate_min=lr / 10,
			warmup_iters=500,
			cosine_iters=900,
			run_name=f"TinyStories_10000_lr={lr}"
		)
		