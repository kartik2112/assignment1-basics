import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm.auto import tqdm


def encode_to_bin(
		tokenizer: Tokenizer,
		ip_path: Path,
		op_path: Path,
		chunk_lines: int = 50000,
		dtype=np.uint16,
):
	fout = open(op_path, 'ab')

	with open(ip_path, encoding='utf-8') as f:
		total_lines = len([None for _ in f])

	with open(ip_path, encoding='utf-8') as f:
		lines = []
		for line in tqdm(f, total=total_lines):
			lines.append(line)
			if len(lines) >= chunk_lines:
				encodings = tokenizer.encode_batch(lines)
				for enc in encodings:
					ids = enc.ids
					if ids:
						arr = np.array(ids, dtype=dtype)
						arr.tofile(fout)
				lines = []

	if lines:
		encodings = tokenizer.encode_batch(lines)
		for enc in encodings:
			ids = enc.ids
			if ids:
				arr = np.array(ids, dtype=dtype)
				arr.tofile(fout)
		lines = []

def tokenize_files(filename_prefix, vocab_size):
	input_dir = Path("data")
	output_dir = Path("data") / "tokenized_data"
	ip_file_list = [str(input_dir / f"{filename_prefix}{suffix}.txt") for suffix in ["train", "valid"]]
	op_file_list = [str(output_dir / f"{filename_prefix}{suffix}_tokenized.bin") for suffix in ["train", "valid"]]
	tokenizer = Tokenizer.from_file(str(Path("tokenizers") / f"{filename_prefix}{vocab_size}_tokenizer.json"))
	for ip_fpath, op_fpath in zip(ip_file_list, op_file_list):
		encode_to_bin(tokenizer, ip_fpath, op_fpath, chunk_lines=1000000)

if __name__ == "__main__":
	tokenize_files("owt_", 32000)
	tokenize_files("TinyStoriesV2-GPT4-", 10000)