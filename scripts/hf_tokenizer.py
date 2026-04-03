from pathlib import Path
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers import decoders

def hf_bpe_train(filename_prefix, vocab_size):
	input_dir = Path("data")
	file_list = [str(input_dir / f"{filename_prefix}{suffix}.txt") for suffix in ["train", "valid"]]
	print(file_list)
	tokenizer = Tokenizer(BPE(unk_token="<unk>"))
	tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
	tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
	tokenizer.decoder = decoders.ByteLevel()

	special_tokens = ["<|endoftext|>"]
	trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
	tokenizer.train(file_list, trainer)

	output_dir = Path("tokenizers")
	tokenizer.save(str(output_dir / f"{filename_prefix}{vocab_size}_tokenizer.json"))

	with open(output_dir / f"{filename_prefix}{vocab_size}_vocab.json", "w") as f:
		json.dump(tokenizer.get_vocab(), f)

if __name__ == "__main__":
	hf_bpe_train("owt_", 32000)
	hf_bpe_train("TinyStoriesV2-GPT4-", 10000)