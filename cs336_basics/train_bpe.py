import regex as re
import pickle
from typing import BinaryIO
from multiprocessing import get_context, Pool
from collections import defaultdict
import os
from tqdm.auto import tqdm

# from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_GPT2_SPLIT_REGEX = re.compile(PAT)

def train_bpe(input_path: str | os.PathLike,
			  vocab_size: int,
			  special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
	NUM_PROCESSES = 4
	vocab = {i:bytes([i]) for i in range(256)}
	for special_tok in special_tokens:
		vocab[len(vocab)] = special_tok.encode('utf-8')
	merges: list[tuple[bytes, bytes]] = []

	# Create chunks
	with open(input_path, 'rb') as f:
		boundaries = find_chunk_boundaries(f, NUM_PROCESSES, "<|endoftext|>".encode('utf-8'))

	special_tokens_split = "|".join([re.escape(tok) for tok in special_tokens])
	# special_tokens_split = f"({special_tokens_split})"

	# Get pre-tokens for each chunk using multiprocessing
	task_args = []
	for start, end in zip(boundaries[:-1], boundaries[1:]):
		task_args.append([input_path, start, end, special_tokens_split])
	with get_context('forkserver').Pool(processes=NUM_PROCESSES) as pool:
		chunk_results = pool.map(process_chunk, task_args) # list[list[list[int]]]
	ids: list[list[int]] = [token_ids for chunk_ids in chunk_results for token_ids in chunk_ids]

	# Determine pair indices and pair counts
	pair_to_indices, counts = find_pair_counts(ids)

	# Determine number of merges
	num_merges = vocab_size - len(vocab)

	# Iteratively get highest rank byte pair based on count and create new index
	for _ in tqdm(range(num_merges)):
		def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
			return counts[pair], (vocab[pair[0]], vocab[pair[1]])
		
		max_pair = max(counts, key=rank)
		new_id = len(vocab)
		new_token = vocab[max_pair[0]] + vocab[max_pair[1]]
		vocab[new_id] = new_token

		# Iteratively look at each token_ids where max_pair was referenced
		affected_indices = pair_to_indices[max_pair].copy()
		for j in affected_indices:
			token_ids = ids[j]

			if len(token_ids) < 2:
				continue

			# Ignore all counts for this token_ids list
			for pair in zip(token_ids, token_ids[1:]):
				counts[pair] -= 1
				pair_to_indices[pair].discard(j)

				if counts[pair] == 0:
					del counts[pair]
					del pair_to_indices[pair]
			
			# Create list of new_token_ids after merged token
			new_token_ids = _merge_tokens(token_ids, max_pair, new_id)

			# Add back new references and counts
			for pair in zip(new_token_ids, new_token_ids[1:]):
				counts[pair] += 1
				pair_to_indices[pair].add(j)

			ids[j] = new_token_ids
		merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
	return vocab, merges


def process_chunk(args: tuple[str, int, int, str]) -> list[list[int]]:
	input_path, start, end, special_toks = args
	with open(input_path, 'rb') as f:
		f.seek(start)
		chunk = f.read(end-start).decode('utf-8', errors='ignore')

	documents = re.split(special_toks, chunk)
	chunk_ids = []
	for doc in documents:
		toks = [match.group(0).encode('utf-8') for match in re.finditer(_GPT2_SPLIT_REGEX, doc)]
		chunk_ids.extend([list(tok) for tok in toks])
	return chunk_ids

def _merge_tokens(token_ids: list[int], max_pair: tuple[int, int], new_id: int) -> list[int]:
	i = 0
	new_token_ids = []
	while i < len(token_ids):
		if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == max_pair:
			new_token_ids.append(new_id)
			i += 2
		else:
			new_token_ids.append(token_ids[i])
			i += 1
	return new_token_ids

def find_pair_counts(ids: list[list[int]]) -> tuple[
	dict[tuple[int, int], set],
	dict[tuple[int, int], int]
]:
	pair_to_indices = defaultdict(set)
	counts = defaultdict(int)

	for i, token_ids in enumerate(ids):
		for pair in zip(token_ids[:-1], token_ids[1:]):
			pair_to_indices[pair].add(i)
			counts[pair] += 1

	return pair_to_indices, counts

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


if __name__ == "__main__":
	vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10_000, ["<|endoftext|>"])
	pickle.dump(vocab, open("artifacts/vocab_tinystories_train.pkl", "wb"))
	pickle.dump(merges, open("artifacts/merges_tinystories_train.pkl", "wb"))
