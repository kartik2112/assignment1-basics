import pickle
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_GPT2_SPLIT_REGEX = re.compile(PAT)

class Tokenizer:
	def __init__(self, 
			  	vocab: dict[int, bytes], 
				merges: list[tuple[bytes, bytes]], 
				special_tokens: list[str] | None = None):
		self.vocab = vocab
		self.vocab_reversed = {b:i for i, b in self.vocab.items()}
		self.merges = merges
		self.merge_ranks = {merge: i for i, merge in enumerate(self.merges)}
		self.special_tokens = sorted(special_tokens, key=lambda x: -len(x)) if special_tokens is not None else None
		
	@classmethod
	def from_files(cls, 
				vocab_filepath: str, 
				merges_filepath: str, 
				special_tokens: list[str] | None = None):
		vocab = pickle.load(open(vocab_filepath, 'rb'))
		merges = pickle.load(open(merges_filepath, 'rb'))
		return cls(vocab, merges, special_tokens)
	
	def _fetch_pre_tokens(self, text):
		if self.special_tokens is not None:
			special_toks = "|".join([re.escape(special_tok) for special_tok in self.special_tokens])
			special_toks = f"({special_toks})"
			segments = re.split(special_toks, text)
		else:
			segments = [text]
		pre_token_bytes: list[list[bytes]] = []
		for segment in segments:
			if self.special_tokens and segment in self.special_tokens:
				pre_token_bytes.append([segment.encode('utf-8')])
			else:
				toks = [match.group(0).encode('utf-8') for match in re.finditer(_GPT2_SPLIT_REGEX, segment)]
				for tok in toks:
					pre_token_bytes.append([bytes([b]) for b in tok])
		return pre_token_bytes
	
	def _merge_pairs(self, pre_tokens, merge_pair):
		new_pre_tokens = []
		i = 0
		while i < len(pre_tokens):
			if i < len(pre_tokens) - 1 and (pre_tokens[i], pre_tokens[i + 1]) == merge_pair:
				new_pre_tokens.append(pre_tokens[i]+pre_tokens[i+1])
				i += 2
			else:
				new_pre_tokens.append(pre_tokens[i])
				i += 1
		return new_pre_tokens

	def encode(self, text: str) -> list[int]:
		pre_token_bytes = self._fetch_pre_tokens(text)
		tokens = []
		for pre_token in pre_token_bytes:
			if len(pre_token) == 1 and pre_token[0] in self.vocab_reversed:
				token = self.vocab_reversed.get(pre_token[0])
				if token is not None:
					tokens.append(token)
				continue
			
			while len(pre_token) >= 2:
				def get_merge_rank(pair):
					return self.merge_ranks.get(pair, float('inf'))
				pairs = [(elem1, elem2) for elem1, elem2 in zip(pre_token, pre_token[1:])]
				min_pair = min(pairs, key=get_merge_rank)
				if min_pair not in self.merge_ranks:
					break
				pre_token = self._merge_pairs(pre_token, min_pair)

			for pre_t in pre_token:
				tokens.append(self.vocab_reversed.get(pre_t))
		return tokens


	def encode_iterable(self, iterable: list[str]) -> iter:
		for text in iterable:
			token_ids =  self.encode(text)
			yield from token_ids

	def decode(self, ids: list[int]) -> str:
		tokens = b"".join([self.vocab.get(id, b'\xefbfbd') for id in ids])
		return tokens.decode('utf-8', errors='ignore')
