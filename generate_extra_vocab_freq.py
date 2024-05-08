import tiktoken
import unicodedata
from typing import Union, Set, Collection
from tiktoken.load import load_tiktoken_bpe


class Arguments:
    def __init__(self):
        self.name = "llama"
        self.vocab_file = "llama.tiktoken"
        self.text_file = "/path/to/your/*.txt"  # training corpus for bpe
        self.extra_vocab_file = "./vocab_freq.txt"
        self.model_path = "./Meta-Llama-3-8B-Instruct/original/tokenizer.model"


class MyTokenizer:
    """Refer to LLaMA3 tokenizer."""

    def __init__(self, configs):
        self.pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        self.num_reserved_special_tokens = 256
        mergeable_ranks = load_tiktoken_bpe(configs.model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
             "<|begin_of_text|>",
             "<|end_of_text|>",
             "<|reserved_special_token_0|>",
             "<|reserved_special_token_1|>",
             "<|reserved_special_token_2|>",
             "<|reserved_special_token_3|>",
             "<|start_header_id|>",
             "<|end_header_id|>",
             "<|reserved_special_token_4|>",
             "<|eot_id|>",  # end of turn
         ] + [
             f"<|reserved_special_token_{i}|>"
             for i in range(5, self.num_reserved_special_tokens - 5)
         ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        self.mergeable_ranks = load_tiktoken_bpe(configs.vocab_file)
        self.tokenizer = tiktoken.Encoding(
            name=configs.name,
            pat_str=self.pat_str,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens
        )
        self.decoder = {v: k.decode("utf-8", errors="replace") for k, v in self.mergeable_ranks.items()}
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

    def tokenize(self, text: str, allowed_special: Union[Set, str] = "all",
                 disallowed_special: Union[Collection, str] = (), **kwargs):
        tokens = []
        text = unicodedata.normalize("NFC", text)
        for t in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder[t])
        return tokens


if __name__ == '__main__':

    configs = Arguments()
    my_tokenizer = MyTokenizer(configs)

    results = {}
    with open(configs.text_file) as f:
        text = f.read()

    tokens = my_tokenizer.tokenize(text)
    for token in tokens:
        results.setdefault(token, 0)
        results[token] += 1

    with open(configs.extra_vocab_file, mode="w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(k+"\t"+str(v))
            f.write("\n")