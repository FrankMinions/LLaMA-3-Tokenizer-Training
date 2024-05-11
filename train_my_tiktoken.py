import base64
from tiktoken._educational import SimpleBytePairEncoding

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Specify your training corpus
with open("/path/to/your/*.txt") as f:
    training_data = f.read()

simply_bpe = SimpleBytePairEncoding.train(training_data=training_data, vocab_size=16000, pat_str=PAT_STR)

with open("/path/to/your/llama.tiktoken", mode="w", encoding="utf-8") as f:
    for k, v in simply_bpe.mergeable_ranks.items():
        line = base64.b64encode(k).decode("utf-8") + " " + str(v) + "\n"
        f.write(line)
