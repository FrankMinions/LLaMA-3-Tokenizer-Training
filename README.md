# LLaMA-3-Tokenizer-Training

Base on [tiktoken](https://github.com/openai/tiktoken) which is a fast BPE tokeniser, I published ways for training LLaMA-3 tokenizer. You can follow steps below:

Step1. Rename `tokenizer.model` to `tokenizer.tiktoken`.

Step2. Run [train_my_tiktoken.py](https://github.com/FrankMinions/LLaMA-3-Tokenizer-Training/blob/main/train_my_tiktoken.py) to train your own corpus based on tiktoken.

Step3. Run [generate_extra_vocab_freq.py](https://github.com/FrankMinions/LLaMA-3-Tokenizer-Training/blob/main/generate_extra_vocab_freq.py) to generate vocabulary frequency file.

Step4. Run [merge_extra_vocab_freq.py](https://github.com/FrankMinions/LLaMA-3-Tokenizer-Training/blob/main/merge_extra_vocab_freq.py) to generate a new tokenizer model based on the vocabulary frequency file.

Step5. Open the tokenizer model generated in step4, for example you can use `Visual Studio Code`, copy and paste it to the end of the original tokenizer model.

Reference to [Qwen](https://github.com/QwenLM/Qwen) and [tiktoken](https://github.com/openai/tiktoken), and thanks for these open source projects!
