import json
from typing import cast

import mlx.core as mx
from mlx import nn


def encode(line: str, vocab: dict[str, int], context_length: int) -> mx.array:
    words = ["<bos>"] + line.split() + ["<eos>"]
    token_ids = mx.array([vocab[word] for word in words])
    return mx.pad(token_ids, (0, context_length - token_ids.shape[0]))

def decode(token_ids: mx.array, inv_vocab: dict[int, str]) -> list[str]:
    ids = cast(list[int], token_ids.tolist())
    return [inv_vocab[token_id] for token_id in ids]

class AttentionModel(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v):
        self.W_Q = nn.Linear(input_dims=d_model, output_dims=d_k, bias=False)
        self.W_K = nn.Linear(input_dims=d_model, output_dims=d_k, bias=False)
        self.W_V = nn.Linear(input_dims=d_model, output_dims=d_v, bias=False)
        self.W_O = nn.Linear(input_dims=d_v, output_dims=d_model, bias=False)

    def __call__(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        print(f"Q.shape: {Q.shape}")
        print(f"K.shape: {K.shape}")
        x = Q @ mx.swapaxes(K, -1, -2)
        print(f"x.shape - 1: {x.shape}")
        x /= mx.sqrt(x)
        x = mx.softmax(x, axis=-1)
        x = x @ V
        print(f"x.shape - 2: {x.shape}")
        x = self.W_O(x)
        return x


def main():
    with open("data/v5/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)
        inv_vocab = {value: key for key, value in vocab.items()}

    with open("data/v5/train.txt") as train_file:
        train_lines = train_file.readlines()

    # with open("data/v5/held_out.txt") as held_out_file:
    #     held_out_lines = held_out_file.readlines()

    # with open("data/v5/held_out_hard.txt") as held_out_hard_file:
    #     held_out_hard_lines = held_out_hard_file.readlines()

    CONTEXT_LENGTH = 32
    d_model = 64
    d_k = 24
    d_v = 24

    train_lines = train_lines[:2]
    train_lines_token_ids = mx.stack([encode(line=line, vocab=vocab, context_length=CONTEXT_LENGTH) for line in train_lines])

    # print(f"vocab: {vocab}")
    print(f"train_lines: {train_lines}")
    print(f"train_lines_token_ids: {train_lines_token_ids}")
    print(f"train_lines_token_ids.shape: {train_lines_token_ids.shape}")

    mx.random.seed(0)
    word_embedding = nn.Embedding(num_embeddings=len(vocab), dims=d_model)
    word_embeds = word_embedding(train_lines_token_ids)
    print(f"word_embeds.shape: {word_embeds.shape}")

    model = AttentionModel(d_model=d_model, d_k=d_k, d_v=d_v)
    x = model(word_embeds)
    print(f"x.shape - 3: {x.shape}")
    x = mx.argmax(x, axis=2)
    print(f"x.shape - 4: {x.shape}")
    print(f"x: {x}")
    output_line = decode(token_ids=x[0], inv_vocab=inv_vocab)
    print(f"output_line: {output_line}")

if __name__ == "__main__":
    main()
