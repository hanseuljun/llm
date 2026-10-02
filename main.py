import json

import mlx.core as mx
from mlx import nn


def encode(line: str, vocab: dict[str, int], context_length: int) -> mx.array:
    words = ["<bos>"] + line.split() + ["<eos>"]
    token_ids = mx.array([vocab[word] for word in words])
    return mx.pad(token_ids, (0, context_length - token_ids.shape[0]))

def main():
    with open("data/v5/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)

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

    W_Q = nn.Linear(input_dims=d_model, output_dims=d_k, bias=False)
    W_K = nn.Linear(input_dims=d_model, output_dims=d_k, bias=False)
    W_V = nn.Linear(input_dims=d_model, output_dims=d_v, bias=False)
    W_O = nn.Linear(input_dims=d_v, output_dims=d_model, bias=False)
    Q = W_Q(word_embeds)
    K = W_K(word_embeds)
    V = W_V(word_embeds)
    print(f"Q.shape: {Q.shape}")
    print(f"K.shape: {K.shape}")
    x = Q @ mx.swapaxes(K, -1, -2)
    print(f"x.shape - 1: {x.shape}")
    x /= mx.sqrt(x)
    x = mx.softmax(x, axis=-1)
    x = x @ V
    print(f"x.shape - 2: {x.shape}")
    x = W_O(x)
    print(f"x.shape - 3: {x.shape}")

if __name__ == "__main__":
    main()
