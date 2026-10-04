import json
import random
from typing import cast

import mlx.core as mx
from mlx import nn, optimizers

# Dimensions
# N: size of dataset,
# B: batch size
# C: context length
# D: d_model
# K: d_k
# V: d_v

def encode(line: str, vocab: dict[str, int], context_length: int) -> mx.array:
    words = ["<bos>"] + line.split() + ["<eos>"]
    token_ids = mx.array([vocab[word] for word in words])
    return mx.pad(token_ids, (0, context_length - token_ids.shape[0]))

def decode(token_ids: mx.array, inv_vocab: dict[int, str]) -> list[str]:
    ids = cast(list[int], token_ids.tolist())
    return [inv_vocab[token_id] for token_id in ids]

class AttentionModel(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, vocab_count: int):
        super().__init__()
        self.d_k = d_k
        self.W_Q = nn.Linear(input_dims=d_model, output_dims=d_k, bias=False)
        self.W_K = nn.Linear(input_dims=d_model, output_dims=d_k, bias=False)
        self.W_V = nn.Linear(input_dims=d_model, output_dims=d_v, bias=False)
        self.W_O = nn.Linear(input_dims=d_v, output_dims=d_model, bias=False)
        self.linear = nn.Linear(d_model, vocab_count)

    def __call__(self, x):
        # x's dimensions: 
        # Q's dimensions: ()
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # print(f"Q.shape: {Q.shape}")
        # print(f"K.shape: {K.shape}")
        x = Q @ mx.swapaxes(K, -1, -2)
        # print(f"x.shape - 1: {x.shape}")
        x /= mx.sqrt(mx.array(self.d_k))
        x = mx.softmax(x, axis=-1)
        x = x @ V
        # print(f"x.shape - 2: {x.shape}")
        x = self.W_O(x)
        return self.linear(x)


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

    BATCH_SIZE = 64
    CONTEXT_LENGTH = 32
    LEARNING_RATE = 0.5
    d_model = 64
    d_k = 24
    d_v = 24

    # train_lines = train_lines[:2]
    # train_lines_token_ids dims: (# dataset lines, context length)
    train_lines_token_ids_NC = mx.stack([encode(line=line, vocab=vocab, context_length=CONTEXT_LENGTH) for line in train_lines])

    # print(f"vocab: {vocab}")
    # print(f"train_lines: {train_lines}")
    # print(f"train_lines_token_ids: {train_lines_token_ids}")
    print(f"train_lines_token_ids_NC.shape: {train_lines_token_ids_NC.shape}")

    mx.random.seed(0)
    model = AttentionModel(d_model=d_model, d_k=d_k, d_v=d_v, vocab_count=len(vocab))
    optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

    word_embedding = nn.Embedding(num_embeddings=len(vocab), dims=d_model)
    # word_embeds's dims: (# dataset lines, context length, d_model)
    word_embeds = word_embedding(train_lines_token_ids_NC)
    print(f"word_embeds.shape: {word_embeds.shape}")

    def loss_fn(x, target):
        return nn.losses.cross_entropy(model(x), target, reduction="mean")

    def train_fn():
        indices = list(range(word_embeds.shape[0]))
        random.shuffle(indices)
        loss_sum = mx.array(0)
        batch_count = word_embeds.shape[0] // BATCH_SIZE
        for batch_index in range(batch_count):
            batch_word_embeds = []
            batch_train_lines_token_ids = []
            for i in range(batch_index * BATCH_SIZE, (batch_index+1) * BATCH_SIZE):
                index = indices[i]
                batch_word_embeds.append(word_embeds[index])
                batch_train_lines_token_ids.append(train_lines_token_ids_NC[index])
            # batch_word_embeds's dims: (batch size, context length, d_model)
            batch_word_embeds = mx.stack(batch_word_embeds)
            # batch_train_lines_token_ids's dims: (batch size, context length)
            batch_train_lines_token_ids = mx.stack(batch_train_lines_token_ids)
            loss, grads = nn.value_and_grad(model, loss_fn)(batch_word_embeds[:, :-1], batch_train_lines_token_ids[:, 1:])
            loss_sum += loss
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_sum)
        return loss_sum / batch_count

    for i in range(5):
        loss = train_fn()
        print(f"loss - {i}: {loss}")

if __name__ == "__main__":
    main()
