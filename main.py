import json
import random
from typing import cast

import mlx.core as mx
from mlx import nn, optimizers

# Dimensions
# N: size of dataset,
# B: batch size
# C: context length
# E: d_model
# D: d_head
# V: number of vocabs

def encode(line: str, vocab: dict[str, int], context_length: int) -> mx.array:
    words = ["<bos>"] + line.split() + ["<eos>"]
    token_ids = mx.array([vocab[word] for word in words])
    return mx.pad(token_ids, (0, context_length - token_ids.shape[0]))

def decode(token_ids: mx.array, inv_vocab: dict[int, str]) -> list[str]:
    ids = cast(list[int], token_ids.tolist())
    return [inv_vocab[token_id] for token_id in ids]

class AttentionModel(nn.Module):
    def __init__(self, d_model: int, d_head: int, context_length: int, vocab_count: int):
        super().__init__()
        self.d_head = d_head
        self.context_length = context_length
        self.word_embedding = nn.Embedding(num_embeddings=vocab_count, dims=d_model)
        self.position_embedding = nn.Embedding(num_embeddings=context_length, dims=d_model)
        self.W_Q = nn.Linear(input_dims=d_model, output_dims=d_head, bias=False)
        self.W_K = nn.Linear(input_dims=d_model, output_dims=d_head, bias=False)
        self.W_V = nn.Linear(input_dims=d_model, output_dims=d_head, bias=False)
        self.W_O = nn.Linear(input_dims=d_head, output_dims=d_model, bias=False)
        self.linear = nn.Linear(d_model, vocab_count)

    def __call__(self, token_ids_BC: mx.array):
        word_embeds_BCE = self.word_embedding(token_ids_BC)
        position_embed_CE = self.position_embedding(mx.arange(self.context_length)[:-1])
        embeds_BCE = word_embeds_BCE + position_embed_CE
        queries_BCD = self.W_Q(embeds_BCE)
        keys_BCD = self.W_K(embeds_BCE)
        values_BCD = self.W_V(embeds_BCE)
        attention_BCC = queries_BCD @ mx.swapaxes(keys_BCD, -1, -2)
        attention_BCC /= mx.sqrt(mx.array(self.d_head))
        attention_BCC = mx.softmax(attention_BCC, axis=-1)
        x_BCD = attention_BCC @ values_BCD
        x_BCE = self.W_O(x_BCD)
        x_BCV = self.linear(x_BCE)
        return x_BCV


def main():
    with open("data/v5/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)
        inv_vocab = {value: key for key, value in vocab.items()}

    with open("data/v5/train.txt") as train_file:
        train_lines = train_file.readlines()

    with open("data/v5/held_out.txt") as held_out_file:
        held_out_lines = held_out_file.readlines()

    with open("data/v5/held_out_hard.txt") as held_out_hard_file:
        held_out_hard_lines = held_out_hard_file.readlines()

    BATCH_SIZE = 64
    CONTEXT_LENGTH = 32
    LEARNING_RATE = 1e-3
    d_model = 64
    d_head = 16

    mx.set_default_device(mx.cpu)
    random.seed(0)
    mx.random.seed(0)
    model = AttentionModel(d_model=d_model, d_head=d_head, context_length=CONTEXT_LENGTH, vocab_count=len(vocab))
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

    train_ids_NC = mx.stack([encode(line=line, vocab=vocab, context_length=CONTEXT_LENGTH) for line in train_lines])

    def loss_fn(x, target):
        return nn.losses.cross_entropy(model(x), target, reduction="mean")

    def train_fn():
        indices = list(range(train_ids_NC.shape[0]))
        random.shuffle(indices)
        loss_sum = mx.array(0)
        batch_count = train_ids_NC.shape[0] // BATCH_SIZE
        for batch_index in range(batch_count):
            batch_ids_BC = []
            for i in range(batch_index * BATCH_SIZE, (batch_index+1) * BATCH_SIZE):
                index = indices[i]
                batch_ids_BC.append(train_ids_NC[index])
            batch_ids_BC = mx.stack(batch_ids_BC)
            loss, grads = nn.value_and_grad(model, loss_fn)(batch_ids_BC[:, :-1], batch_ids_BC[:, 1:])
            loss_sum += loss
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_sum)
        return loss_sum / batch_count

    for i in range(5):
        loss = train_fn()
        print(f"loss - {i}: {loss}")

if __name__ == "__main__":
    main()
