import json
import math
import random

import mlx.core as mx
from mlx import nn, optimizers

# Dimensions
# N: size of dataset,
# B: batch size
# C: context length
# E: d_model
# D: d_head
# V: number of vocabs

def encode(line: str, vocab: dict[str, int], target_length: int) -> mx.array:
    words = ["<bos>"] + line.split() + ["<eos>"]
    token_ids = mx.array([vocab[word] for word in words])
    return mx.pad(token_ids, (0, target_length - token_ids.shape[0]), constant_values=vocab["<pad>"])

def decode(token_ids: list[int], inv_vocab: dict[int, str]) -> list[str]:
    return [inv_vocab[token_id] for token_id in token_ids]

class AttentionModel(nn.Module):
    def __init__(self, d_model: int, d_head: int, max_context_length: int, vocab_count: int):
        super().__init__()
        self.d_head = d_head
        self.word_embedding = nn.Embedding(num_embeddings=vocab_count, dims=d_model)
        self.position_embedding = nn.Embedding(num_embeddings=max_context_length, dims=d_model)
        self.W_Q = nn.Linear(input_dims=d_model, output_dims=d_head, bias=False)
        self.W_K = nn.Linear(input_dims=d_model, output_dims=d_head, bias=False)
        self.W_V = nn.Linear(input_dims=d_model, output_dims=d_head, bias=False)
        self.W_O = nn.Linear(input_dims=d_head, output_dims=d_model, bias=False)
        self.linear = nn.Linear(d_model, vocab_count)

    def __call__(self, token_ids_BC: mx.array):
        context_length = token_ids_BC.shape[1]
        word_embeds_BCE = self.word_embedding(token_ids_BC)
        position_embed_CE = self.position_embedding(mx.arange(context_length))
        embeds_BCE = word_embeds_BCE + position_embed_CE
        queries_BCD = self.W_Q(embeds_BCE)
        keys_BCD = self.W_K(embeds_BCE)
        values_BCD = self.W_V(embeds_BCE)
        attention_BCC = queries_BCD @ mx.swapaxes(keys_BCD, -1, -2)
        attention_BCC /= math.sqrt(self.d_head)
        causal_mask = mx.tri(context_length)
        causal_mask = mx.where(causal_mask, 0, float("-inf"))
        attention_BCC = mx.softmax(attention_BCC + causal_mask, axis=-1)
        attended_BCD = attention_BCC @ values_BCD
        attended_BCE = self.W_O(attended_BCD)
        logits_BCV = self.linear(attended_BCE)
        return logits_BCV


def main():
    with open("data/v5/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)
        vocab["<pad>"] = len(vocab)
        inv_vocab = {value: key for key, value in vocab.items()}

    with open("data/v5/train.txt") as train_file:
        train_lines = train_file.readlines()

    with open("data/v5/held_out.txt") as held_out_file:
        held_out_lines = held_out_file.readlines()

    with open("data/v5/held_out_hard.txt") as held_out_hard_file:
        held_out_hard_lines = held_out_hard_file.readlines()

    BATCH_SIZE = 64
    MAX_CONTEXT_LENGTH = 32
    LEARNING_RATE = 1e-3
    d_model = 64
    d_head = 16

    mx.set_default_device(mx.cpu)
    random.seed(0)
    mx.random.seed(0)
    model = AttentionModel(d_model=d_model, d_head=d_head, max_context_length=MAX_CONTEXT_LENGTH, vocab_count=len(vocab))
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

    train_ids_NC = mx.stack([encode(line=line, vocab=vocab, target_length=MAX_CONTEXT_LENGTH) for line in train_lines])

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

    output_token_ids = [vocab["<bos>"]]
    for _ in range(20):
        print(f"output_token_ids: {output_token_ids}")
        output_token_ids_array = mx.array([output_token_ids])
        print(f"output_token_ids_array.shape: {output_token_ids_array.shape}")
        output_logits = model(output_token_ids_array)
        print(f"output_logits.shape: {output_logits.shape}")
        output_token_id = int(mx.random.categorical(output_logits[0][-1]))
        output_token_ids.append(output_token_id)
        if output_token_id == vocab["<eos>"]:
            break
    output_str = decode(output_token_ids, inv_vocab)
    print(f"output_str: {output_str}")


if __name__ == "__main__":
    main()
