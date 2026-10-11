import argparse
import collections
import json
import math
import os
import random
import time

import matplotlib.pyplot as plt
import mlx.core as mx
from mlx import nn, optimizers

# Dimensions
# N: size of dataset,
# B: batch size
# C: context length
# E: d_model
# D: d_head
# V: number of vocabs

EPOCH_COUNT = 100
BATCH_SIZE = 64
MAX_CONTEXT_LENGTH = 32
LEARNING_RATE = 1e-3
D_MODEL = 64
D_HEAD = 16

def encode(line: str, vocab: dict[str, int]) -> list[int]:
    words = ["<bos>"] + line.split() + ["<eos>"]
    return [vocab[word] for word in words]

def pad(token_ids: mx.array, vocab: dict[str, int], target_length: int) -> mx.array:
    return mx.pad(token_ids, (0, target_length - token_ids.shape[0]), constant_values=vocab["<pad>"])

def decode(token_ids: list[int], inv_vocab: dict[int, str]) -> list[str]:
    return [inv_vocab[token_id] for token_id in token_ids]


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.W_Q = nn.Linear(input_dims=d_model, output_dims=d_head, bias=False)
        self.W_K = nn.Linear(input_dims=d_model, output_dims=d_head, bias=False)
        self.W_V = nn.Linear(input_dims=d_model, output_dims=d_head, bias=False)
        self.W_O = nn.Linear(input_dims=d_head, output_dims=d_model, bias=False)

    def __call__(self, embeds_BCE: mx.array, mask_BCC: mx.array):
        queries_BCD = self.W_Q(embeds_BCE)
        keys_BCD = self.W_K(embeds_BCE)
        values_BCD = self.W_V(embeds_BCE)
        attention_BCC = queries_BCD @ mx.swapaxes(keys_BCD, -1, -2)
        attention_BCC /= math.sqrt(self.d_head)
        attention_BCC = mx.softmax(attention_BCC + mask_BCC, axis=-1)
        attended_BCD = attention_BCC @ values_BCD
        attended_BCE = self.W_O(attended_BCD)
        return attended_BCE


class MLP(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.linear1 = nn.Linear(dims, dims)
        self.linear2 = nn.Linear(dims, dims)
        self.linear3 = nn.Linear(dims, dims)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        x = nn.relu(x)
        x = self.linear3(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, d_head: int, max_context_length: int, vocab: dict[str, int]):
        super().__init__()
        vocab_count = len(vocab)
        self.pad_id = vocab["<pad>"]
        self.word_embedding = nn.Embedding(num_embeddings=vocab_count, dims=d_model)
        self.position_embedding = nn.Embedding(num_embeddings=max_context_length, dims=d_model)
        self.attention_layer = AttentionLayer(d_model=d_model, d_head=d_head)
        self.layer_norm1 = nn.LayerNorm(dims=d_model)
        self.mlp = MLP(d_model)
        self.layer_norm2 = nn.LayerNorm(dims=d_model)
        self.linear = nn.Linear(d_model, vocab_count)

    def __call__(self, token_ids_BC: mx.array):
        context_length = token_ids_BC.shape[1]

        word_embeds_BCE = self.word_embedding(token_ids_BC)
        position_embed_CE = self.position_embedding(mx.arange(context_length))
        embeds_BCE = word_embeds_BCE + position_embed_CE

        causal_mask_CC = mx.tri(context_length)
        causal_mask_CC = mx.where(causal_mask_CC, 0, float("-inf"))
        padding_mask_BC = token_ids_BC != self.pad_id
        padding_mask_BC = mx.where(padding_mask_BC, 0, float("-inf"))
        mask_BCC = causal_mask_CC + padding_mask_BC[:, None, :]

        attended1_BCE = self.attention_layer(
            embeds_BCE=embeds_BCE,
            mask_BCC=mask_BCC,
        )
        attended1_BCE = self.layer_norm1(embeds_BCE + attended1_BCE)
        activation1_BCE = self.mlp(attended1_BCE)

        attended2_BCE = self.attention_layer(
            embeds_BCE=activation1_BCE,
            mask_BCC=mask_BCC,
        )
        attended2_BCE = self.layer_norm2(activation1_BCE + attended2_BCE)
        activation2_BCE = self.mlp(attended2_BCE)

        logits_BCV = self.linear(activation2_BCE)
        return logits_BCV


def train():
    with open("data/v5/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)
        vocab["<pad>"] = len(vocab)

    with open("data/v5/train.txt") as train_file:
        train_lines = train_file.readlines()

    with open("data/v5/held_out.txt") as held_out_file:
        held_out_lines = held_out_file.readlines()

    with open("data/v5/held_out_hard.txt") as held_out_hard_file:
        held_out_hard_lines = held_out_hard_file.readlines()

    inv_vocab = {value: key for key, value in vocab.items()}

    mx.set_default_device(mx.cpu)
    random.seed(42)
    mx.random.seed(42)
    model = Transformer(
        d_model=D_MODEL,
        d_head=D_HEAD,
        max_context_length=MAX_CONTEXT_LENGTH,
        vocab=vocab,
    )
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

    train_ids_NC = mx.stack([pad(mx.array(encode(line=line, vocab=vocab)), vocab=vocab, target_length=MAX_CONTEXT_LENGTH) for line in train_lines])

    def loss_fn(token_ids_BC: mx.array, targets_BC: mx.array):
        logits_BCV = model(token_ids_BC)
        losses_BC = nn.losses.cross_entropy(logits_BCV, targets_BC)
        mask_BC = targets_BC != vocab["<pad>"]
        return (losses_BC * mask_BC).sum() / mask_BC.sum()

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

    def eval_fn(lines: list[str]):
        groups = collections.defaultdict(list)
        for line in lines:
            groups[len(line.split())].append(line)
        correct_count = mx.array(0)
        for group in groups.values():
            input_ids = mx.stack([mx.array(encode(line, vocab=vocab)) for line in group])
            output_ids = model(input_ids[:, :-2])
            output_ids = mx.argmax(output_ids[:, -1], axis=-1)
            correct_count += mx.sum(input_ids[:, -2] == output_ids)
            mx.eval(correct_count)
        accuracy = correct_count / len(lines)
        return accuracy

    losses = []
    accuracies = []
    hard_accuracies = []
    for i in range(EPOCH_COUNT):
        start_time = time.perf_counter()
        loss = train_fn()
        losses.append(loss)
        training_accuracy = eval_fn(lines=train_lines)
        validation_accuracy = eval_fn(lines=held_out_lines)
        hard_validation_accuracy = eval_fn(lines=held_out_hard_lines)
        accuracies.append(validation_accuracy)
        hard_accuracies.append(hard_validation_accuracy)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"epoch {i} loss: {loss:.6f}, training accuracy: {training_accuracy:.6f}, validation accuracy: {validation_accuracy:.6f}, hard validation accuracy: {hard_validation_accuracy:.6f}, elapsed time: {elapsed_time:.6f} sec")

    os.makedirs("tmp", exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(losses)
    fig.savefig("tmp/v5-loss.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(accuracies)
    ax.plot(hard_accuracies)
    fig.savefig("tmp/v5-accuracy.png")
    plt.close(fig)

    output_token_ids = [vocab["<bos>"]]
    for _ in range(20):
        output_token_ids_array = mx.array([output_token_ids])
        output_logits = model(output_token_ids_array)
        output_token_id = int(mx.random.categorical(output_logits[0][-1]))
        output_token_ids.append(output_token_id)
        if output_token_id == vocab["<eos>"]:
            break
    output_str = decode(output_token_ids, inv_vocab)
    print(f"output_str: {output_str}")

    model.save_weights("tmp/v5-model_weights.safetensors")


def run(prompt: str):
    print(f"running with prompt: {prompt}")
    with open("data/v5/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)
        vocab["<pad>"] = len(vocab)

    inv_vocab = {value: key for key, value in vocab.items()}

    model = Transformer(
        d_model=D_MODEL,
        d_head=D_HEAD,
        max_context_length=MAX_CONTEXT_LENGTH,
        vocab=vocab,
    )
    model.load_weights("tmp/v5-model_weights.safetensors")

    ids = encode(line=prompt, vocab=vocab)[:-1]
    ids_array = mx.stack([mx.array(ids)])
    print(f"ids_array.shape: {ids_array.shape}")
    print(f"decode: {decode(ids, inv_vocab=inv_vocab)}")
    logits = model(ids_array)
    output_token_id = int(mx.argmax(logits[0][-1]))
    output_str = decode([output_token_id], inv_vocab=inv_vocab)
    print(f"output: {output_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
        return

    run(args.prompt)


if __name__ == "__main__":
    main()
