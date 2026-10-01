import json
import os
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlx.core as mx
from mlx import nn, optimizers


@dataclass
class QAPair:
    question: list[int]
    answer: int
    is_last_noun: bool


def encode(line: str, vocab: dict[str, int]) -> list[int]:
    words = ["<bos>"] + line.split() + ["<eos>"]
    return [vocab[word] for word in words]

def create_bow_qa_pairs(lines_token_ids: list[list[int]]) -> list[QAPair]:
    def convert_token_ids_to_bow_qa_pairs(token_ids: list[int]) -> list[QAPair]:
        return [QAPair(question=token_ids[:i], answer=token_ids[i], is_last_noun=i==(len(token_ids)-2)) for i in range(1, len(token_ids))]

    qa_pairs = []
    for token_ids in lines_token_ids:
        qa_pairs += convert_token_ids_to_bow_qa_pairs(token_ids=token_ids)
    return qa_pairs

def decode(token_ids: list[int], inv_vocab: dict[int, str]) -> list[str]:
    return [inv_vocab[token_id] for token_id in token_ids]

class AttentionModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, dims=embed_dim)
        # self.position_embedding = nn.Embedding(num_embeddings=context_length, dims=embed_dim)
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.layers = [
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, vocab_size),
        ]

    def __call__(self, x):
        padded_x = mx.array([token_ids + [0] * (self.context_length - len(token_ids)) for token_ids in x])
        # lengths = [len(token_ids) for token_ids in x]

        # B: batch size, C: context length, E: embed_dim
        # word_embeds_BCE = self.word_embedding(padded_x)
        # position_embed_CE = self.position_embedding(mx.arange(self.context_length))
        # masks_BC = mx.array([[1] * length + [0] * (self.context_length - length) for length in lengths])

        # x = mx.sum(word_embeds_BCE * position_embed_CE * masks_BC[:, :, None], axis=1) / mx.sum(masks_BC, axis=1)[:, None]

        word_embeds_BCE = self.word_embedding(padded_x)
        q_BCE = self.wq(word_embeds_BCE)
        k_BCE = self.wk(word_embeds_BCE)
        v_BCE = self.wv(word_embeds_BCE)
        x = q_BCE @ mx.transpose(k_BCE, [0, 2, 1]) @ v_BCE
        x = mx.sum(x, axis=1)

        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)

def run_bow():
    with open("data/v5/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)

    with open("data/v5/train.txt") as train_file:
        train_lines = train_file.readlines()

    with open("data/v5/held_out.txt") as held_out_file:
        held_out_lines = held_out_file.readlines()

    with open("data/v5/held_out_hard.txt") as held_out_hard_file:
        held_out_hard_lines = held_out_hard_file.readlines()

    inv_vocab = {value: key for key, value in vocab.items()}

    train_lines_token_ids = [encode(line, vocab=vocab) for line in train_lines]
    print(f"train_lines_token_ids: {len(train_lines_token_ids)}")

    held_out_lines_token_ids = [encode(line, vocab=vocab) for line in held_out_lines]
    print(f"held_out_lines_token_ids: {len(held_out_lines_token_ids)}")

    held_out_hard_lines_token_ids = [encode(line, vocab=vocab) for line in held_out_hard_lines]
    print(f"held_out_hard_lines_token_ids: {len(held_out_hard_lines_token_ids)}")

    train_qa_pairs = create_bow_qa_pairs(lines_token_ids=train_lines_token_ids)
    held_out_qa_pairs = create_bow_qa_pairs(lines_token_ids=held_out_lines_token_ids)
    held_out_hard_qa_pairs = create_bow_qa_pairs(lines_token_ids=held_out_hard_lines_token_ids)

    EPOCH_COUNT = 10
    BATCH_SIZE = 64
    CONTEXT_LENGTH = 32
    LEARNING_RATE = 0.1

    random.seed(0)
    mx.random.seed(0)
    model = AttentionModel(vocab_size=len(vocab), context_length=CONTEXT_LENGTH, embed_dim=64)
    optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

    def loss_fn(x, target):
        return nn.losses.cross_entropy(model(x), target, reduction="mean")

    def train_fn(qa_pairs: list[QAPair]):
        indices = list(range(len(qa_pairs)))
        random.shuffle(indices)
        loss_sum = 0

        batch_count = len(qa_pairs) // BATCH_SIZE
        for batch_index in range(batch_count):
            pairs = []
            for i in range(batch_index * BATCH_SIZE, (batch_index+1) * BATCH_SIZE):
                pairs.append(qa_pairs[indices[i]])
            inputs = [pair.question for pair in pairs]
            targets = [pair.answer for pair in pairs]
            targets = mx.array(targets)
            loss, grads = nn.value_and_grad(model, loss_fn)(inputs, targets)
            optimizer.update(model, grads)
            loss_sum += loss
            mx.eval(model.parameters(), optimizer.state, loss_sum)
        return loss_sum, batch_count

    def eval_fn(qa_pairs: list[QAPair]):
        correct_count = mx.array(0)
        last_word_correct_count = mx.array(0)
        last_word_incorrect_count = mx.array(0)

        # for pair in qa_pairs:
        for start_index in range(0, len(qa_pairs), BATCH_SIZE):
            pairs = qa_pairs[start_index:start_index+BATCH_SIZE]
            inputs = [pair.question for pair in pairs]
            targets = [pair.answer for pair in pairs]
            targets = mx.array(targets)
            output = model(inputs)
            output_token_ids = output.argmax(axis=1)
            # print(f"targets.shape: {targets.shape}")
            # print(f"output.shape: {output.shape}")
            # print(f"output_token_ids.shape: {output_token_ids.shape}")
            correct_count += mx.sum(output_token_ids == targets)
            # for i in range(len(targets)):
            for i in range(targets.shape[0]):
                target = targets[i]
                output_token_id = output_token_ids[i]
                # print(f"target: {target}, output_token_id: {output_token_id}")
                if pairs[i].is_last_noun:
                    if target == output_token_id:
                        last_word_correct_count += 1
                    else:
                        last_word_incorrect_count += 1
        return int(correct_count), int(last_word_correct_count), int(last_word_incorrect_count)

    train_start_time = time.perf_counter()

    losses = []
    for _ in range(EPOCH_COUNT):
        loss_sum, batch_count = train_fn(train_qa_pairs)
        losses.append(loss_sum / batch_count)

    train_end_time = time.perf_counter()
    print(f"Train elapsed time: {(train_end_time - train_start_time):.6f} seconds")

    eval_start_time = time.perf_counter()

    held_out_correct_count, held_out_last_word_correct_count, held_out_last_word_incorrect_count = eval_fn(held_out_qa_pairs)
    held_out_incorrect_count = len(held_out_qa_pairs) - held_out_correct_count
    held_out_accuracy = held_out_correct_count / (held_out_correct_count + held_out_incorrect_count)
    held_out_last_word_accuracy = held_out_last_word_correct_count / (held_out_last_word_correct_count + held_out_last_word_incorrect_count)

    held_out_hard_correct_count, _, _ = eval_fn(held_out_hard_qa_pairs)
    held_out_hard_incorrect_count = len(held_out_hard_qa_pairs) - held_out_hard_correct_count
    held_out_hard_accuracy = held_out_hard_correct_count / (held_out_hard_correct_count + held_out_hard_incorrect_count)

    eval_end_time = time.perf_counter()
    print(f"Eval elapsed time: {(eval_end_time - eval_start_time):.6f} seconds")

    generated_token_ids = [0]
    while len(generated_token_ids) < CONTEXT_LENGTH:
        output = model([generated_token_ids])
        output_token_id = int(output.argmax())
        generated_token_ids.append(output_token_id)
        output_word = inv_vocab[output_token_id]
        if output_word == "<eos>":
            break
    
    generated_words = decode(token_ids=generated_token_ids, inv_vocab=inv_vocab)

    # print(f"train_correct_count: {train_correct_count}, train_incorrect_count: {train_incorrect_count}, train_accuracy: {train_accuracy}")
    print(f"held_out_correct_count: {held_out_correct_count}, held_out_incorrect_count: {held_out_incorrect_count}, held_out_accuracy: {held_out_accuracy}")
    print(f"held_out_last_word_correct_count: {held_out_last_word_correct_count}, held_out_last_word_incorrect_count: {held_out_last_word_incorrect_count}, held_out_last_word_accuracy: {held_out_last_word_accuracy}")
    print(f"held_out_hard_correct_count: {held_out_hard_correct_count}, held_out_hard_incorrect_count: {held_out_hard_incorrect_count}, held_out_hard_accuracy: {held_out_hard_accuracy}")
    print(f"generated_words: {generated_words}")

    os.makedirs("tmp", exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(losses)
    fig.savefig("tmp/v4.png")
    plt.close(fig)

def main():
    run_bow()

if __name__ == "__main__":
    main()
