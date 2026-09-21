import json
import random
import time

import mlx.core as mx
from mlx import nn, optimizers


def encode(line: str, vocab: dict[str, int]) -> list[int]:
    words = ["<bos>"] + line.split() + ["<eos>"]
    return [vocab[word] for word in words]

def create_bow_qa_pairs(lines_token_ids: list[list[int]]) -> list[tuple[list[int], int]]:
    def convert_token_ids_to_bow_qa_pairs(token_ids: list[int]) -> list[tuple[list[int], int]]:
        return [(token_ids[:i], token_ids[i]) for i in range(1, len(token_ids))]

    qa_pairs = []
    for token_ids in lines_token_ids:
        qa_pairs += convert_token_ids_to_bow_qa_pairs(token_ids=token_ids)
    return qa_pairs

def decode(token_ids: list[int], inv_vocab: dict[int, str]) -> list[str]:
    return [inv_vocab[token_id] for token_id in token_ids]

class BOWModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, dims=embed_dim)
        self.position_embedding = nn.Embedding(num_embeddings=32, dims=embed_dim)
        self.layers = [
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, vocab_size),
        ]

    def __call__(self, x):
        embeds = []
        for token_ids in x:
            word_embeds: mx.array = mx.stack([self.word_embedding(token_id) for token_id in token_ids])
            bow_embed: mx.array = mx.mean(word_embeds)
            position_embed: mx.array = self.position_embedding(len(token_ids))
            embed = bow_embed * position_embed
            embeds.append(embed)
        x = mx.stack(embeds, axis=0)
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)

def run_bow():
    with open("data/v2/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)

    with open("data/v2/train.txt") as train_file:
        train_lines = train_file.readlines()

    with open("data/v2/held_out.txt") as held_out_file:
        held_out_lines = held_out_file.readlines()

    with open("data/v2/held_out_hard.txt") as held_out_hard_file:
        held_out_hard_lines = held_out_hard_file.readlines()

    inv_vocab = {value: key for key, value in vocab.items()}

    train_lines_token_ids = [encode(line, vocab=vocab) for line in train_lines]
    # train_lines_token_ids = train_lines_token_ids[:10000]
    print(f"train_lines_token_ids: {len(train_lines_token_ids)}")

    held_out_lines_token_ids = [encode(line, vocab=vocab) for line in held_out_lines]
    # held_out_lines_token_ids = held_out_lines_token_ids[:100]
    print(f"held_out_lines_token_ids: {len(held_out_lines_token_ids)}")

    held_out_hard_lines_token_ids = [encode(line, vocab=vocab) for line in held_out_hard_lines]
    # held_out_hard_lines_token_ids = held_out_hard_lines_token_ids[:100]
    print(f"held_out_hard_lines_token_ids: {len(held_out_hard_lines_token_ids)}")

    train_qa_pairs = create_bow_qa_pairs(lines_token_ids=train_lines_token_ids)
    held_out_qa_pairs = create_bow_qa_pairs(lines_token_ids=held_out_lines_token_ids)
    held_out_hard_qa_pairs = create_bow_qa_pairs(lines_token_ids=held_out_hard_lines_token_ids)

    EPOCH_COUNT = 2
    BATCH_SIZE = 64

    mx.random.seed(0)
    model = BOWModel(vocab_size=len(vocab), embed_dim=64)
    optimizer = optimizers.SGD(learning_rate=0.01 * BATCH_SIZE)

    def loss_fn(x, target):
        return nn.losses.cross_entropy(model(x), target, reduction="mean")
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    train_start_time = time.perf_counter()

    for _ in range(EPOCH_COUNT):
        indices = list(range(len(train_qa_pairs)))
        random.shuffle(indices)
        for batch_token_id in range(len(train_qa_pairs) // BATCH_SIZE):
            batch_start_token_id = batch_token_id * BATCH_SIZE
            # pairs = train_qa_pairs[batch_start_token_id:batch_start_token_id+BATCH_SIZE]
            pairs = []
            for i in range(batch_start_token_id, batch_start_token_id+BATCH_SIZE):
                pairs.append(train_qa_pairs[indices[i]])
            inputs = [pair[0] for pair in pairs]
            targets = [pair[1] for pair in pairs]
            targets = mx.array(targets)
            _, grads = loss_and_grad_fn(inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

    train_end_time = time.perf_counter()
    print(f"Train elapsed time: {(train_end_time - train_start_time):.6f} seconds")

    def eval_fn(qa_pairs: list[tuple[list[int], int]]):
        correct_count = mx.array(0)
        for pair in qa_pairs:
            gt_token_ids = mx.array([pair[1]])
            output = model([pair[0]])
            output_token_ids = output.argmax(axis=1)
            # if output_token_id == gt_token_id:
                # correct_count += 1
            correct_count += mx.sum(output_token_ids == gt_token_ids)
        return correct_count

    eval_start_time = time.perf_counter()

    held_out_correct_count = int(eval_fn(held_out_qa_pairs))
    held_out_incorrect_count = len(held_out_qa_pairs) - held_out_correct_count
    held_out_accuracy = held_out_correct_count / (held_out_correct_count + held_out_incorrect_count)

    held_out_hard_correct_count = int(eval_fn(held_out_hard_qa_pairs))
    held_out_hard_incorrect_count = len(held_out_hard_qa_pairs) - held_out_hard_correct_count
    held_out_hard_accuracy = held_out_hard_correct_count / (held_out_hard_correct_count + held_out_hard_incorrect_count)

    eval_end_time = time.perf_counter()
    print(f"Eval elapsed time: {(eval_end_time - eval_start_time):.6f} seconds")

    generated_token_ids = [0]
    while len(generated_token_ids) < 10:
        output = model([generated_token_ids])
        output_token_id = output.argmax().item()
        generated_token_ids.append(output_token_id)
        output_word = inv_vocab[output_token_id]
        if output_word == "<eos>":
            break
    
    generated_words = decode(token_ids=generated_token_ids, inv_vocab=inv_vocab)

    # print(f"train_correct_count: {train_correct_count}, train_incorrect_count: {train_incorrect_count}, train_accuracy: {train_accuracy}")
    print(f"held_out_correct_count: {held_out_correct_count}, held_out_incorrect_count: {held_out_incorrect_count}, held_out_accuracy: {held_out_accuracy}")
    print(f"held_out_hard_correct_count: {held_out_hard_correct_count}, held_out_hard_incorrect_count: {held_out_hard_incorrect_count}, held_out_hard_accuracy: {held_out_hard_accuracy}")
    print(f"generated_words: {generated_words}")

def main():
    run_bow()

if __name__ == "__main__":
    main()
