import json
import time

import mlx.core as mx
from mlx import nn, optimizers


def encode(line: str, vocab: dict[str, int]) -> list[int]:
    words = ["<bos>"] + line.split() + ["<eos>"]
    return [vocab[word] for word in words]

def create_word_embeds(token_ids: list[int], vocab_size: int) -> list[mx.array]:
    return [mx.eye(vocab_size)[token_id] for token_id in token_ids]

def create_bow_embed(word_embeds: list[mx.array]) -> mx.array:
    return mx.sum(mx.stack(word_embeds[:len(word_embeds)]), axis=0) / len(word_embeds)

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
        self.embedding = nn.Embedding(num_embeddings=vocab_size, dims=embed_dim)
        self.layers = [
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, vocab_size),
        ]

    def __call__(self, x):
        # word_embeds: list[mx.array] = create_word_embeds(token_ids=x, vocab_size=self.vocab_size)
        # bow_embed: mx.array = create_bow_embed(word_embeds=word_embeds)
        # x = bow_embed
        x = self.embedding(x[-1])
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
    train_lines_token_ids = train_lines_token_ids[:1000]

    held_out_lines_token_ids = [encode(line, vocab=vocab) for line in held_out_lines]
    held_out_lines_token_ids = held_out_lines_token_ids[:100]

    held_out_hard_lines_token_ids = [encode(line, vocab=vocab) for line in held_out_hard_lines]
    held_out_hard_lines_token_ids = held_out_hard_lines_token_ids[:100]

    train_qa_pairs = create_bow_qa_pairs(lines_token_ids=train_lines_token_ids)
    held_out_qa_pairs = create_bow_qa_pairs(lines_token_ids=held_out_lines_token_ids)
    held_out_hard_qa_pairs = create_bow_qa_pairs(lines_token_ids=held_out_hard_lines_token_ids)

    mx.random.seed(0)
    model = BOWModel(vocab_size=len(vocab), embed_dim=64)
    optimizer = optimizers.SGD(learning_rate=0.05)

    def loss_fn(x, target):
        return nn.losses.cross_entropy(model(x), target)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    train_start_time = time.perf_counter()

    BATCH_SIZE = 16
    for batch_token_id in range(len(train_qa_pairs) // BATCH_SIZE):
        batch_start_token_id = batch_token_id * BATCH_SIZE
        batch_size = min(BATCH_SIZE, len(train_qa_pairs) - batch_start_token_id)
        for i in range(batch_size):
            pair = train_qa_pairs[batch_start_token_id + i]
            gt_token_id = pair[1]
            target = mx.zeros(len(vocab))
            target[gt_token_id] = 1
            _, grads = loss_and_grad_fn(pair[0], target)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

    train_end_time = time.perf_counter()
    print(f"Train elapsed time: {(train_end_time - train_start_time):.6f} seconds")

    def eval_fn(qa_pairs: list[tuple[mx.array, int]]):
        correct_count = 0
        for pair in qa_pairs:
            gt_token_id = pair[1]
            output = model(pair[0])
            output_token_id = output.argmax()
            if output_token_id == gt_token_id:
                correct_count += 1
        return correct_count

    held_out_correct_count = eval_fn(held_out_qa_pairs)
    held_out_incorrect_count = len(held_out_qa_pairs) - held_out_correct_count
    held_out_accuracy = held_out_correct_count / (held_out_correct_count + held_out_incorrect_count)

    held_out_hard_correct_count = eval_fn(held_out_hard_qa_pairs)
    held_out_hard_incorrect_count = len(held_out_hard_qa_pairs) - held_out_hard_correct_count
    held_out_hard_accuracy = held_out_hard_correct_count / (held_out_hard_correct_count + held_out_hard_incorrect_count)

    generated_token_ids = [0]
    while len(generated_token_ids) < 10:
        output = model(generated_token_ids)
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
