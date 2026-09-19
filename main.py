import json
import time

import mlx.core as mx
from mlx import nn, optimizers


def convert_line_to_indices(line: str, vocab: dict[str, int]) -> list[int]:
    words = ["<bos>"] + line.split() + ["<eos>"]
    return [vocab[word] for word in words]

def convert_lines_indices_to_bow_qa_pairs(
        lines_indices: list[list[int]],
        vocab_size: int,
) -> list[tuple[mx.array, int]]:
    def convert_indices_to_word_embeds(indices: list[int]) -> list[mx.array]:
        return [mx.eye(vocab_size)[index] for index in indices]

    def convert_word_embeds_to_bow_embeds(word_embeds: list[mx.array]) -> list[mx.array]:
        return [mx.sum(mx.stack(word_embeds[:i+1]), axis=0) / (i+1) for i in range(len(word_embeds))]

    def convert_indices_to_bow_qa_pairs(indices: list[int]) -> list[tuple[mx.array, int]]:
        word_embeds: list[mx.array] = convert_indices_to_word_embeds(indices=indices)
        bow_embeds: list[mx.array] = convert_word_embeds_to_bow_embeds(word_embeds=word_embeds)
        return [(bow_embeds[i-1], indices[i]) for i in range(1, len(indices))]

    qa_pairs = []
    for indices in lines_indices:
        qa_pairs += convert_indices_to_bow_qa_pairs(indices=indices)
    return qa_pairs

def convert_indices_to_words(indices: list[int], inv_vocab: dict[int, str]) -> list[str]:
    return [inv_vocab[index] for index in indices]

class BOWModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.layer1 = nn.Linear(vocab_size, vocab_size)
        self.layer2 = nn.Linear(vocab_size, vocab_size)

    def __call__(self, x):
        return self.layer2(nn.relu(self.layer1(x)))

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

    train_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in train_lines]
    train_lines_indices = train_lines_indices[:1000]

    held_out_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in held_out_lines]
    held_out_lines_indices = held_out_lines_indices[:100]

    held_out_hard_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in held_out_hard_lines]
    held_out_hard_lines_indices = held_out_hard_lines_indices[:100]

    train_qa_pairs = convert_lines_indices_to_bow_qa_pairs(lines_indices=train_lines_indices, vocab_size=len(vocab))
    held_out_qa_pairs = convert_lines_indices_to_bow_qa_pairs(lines_indices=held_out_lines_indices, vocab_size=len(vocab))
    held_out_hard_qa_pairs = convert_lines_indices_to_bow_qa_pairs(lines_indices=held_out_hard_lines_indices, vocab_size=len(vocab))

    mx.random.seed(0)
    model = BOWModel(len(vocab))
    optimizer = optimizers.SGD(learning_rate=0.05)

    def loss_fn(x, target):
        return nn.losses.cross_entropy(model(x), target)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    train_start_time = time.perf_counter()

    BATCH_SIZE = 16
    for batch_index in range(len(train_qa_pairs) // BATCH_SIZE):
        batch_start_index = batch_index * BATCH_SIZE
        batch_size = min(BATCH_SIZE, len(train_qa_pairs) - batch_start_index)
        for i in range(batch_size):
            pair = train_qa_pairs[batch_start_index + i]
            gt_index = pair[1]
            target = mx.zeros(len(vocab))
            target[gt_index] = 1
            _, grads = loss_and_grad_fn(pair[0], target)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

    train_end_time = time.perf_counter()
    print(f"Train elapsed time: {(train_end_time - train_start_time):.6f} seconds")

    def eval_fn(qa_pairs: list[tuple[mx.array, int]]):
        correct_count = 0
        for pair in qa_pairs:
            gt_index = pair[1]
            output = model(pair[0])
            output_index = output.argmax()
            if output_index == gt_index:
                correct_count += 1
        return correct_count

    held_out_correct_count = eval_fn(held_out_qa_pairs)
    held_out_incorrect_count = len(held_out_qa_pairs) - held_out_correct_count
    held_out_accuracy = held_out_correct_count / (held_out_correct_count + held_out_incorrect_count)

    held_out_hard_correct_count = eval_fn(held_out_hard_qa_pairs)
    held_out_hard_incorrect_count = len(held_out_hard_qa_pairs) - held_out_hard_correct_count
    held_out_hard_accuracy = held_out_hard_correct_count / (held_out_hard_correct_count + held_out_hard_incorrect_count)

    generated_indices = [0]
    while len(generated_indices) < 10:
        input = mx.zeros(len(vocab))
        for index in generated_indices:
            input[index] += 1
        input /= len(generated_indices)
        output = model(input)
        output_index = output.argmax().item()
        # print(f"output_index: {output_index}")
        generated_indices.append(output_index)
        output_word = inv_vocab[output_index]
        if output_word == "<eos>":
            break
    
    generated_words = convert_indices_to_words(indices=generated_indices, inv_vocab=inv_vocab)

    # print(f"train_correct_count: {train_correct_count}, train_incorrect_count: {train_incorrect_count}, train_accuracy: {train_accuracy}")
    print(f"held_out_correct_count: {held_out_correct_count}, held_out_incorrect_count: {held_out_incorrect_count}, held_out_accuracy: {held_out_accuracy}")
    print(f"held_out_hard_correct_count: {held_out_hard_correct_count}, held_out_hard_incorrect_count: {held_out_hard_incorrect_count}, held_out_hard_accuracy: {held_out_hard_accuracy}")
    print(f"generated_words: {generated_words}")

def main():
    run_bow()

if __name__ == "__main__":
    main()
