import json

import mlx.core as mx
from mlx import nn, optimizers


def convert_line_to_indices(line: str, vocab: dict[str, int]) -> list[int]:
    words = ["<bos>"] + line.split() + ["<eos>"]
    return [vocab[word] for word in words]

def convert_lines_indices_to_index_pairs(lines_indices: list[list[int]]) -> list[tuple[int, int]]:
    index_pairs = []
    for indices in lines_indices:
        for i in range(len(indices) - 1):
            index_pairs.append((indices[i], indices[i+1]))
    return index_pairs

def convert_indices_to_words(indices: list[int], inv_vocab: dict[int, str]) -> list[str]:
    return [inv_vocab[index] for index in indices]

def run_linear():
    with open("data/v2/held_out.txt") as held_out_file:
        held_out_lines = held_out_file.readlines()

    with open("data/v2/train.txt") as train_file:
        train_lines = train_file.readlines()

    with open("data/v2/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)

    held_out_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in held_out_lines]
    train_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in train_lines]
    train_lines_indices = train_lines_indices[:100]
    held_out_lines_index_pairs = convert_lines_indices_to_index_pairs(held_out_lines_indices)
    train_lines_index_pairs = convert_lines_indices_to_index_pairs(train_lines_indices)

    mx.random.seed(0)
    model = nn.Linear(len(vocab), len(vocab))
    initial_weight = model.weight

    optimizer = optimizers.SGD(0.05)

    train_correct_count = 0
    train_incorrect_count = 0
    # loss_sum = 0
    # loss_count = 0

    for pair in train_lines_index_pairs:
        input = mx.zeros(len(vocab))
        input[pair[0]] = 1
        output = model(input)
        answer = mx.zeros(len(vocab))
        answer[pair[1]] = 1
        _, grads = mx.value_and_grad(nn.losses.cross_entropy)(output, answer)
        # print(f"grads: {grads}")
        grads_mat = mx.zeros((len(vocab), len(vocab)))
        grads_mat[:, pair[0]] = grads
        # print(f"grads_mat: {grads_mat}")
        # loss_sum += loss
        # loss_count += 1
        pred_index = output.argmax()
        if pred_index == pair[1]:
            train_correct_count += 1
            # print("correct")
        else:
            train_incorrect_count += 1
            # print("incorrect")
        # model.weight = model.weight - grads_mat * 0.05
        optimizer.update(model, {"weight": grads_mat})
    train_accuracy = train_correct_count / (train_correct_count + train_incorrect_count)

    held_out_correct_count = 0
    held_out_incorrect_count = 0
    # loss_sum = 0
    # loss_count = 0
    for pair in held_out_lines_index_pairs:
        input = mx.zeros(len(vocab))
        input[pair[0]] = 1
        output = model(input)
        answer = mx.zeros(len(vocab))
        answer[pair[1]] = 1
        _, grads = mx.value_and_grad(nn.losses.cross_entropy)(output, answer)
        # print(f"grads: {grads}")
        grads_mat = mx.zeros((len(vocab), len(vocab)))
        grads_mat[:, pair[0]] = grads
        # print(f"grads_mat: {grads_mat}")
        # loss_sum += loss
        # loss_count += 1
        pred_index = output.argmax()
        if pred_index == pair[1]:
            held_out_correct_count += 1
            # print("correct")
        else:
            held_out_incorrect_count += 1
            # print("incorrect")
        # model.weight = model.weight - grads_mat * 0.05
    held_out_accuracy = held_out_correct_count / (held_out_correct_count + held_out_incorrect_count)

    print(f"model: {model.parameters()}")
    print(f"train_correct_count: {train_correct_count}, train_incorrect_count: {train_incorrect_count}, train_accuracy: {train_accuracy}")
    print(f"held_out_correct_count: {held_out_correct_count}, held_out_incorrect_count: {held_out_incorrect_count}, held_out_accuracy: {held_out_accuracy}")
    # print(f"loss mean: {loss_sum / loss_count}")
    print(f"weight diff: {model.weight - initial_weight}")

def main():
    run_linear()

if __name__ == "__main__":
    main()
