import json

import mlx.core as mx


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

def main():
    with open("data/held_out.txt") as held_out_file:
        held_out_lines = held_out_file.readlines()

    with open("data/train.txt") as train_file:
        train_lines = train_file.readlines()

    with open("data/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)

    held_out_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in held_out_lines]
    train_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in train_lines]
    first_train_line_indices = train_lines_indices[0]

    held_out_lines_index_pairs = convert_lines_indices_to_index_pairs(held_out_lines_indices)
    train_lines_index_pairs = convert_lines_indices_to_index_pairs(train_lines_indices)
    first_train_line_index_pairs = train_lines_index_pairs[:6]

    inv_vocab = {value: key for key, value in vocab.items()}
    first_train_line_words = convert_indices_to_words(first_train_line_indices, inv_vocab=inv_vocab)

    vocab_count = len(vocab)
    freq_table = {i: [0] * vocab_count for i in range(vocab_count)}
    freq_arr = mx.zeros((vocab_count, vocab_count))

    for indices in train_lines_indices:
        for i in range(len(indices) - 1):
            prev = indices[i]
            next = indices[i+1]
            freq_table[prev][next] += 1
            freq_arr[prev, next] += 1

    next_index_arr = freq_arr.argmax(axis=1)
    pred_indices = [0]
    for _ in range(6):
        next_index = int(next_index_arr[pred_indices[-1]])
        pred_indices.append(next_index)
        if next_index == vocab["<eos>"]:
            break
    pred_words = convert_indices_to_words(pred_indices, inv_vocab=inv_vocab)

    held_out_correct_count = 0
    held_out_incorrect_count = 0
    for pair in held_out_lines_index_pairs:
        pred_index = int(next_index_arr[pair[0]])
        if pred_index == pair[1]:
            held_out_correct_count += 1
        else:
            held_out_incorrect_count += 1
    held_out_accuracy = held_out_correct_count / (held_out_correct_count + held_out_incorrect_count)

    train_correct_count = 0
    train_incorrect_count = 0
    for pair in train_lines_index_pairs:
        pred_index = int(next_index_arr[pair[0]])
        if pred_index == pair[1]:
            train_correct_count += 1
        else:
            train_incorrect_count += 1
    train_accuracy = train_correct_count / (train_correct_count + train_incorrect_count)

    print(f"vocab: {vocab}")
    print(f"first_train_line_indices: {first_train_line_indices}")
    print(f"first_train_line_index_pairs: {first_train_line_index_pairs}")
    print(f"inv_vocab: {inv_vocab}")
    print(f"first_train_line_words: {first_train_line_words}")
    print(f"freq_table: {freq_table}")
    print(f"freq_arr: {freq_arr}")
    print(f"next_index_arr: {next_index_arr}")
    print(f"pred_indices: {pred_indices}")
    print(f"pred_words: {pred_words}")
    print(f"held_out_correct_count: {held_out_correct_count}, held_out_incorrect_count: {held_out_incorrect_count}, held_out_accuracy: {held_out_accuracy}")
    print(f"train_correct_count: {train_correct_count}, train_incorrect_count: {train_incorrect_count}, train_accuracy: {train_accuracy}")

main()
