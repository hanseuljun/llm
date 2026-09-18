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

def convert_index_to_word_embed(vocab_count: int, index: int):
    embed = mx.zeros(vocab_count)
    embed[index] = 1
    return embed

def convert_indices_to_word_embeds(vocab_count: int, indices: list[int]):
    return [convert_index_to_word_embed(vocab_count=vocab_count, index=index) for index in indices]

def convert_word_embeds_to_bow_embeds(word_embeds: list[mx.array]):
    bow_embeds = []
    for i in range(len(word_embeds)):
        word_embed = word_embeds[i]
        bow_embed = mx.zeros(word_embeds[0].shape)
        for word_embed in word_embeds[:i+1]:
            bow_embed += word_embed
        bow_embed /= i + 1
        bow_embeds.append(bow_embed)
    return bow_embeds

def run_bow():
    # with open("data/v2/held_out.txt") as held_out_file:
    #     held_out_lines = held_out_file.readlines()

    with open("data/v2/train.txt") as train_file:
        train_lines = train_file.readlines()

    with open("data/v2/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)

    train_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in train_lines]
    train_lines_indices = train_lines_indices[:2]

    train_pairs = []
    for train_line_indices in train_lines_indices:
        train_line_indices = train_lines_indices[0]
        train_line_word_embeds = convert_indices_to_word_embeds(vocab_count=len(vocab), indices=train_line_indices)
        train_line_bow_embeds = convert_word_embeds_to_bow_embeds(train_line_word_embeds)

        for i in range(1, len(train_line_indices)):
            input = train_line_bow_embeds[i-1]
            answer = train_line_indices[i]
            train_pairs.append((input, answer))

    mx.random.seed(0)
    model = nn.Linear(len(vocab), len(vocab))

    for pair in train_pairs:
        gt_index = pair[1]
        output = model(pair[0])
        output_index = output.argmax()
        print(f"gt_index: {gt_index}")
        print(f"output: {output}")
        print(f"output[16]: {output[16]}")
        print(f"output_index: {output_index}")

    # print(train_line_indices)
    # print(train_line_word_embeds)
    # print(train_line_bow_embeds)

def main():
    # run_linear()
    run_bow()

if __name__ == "__main__":
    main()
