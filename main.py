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

def convert_index_to_word_embed(vocab_count: int, index: int) -> mx.array:
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

def convert_lines_indices_to_qa_pairs(vocab_count: int, lines_indices: list[list[int]]):
    qa_pairs = []
    for indices in lines_indices:
        word_embeds = convert_indices_to_word_embeds(vocab_count=vocab_count, indices=indices)
        bow_embeds = convert_word_embeds_to_bow_embeds(word_embeds)

        for i in range(1, len(indices)):
            input = bow_embeds[i-1]
            answer = indices[i]
            qa_pairs.append((input, answer))
    return qa_pairs


def run_bow():
    with open("data/v2/train.txt") as train_file:
        train_lines = train_file.readlines()

    with open("data/v2/held_out.txt") as held_out_file:
        held_out_lines = held_out_file.readlines()

    with open("data/v2/held_out_hard.txt") as held_out_hard_file:
        held_out_hard_lines = held_out_hard_file.readlines()

    with open("data/v2/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)

    train_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in train_lines]
    # train_lines_indices = train_lines_indices[:1000]

    held_out_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in held_out_lines]
    held_out_lines_indices = held_out_lines_indices[:100]

    held_out_hard_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in held_out_hard_lines]
    held_out_hard_lines_indices = held_out_hard_lines_indices[:100]

    train_qa_pairs = convert_lines_indices_to_qa_pairs(vocab_count=len(vocab), lines_indices=train_lines_indices)
    held_out_qa_pairs = convert_lines_indices_to_qa_pairs(vocab_count=len(vocab), lines_indices=held_out_lines_indices)
    held_out_hard_qa_pairs = convert_lines_indices_to_qa_pairs(vocab_count=len(vocab), lines_indices=held_out_hard_lines_indices)

    class BOWModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(len(vocab), len(vocab))
            self.layer2 = nn.Linear(len(vocab), len(vocab))

        def __call__(self, x):
            # return self.layer2(self.layer1(x))
            return self.layer2(nn.relu(self.layer1(x)))


    mx.random.seed(0)
    # model = nn.Linear(len(vocab), len(vocab))
    model = BOWModel()
    optimizer = optimizers.SGD(0.05)

    def loss_fn(x, target):
        return nn.losses.cross_entropy(model(x), target)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # train_correct_count = 0
    # train_incorrect_count = 0
    for pair in train_qa_pairs:
        gt_index = pair[1]
        # output = model(pair[0])
        # output_index = output.argmax()
        target = mx.zeros(len(vocab))
        target[gt_index] = 1
        _, grads = loss_and_grad_fn(pair[0], target)
        # grads = mx.grad(nn.losses.cross_entropy)(output, target)
        # grads_mat = mx.zeros((len(vocab), len(vocab)))
        # for i in range(len(pair[0])):
        #     grads_mat[:, i] = grads * pair[0][i]
        # optimizer.update(model, {"weight": grads_mat})
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # if output_index == gt_index:
        #     train_correct_count += 1
        # else:
        #     train_incorrect_count += 1
        # print(f"gt_index: {gt_index}")
        # print(f"output: {output}")
        # print(f"output[16]: {output[16]}")
        # print(f"output_index: {output_index}")
        # print(f"grads: {grads}")
    # train_accuracy = train_correct_count / (train_correct_count + train_incorrect_count)

    held_out_correct_count = 0
    held_out_incorrect_count = 0
    for pair in held_out_qa_pairs:
        gt_index = pair[1]
        output = model(pair[0])
        output_index = output.argmax()
        if output_index == gt_index:
            held_out_correct_count += 1
        else:
            held_out_incorrect_count += 1
    held_out_accuracy = held_out_correct_count / (held_out_correct_count + held_out_incorrect_count)

    held_out_hard_correct_count = 0
    held_out_hard_incorrect_count = 0
    for pair in held_out_hard_qa_pairs:
        gt_index = pair[1]
        output = model(pair[0])
        output_index = output.argmax()
        if output_index == gt_index:
            held_out_hard_correct_count += 1
        else:
            held_out_hard_incorrect_count += 1
    held_out_hard_accuracy = held_out_hard_correct_count / (held_out_hard_correct_count + held_out_hard_incorrect_count)

    # print(f"train_correct_count: {train_correct_count}, train_incorrect_count: {train_incorrect_count}, train_accuracy: {train_accuracy}")
    print(f"held_out_correct_count: {held_out_correct_count}, held_out_incorrect_count: {held_out_incorrect_count}, held_out_accuracy: {held_out_accuracy}")
    print(f"held_out_hard_correct_count: {held_out_hard_correct_count}, held_out_hard_incorrect_count: {held_out_hard_incorrect_count}, held_out_hard_accuracy: {held_out_hard_accuracy}")

def main():
    # run_linear()
    run_bow()

if __name__ == "__main__":
    main()
