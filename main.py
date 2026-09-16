import json


def convert_line_to_indices(line: str, vocab: dict[str, int]) -> list[int]:
    words = ["<bos>"] + line.split() + ["<eos>"]
    return [vocab[word] for word in words]

def convert_indices_to_words(indices: list[int], inv_vocab: dict[int, str]) -> list[str]:
    return [inv_vocab[index] for index in indices]

with open("data/held_out.txt") as held_out_file:
    held_out_lines = held_out_file.readlines()

with open("data/train.txt") as train_file:
    train_lines = train_file.readlines()

with open("data/vocab.json") as vocab_file:
    vocab = json.load(vocab_file)

train_lines_indices = [convert_line_to_indices(line, vocab=vocab) for line in train_lines]
first_train_line_indices = train_lines_indices[0]

inv_vocab = {value: key for key, value in vocab.items()}
first_train_line_words = convert_indices_to_words(first_train_line_indices, inv_vocab=inv_vocab)

vocab_count = len(vocab)
freq_table = {i: [0] * vocab_count for i in range(vocab_count)}

for indices in train_lines_indices:
    for i in range(len(indices) - 1):
        prev = indices[i]
        next = indices[i+1]
        freq_table[prev][next] += 1

print(vocab)
print(first_train_line_indices)
print(inv_vocab)
print(first_train_line_words)
print(freq_table)
