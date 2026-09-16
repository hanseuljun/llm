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

train_line = train_lines[0]
train_indices = convert_line_to_indices(train_line, vocab=vocab)

inv_vocab = {value: key for key, value in vocab.items()}
train_words = convert_indices_to_words(train_indices, inv_vocab=inv_vocab)

print(vocab)
print(train_indices)
print(inv_vocab)
print(train_words)
