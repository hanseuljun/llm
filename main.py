import json


def line_to_indices(line, vocab):
    words = ["<bos>"] + line.split() + ["<eos>"]
    return [vocab[word] for word in words]

with open("data/held_out.txt") as held_out_file:
    held_out_lines = held_out_file.readlines()

with open("data/train.txt") as train_file:
    train_lines = train_file.readlines()

with open("data/vocab.json") as vocab_file:
    vocab = json.load(vocab_file)

train_line = train_lines[0]
# train_words = ["<bos>"] + train_line.split() + ["<eos>"]
# train_indices = [vocab[word] for word in train_words]
train_indices = line_to_indices(train_line, vocab=vocab)

print(vocab)
# print(train_words)
print(train_indices)
