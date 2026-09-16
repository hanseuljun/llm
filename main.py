with open("data/held_out.txt") as held_out_file:
    held_out_lines = held_out_file.readlines()

with open("data/train.txt") as train_file:
    train_lines = train_file.readlines()

with open("data/vocab.json") as vocab_file:
    vocab = vocab_file.read()

print(held_out_lines)
print(train_lines)
print(vocab)
