def main():
    with open("data/tinystories/train.txt") as train_file:
        lines = [train_file.readline() for _ in range(1)]
        words = []
        for line in lines:
            words += line.split()
        print(f"lines: {lines}")
        print(f"words: {words}")
    word_set = set()
    for word in words:
        word_set.add(word)
    print(f"len(words): {len(words)}")
    print(f"len(word_set): {len(word_set)}")
    vocab = {}
    for index, word in enumerate(word_set):
        vocab[word] = index
    print(f"vocab: {vocab}")
    indices = [vocab[word] for word in words]
    print(f"indices: {indices}")

if __name__ == "__main__":
    main()
