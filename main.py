BEGINNING_OF_TEXT = "<|beginningoftext|>"
END_OF_TEXT = "<|endoftext|>"

def encode(line: str, vocab: dict[str, int]) -> list[int]:
    words = ["<bos>"] + line.split() + ["<eos>"]
    return [vocab[word] for word in words]

def main():
    with open("data/tinystories/train.txt") as train_file:
        stories = []
        story = []
        while len(stories) < 2:
            line = train_file.readline()
            story += line.split()
            if END_OF_TEXT in line:
                stories.append(story)
                story = []
            if len(stories) > 1:
                break

        words = []
        for story in stories:
            words += story
        print(f"lines: {stories}")
        print(f"words: {words}")

    word_set = set()
    word_set.add(BEGINNING_OF_TEXT)
    for word in words:
        word_set.add(word)
    print(f"len(words): {len(words)}")
    print(f"len(word_set): {len(word_set)}")
    vocab = {}
    for index, word in enumerate(word_set):
        vocab[word] = index
    print(f"vocab: {vocab}")
    ids = [vocab[word] for word in words]
    print(f"ids: {ids}")

if __name__ == "__main__":
    main()
