END_OF_TEXT = "<|endoftext|>"

def main():
    with open("data/tinystories/train.txt") as train_file:
        stories = []
        story = []
        while len(stories) < 2:
            line = train_file.readline()
            story.append(line)
            if END_OF_TEXT in line:
                stories.append(story)
                story = []
            if len(stories) > 1:
                break

        words = []
        for story in stories:
            for line in story:
                words += line.split()
        print(f"lines: {stories}")
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
