BEGINNING_OF_TEXT = "<|beginningoftext|>"
END_OF_TEXT = "<|endoftext|>"

def encode(story: list[str], vocab: dict[str, int]) -> list[int]:
    words = [BEGINNING_OF_TEXT] + story
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
        print(f"stories: {stories}")

    vocab = {}
    vocab[BEGINNING_OF_TEXT] = 0
    vocab[END_OF_TEXT] = 1
    for story in stories:
        for word in story:
            if word in vocab:
                continue
            vocab[word] = len(vocab)
    print(f"vocab: {vocab}")

    stories_ids = []
    for story in stories:
        stories_ids.append(encode(story=story, vocab=vocab))
    print(f"stories_ids: {stories_ids}")

if __name__ == "__main__":
    main()
