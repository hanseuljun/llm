import mlx.core as mx

BEGINNING_OF_TEXT = "<|beginningoftext|>"
END_OF_TEXT = "<|endoftext|>"
PADDING = "<|padding|>"

def encode(story: list[str], vocab: dict[str, int]) -> list[int]:
    words = [BEGINNING_OF_TEXT] + story
    return [vocab[word] for word in words]

def pad(token_ids: mx.array, vocab: dict[str, int], target_length: int) -> mx.array:
    return mx.pad(token_ids, (0, target_length - token_ids.shape[0]), constant_values=vocab[PADDING])

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
    vocab[BEGINNING_OF_TEXT] = len(vocab)
    vocab[END_OF_TEXT] = len(vocab)
    vocab[PADDING] = len(vocab)
    for story in stories:
        for word in story:
            if word in vocab:
                continue
            vocab[word] = len(vocab)
    print(f"vocab: {vocab}")

    stories_ids = []
    for story in stories:
        stories_ids.append(pad(mx.array(encode(story=story, vocab=vocab)), vocab=vocab, target_length=256))
    stories_ids = mx.stack(stories_ids)
    print(f"stories_ids.shape: {stories_ids.shape}")

if __name__ == "__main__":
    main()
