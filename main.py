def main():
    with open("data/tinystories/train.txt") as train_file:
        lines = [train_file.readline() for _ in range(1)]
        words = [line.split() for line in lines]
        print(f"lines: {lines}")
        print(f"words: {words}")

if __name__ == "__main__":
    main()
