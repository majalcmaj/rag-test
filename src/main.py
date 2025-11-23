import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"


def read_cats_data() -> list[str]:
    with open("../data/cats.txt", "r") as file:
        cats = file.readlines()
        print(f"Read {len(cats)} cat facts from cats.txt")
        return cats


if __name__ == "__main__":
    cats_data = read_cats_data()
