from typing import List
import json
import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

type VectorDB = List[tuple[str, List[float]]]

def create_vector_db(lines: List[str]) -> VectorDB:
    return list(map(lambda line: (line, ollama.embed(model=EMBEDDING_MODEL, input=line)['embeddings'][0]), lines))


def read_cats_data() -> List[str]:
    with open("../data/cats.txt", "r") as file:
        cats = file.readlines()
        print(f"Read {len(cats)} cat facts from cats.txt")
        return cats

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(vector_db: VectorDB, query: str, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = [(line, cosine_similarity(query_embedding, embedding)) for line, embedding in vector_db]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


if __name__ == "__main__":
    cats_data = read_cats_data()
    vector_db = create_vector_db(cats_data)
    print(f"Created vector DB...")

    while True:
        input_query = input("Your question: ")
        close_embeddings = retrieve(vector_db, input_query)
        print(f"Top relevant cat facts:\n{json.dumps(close_embeddings)}")
        instructions = "Answer the question based on the following context and if you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext:\n" + "\n".join([fact for fact, _ in close_embeddings])

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_query},
            ],
            stream=True)

        print("Answer:")
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print("\n")
