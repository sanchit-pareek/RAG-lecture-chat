import joblib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


data = joblib.load("lecture_index.joblib")
df = data["df"]
embedding_matrix = data["embedding_matrix"]

df = df.sort_values(["lecture", "start"]).reset_index(drop=True)

def create_embeddings(texts):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": texts}
    )
    r.raise_for_status()
    return r.json()["embeddings"]

def expand_with_neighbors(df, indices, window=1):
    expanded = set()
    for idx in indices:
        for i in range(max(0, idx-window), min(len(df), idx+window+1)):
            if df.iloc[i].lecture == df.iloc[idx].lecture:
                expanded.add(i)
    return df.iloc[sorted(expanded)]

def merge_segments(df):
    merged, current = [], None
    for row in df.itertuples():
        if current is None:
            current = {
                "lecture": row.lecture,
                "start": row.start,
                "end": row.end,
                "text": row.text
            }
        elif row.lecture == current["lecture"] and row.start <= current["end"] + 1.0:
            current["end"] = row.end
            current["text"] += " " + row.text
        else:
            merged.append(current)
            current = {
                "lecture": row.lecture,
                "start": row.start,
                "end": row.end,
                "text": row.text
            }
    if current:
        merged.append(current)
    return merged

def call_llm(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1:8b", 
              "prompt": prompt, 
              "stream": False}
    )
    r.raise_for_status()
    return r.json()["response"]

def build_prompt(incoming_query, contexts):
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        blocks.append(
            f"[{i}] Lecture {ctx['lecture']} "
            f"({ctx['start']:.1f}s–{ctx['end']:.1f}s)\n{ctx['text']}"
        )
    return f"""
You are a teaching assistant answering strictly from the provided lecture excerpts.
If the answer is not contained in them, say:
"I don't know based on the provided lectures."

Question:
{incoming_query}

Lecture excerpts:
{chr(10).join(blocks)}

Answer:
""".strip()

while True:
    q = input("Ask a question (or 'exit'): ")
    if q.lower() == "exit":
        break

    q_emb = create_embeddings([q])[0]
    sims = cosine_similarity(embedding_matrix, [q_emb]).flatten()

    if sims.max() < 0.25:
        print("I don't know based on the provided lectures.")
        print("-" * 60)
        continue

    top_idx = sims.argsort()[::-1][:5]
    expanded = expand_with_neighbors(df, top_idx, window=1)
    contexts = merge_segments(expanded)

    prompt = build_prompt(q, contexts)
    answer = call_llm(prompt)
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    with open("answer.txt", "w", encoding="utf-8") as f:
        f.write(answer)    

    print("\nAnswer:\n", answer)
    print("-" * 60)
