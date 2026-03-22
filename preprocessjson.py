import requests
import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embeddings(texts):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": texts
        }
    )
    r.raise_for_status()
    return r.json()["embeddings"]


records = []
segment_id = 0

for file in os.listdir("transcripts"):
    with open(f"transcripts/{file}", "r", encoding="utf-8") as f:
        data = json.load(f)

    lecture = data["lecture"]
    segments = data["segments"]

    texts = [seg["text"] for seg in segments]
    embeddings = create_embeddings(texts)

    for i, seg in enumerate(segments):
        records.append({
            "segment_id": segment_id,
            "lecture": lecture,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "embedding": embeddings[i]
        })
        segment_id += 1

df = pd.DataFrame.from_records(records)
embedding_matrix = np.vstack(df["embedding"].values)

import joblib

joblib.dump(
    {
        "df": df,
        "embedding_matrix": embedding_matrix
    },
    "lecture_index.joblib"
)
incoming_query = input("Ask a Question: ")
query_embedding = create_embeddings([incoming_query])[0]

similarities = cosine_similarity(
    embedding_matrix,
    [query_embedding]
).flatten()

top_k = 5
top_idx = similarities.argsort()[::-1][:top_k]

results = df.iloc[top_idx]

print(results[["lecture", "start", "end", "text"]])


