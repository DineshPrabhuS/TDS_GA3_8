from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import os

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


class SearchRequest(BaseModel):
    query_id: str
    query: str
    candidates: list[str]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [item.embedding for item in response.data]


@app.post("/")
def semantic_search(req: SearchRequest):

    texts = [req.query] + req.candidates

    embeddings = get_embeddings(texts)

    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]

    scores = []

    for i, emb in enumerate(candidate_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    top3 = [idx for idx, score in scores[:3]]

    return {
        "ranking": top3
    }