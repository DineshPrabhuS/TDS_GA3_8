from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RankRequest(BaseModel):
    query_id: str
    query: str
    candidates: list[str]

@app.post("/rank")
async def rank(req: RankRequest):

    texts = [req.query] + req.candidates

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    vectors = [d.embedding for d in response.data]

    q = np.array(vectors[0])
    docs = np.array(vectors[1:])

    q = q / np.linalg.norm(q)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    sims = docs @ q

    top3 = np.argsort(-sims)[:3]

    return {"ranking": top3.tolist()}