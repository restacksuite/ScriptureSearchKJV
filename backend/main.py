import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Scripture for Purpose API")

# Configure CORS so your Cloudflare frontend can communicate with this Railway backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://scriptureforpurpose.com",
        "https://www.scriptureforpurpose.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your embedding model and data on startup
print("Loading model and vectors...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# If your vector files are in the backend directory, load them directly
try:
    vectors = np.load("kjv_vectors.npy")
    import json
    with open("kjv_clean.json", "r", encoding="utf-8") as f:
        verses = json.load(f)
    print(f"Successfully loaded {len(verses)} verses and vector database.")
except Exception as e:
    print(f"Warning during data loading: {e}")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def read_root():
    return {"status": "online", "message": "Scripture for Purpose API is running."}

@app.post("/search")
def search_verses(request: SearchRequest):
    try:
        # Encode the user's query into a vector embedding
        query_vector = model.encode([request.query])
        
        # Calculate cosine similarity against all verse vectors
        # (Assuming standard normalized dot-product or cosine calculation)
        similarities = np.dot(vectors, query_vector.T).flatten()
        
        # Get top matching indices
        top_indices = np.argsort(similarities)[::-1][:request.top_k]
        
        results = []
        for idx in top_indices:
            verse_data = verses[int(idx)]
            results.append({
                "book": verse_data.get("book"),
                "chapter": verse_data.get("chapter"),
                "verse": verse_data.get("verse"),
                "text": verse_data.get("text"),
                "score": float(similarities[idx])
            })
            
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))