import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer

print("1. Loading kjv.json...")
with open("kjv.json", "r", encoding="utf-8") as f:
    raw_kjv = json.load(f)

references = list(raw_kjv.keys())
texts = [re.sub(r"[#\[\]]", "", raw_kjv[ref]).strip() for ref in references]

print("2. Loading free local embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"3. Converting {len(texts)} verses into semantic vectors (this takes ~1-2 minutes)...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

np.save("kjv_vectors.npy", embeddings)
with open("kjv_clean.json", "w", encoding="utf-8") as f:
    json.dump({"references": references, "texts": texts}, f)

print("Done! Saved kjv_vectors.npy and kjv_clean.json.")