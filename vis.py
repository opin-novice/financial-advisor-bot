# vis.py

import faiss
import numpy as np
import pickle
from sklearn.manifold import TSNE
import plotly.express as px
import os

# === Step 1: Load FAISS Index ===
index_path = "faiss_index/index.faiss"

if not os.path.exists(index_path):
    raise FileNotFoundError(f"FAISS index not found at: {index_path}")

index = faiss.read_index(index_path)

# === Step 2: Reconstruct Vectors ===
try:
    vectors = index.reconstruct_n(0, index.ntotal)
except Exception:
    vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])

print(f"Loaded {index.ntotal} vectors of dimension {vectors.shape[1]}")

# === Step 3: Load Metadata (Optional) ===
metadata_path = "faiss_index/index.pkl"

if os.path.exists(metadata_path):
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    # Try to get text labels if possible
    if isinstance(metadata, list):
        labels = [str(item) for item in metadata]
    elif isinstance(metadata, dict):
        labels = list(metadata.values())
    else:
        labels = [str(i) for i in range(index.ntotal)]
else:
    labels = [str(i) for i in range(index.ntotal)]

# === Step 4: Reduce to 3D using t-SNE ===
tsne = TSNE(n_components=3, perplexity=30, random_state=42)
vectors_3d = tsne.fit_transform(vectors)

# === Step 5: Plot with Plotly ===
fig = px.scatter_3d(
    x=vectors_3d[:, 0],
    y=vectors_3d[:, 1],
    z=vectors_3d[:, 2],
    text=labels,  # Hover label
    title="FAISS Vector Visualization (3D t-SNE)",
    labels={'x': 'X', 'y': 'Y', 'z': 'Z'}
)
fig.update_traces(marker=dict(size=4))
fig.show()