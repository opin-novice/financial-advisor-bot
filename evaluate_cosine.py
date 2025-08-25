# evaluate_cosine.py

# Cosine-similarity evaluation on data/qa_paris.jsonl for NextRAG
# Uses SentenceTransformer (default: BAAI/bge-m3) with L2-normalized embeddings.
#to run: python evaluate_cosine.py --data data/qa_paris.jsonl --model BAAI/bge-m3 --device cpu --out-csv cosine_eval_results.csv



import argparse
import json
import os
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit("Please `pip install sentence-transformers` first.") from e


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_text_vocab(rows: List[Dict]) -> List[str]:
    """Collect all unique strings to embed once."""
    seen = set()
    all_texts = []
    for r in rows:
        for key in ("query", "positive"):
            t = r[key].strip()
            if t not in seen:
                seen.add(t)
                all_texts.append(t)
        for neg in r.get("negatives", []):
            t = neg.strip()
            if t not in seen:
                seen.add(t)
                all_texts.append(t)
    return all_texts


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Return L2-normalized embeddings (so dot = cosine)."""
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    # ensure float32
    return np.asarray(emb, dtype=np.float32)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine for already L2-normalized vectors is dot product."""
    return float(np.dot(u, v))


def evaluate(rows: List[Dict],
             model_name: str = "BAAI/bge-m3",
             device: str = "cpu",
             batch_size: int = 64) -> Tuple[List[Dict], Dict[str, float]]:
    # Load model
    print(f"[INFO] Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    # Build vocab & embed once
    print("[INFO] Collecting unique texts...")
    vocab = build_text_vocab(rows)
    print(f"[INFO] Unique strings to embed: {len(vocab)}")

    print("[INFO] Encoding texts (normalized)...")
    embeddings = embed_texts(model, vocab, batch_size=batch_size)

    # Map text -> vector
    index = {text: i for i, text in enumerate(vocab)}

    results = []
    correct = 0
    margins = []
    pos_sims = []
    max_neg_sims = []

    print("[INFO] Scoring pairs...")
    for r in tqdm(rows):
        q = r["query"].strip()
        p = r["positive"].strip()
        negs = [n.strip() for n in r.get("negatives", [])]

        qv = embeddings[index[q]]
        pv = embeddings[index[p]]
        pos_sim = cosine(qv, pv)

        neg_sims_this = []
        for n in negs:
            nv = embeddings[index[n]]
            neg_sims_this.append(cosine(qv, nv))

        max_neg = max(neg_sims_this) if neg_sims_this else -1.0
        is_correct = pos_sim > max_neg
        margin = pos_sim - max_neg

        if is_correct:
            correct += 1
        margins.append(margin)
        pos_sims.append(pos_sim)
        max_neg_sims.append(max_neg)

        results.append({
            "query": q,
            "positive": p,
            "pos_sim": round(pos_sim, 6),
            "max_neg_sim": round(max_neg, 6),
            "margin": round(margin, 6),
            "is_correct": bool(is_correct)
        })

    # Aggregate summary
    n = len(rows)
    summary = {
        "n_items": n,
        "accuracy_pos_gt_all_negs": correct / n if n else 0.0,
        "mean_pos_sim": float(np.mean(pos_sims)) if pos_sims else 0.0,
        "mean_max_neg_sim": float(np.mean(max_neg_sims)) if max_neg_sims else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "min_margin": float(np.min(margins)) if margins else 0.0,
        "max_margin": float(np.max(margins)) if margins else 0.0,
    }
    return results, summary


def save_csv(path: str, rows: List[Dict]) -> None:
    import csv
    keys = ["query", "positive", "pos_sim", "max_neg_sim", "margin", "is_correct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Cosine-similarity eval for NextRAG using qa_paris.jsonl")
    parser.add_argument("--data", default="data/qa_paris.jsonl", help="Path to JSONL")
    parser.add_argument("--model", default="BAAI/bge-m3", help="SentenceTransformer model name")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 ...")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-csv", default="cosine_eval_results.csv")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}")

    rows = read_jsonl(args.data)
    if not rows:
        raise SystemExit("No rows found in dataset.")

    results, summary = evaluate(
        rows,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )

    # Print summary
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # Save per-item CSV
    save_csv(args.out_csv, results)
    print(f"\n[INFO] Saved per-item results to: {args.out_csv}")


if __name__ == "__main__":
    main()
