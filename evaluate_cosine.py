#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cosine_eval.py

Standalone cosine-similarity evaluator for NextRAG.

- Reads QA pairs from data/qa_paris.jsonl
- Uses SentenceTransformer embeddings (default: BAAI/bge-m3)
- Computes cosine similarity between each query and its positive vs. negatives
- Prints per-sample scores and summary stats to the terminal
- Does NOT write any files

Usage:
    python cosine_eval.py
    python cosine_eval.py --model BAAI/bge-m3 --path data/qa_paris.jsonl --batch-size 32 --no-normalize
"""

import os
import sys
import json
import math
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosine similarity evaluation for NextRAG.")
    parser.add_argument("--path", type=str, default="data/qa_paris.jsonl",
                        help="Path to the JSONL evaluation file.")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3",
                        help="SentenceTransformer model to use.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable L2 normalization of embeddings before cosine.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra details per example.")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping malformed JSON on line {line_no}: {e}")
    if not data:
        print("[ERROR] No valid records found in the file.")
        sys.exit(1)
    return data


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 2D numpy array row-wise. Safeguards against division by zero.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each row of a and each row of b.
    Shapes:
        a: (N, D)
        b: (M, D)
    Returns:
        (N, M) matrix of cosine similarities.
    """
    # (N, D) dot (D, M) -> (N, M)
    return np.dot(a, b.T)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int, normalize: bool) -> np.ndarray:
    """
    Embed a list of texts with SentenceTransformer, returns a (len(texts), D) numpy array.
    """
    with torch.no_grad():
        # normalize_embeddings=True already L2-normalizes vectors in many ST models,
        # but we keep manual control for clarity and parity with our pipeline.
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False  # we'll do it manually if requested
        )
    if normalize:
        vecs = l2_normalize(vecs)
    return vecs


def evaluate(
    model_name: str,
    path: str,
    batch_size: int = 32,
    normalize: bool = True,
    verbose: bool = False
) -> None:
    print(f"[INFO] Loading eval set from: {path}")
    records = load_jsonl(path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading embedding model: {model_name} (device: {device})")
    model = SentenceTransformer(model_name, device=device)

    # Collect all texts to embed once (efficient batching)
    queries: List[str] = []
    positives: List[str] = []
    negatives_flat: List[str] = []  # flattened negatives
    neg_counts: List[int] = []      # how many negatives per record (to slice back)

    for rec in records:
        q = rec.get("query", "").strip()
        p = rec.get("positive", "").strip()
        negs = [n.strip() for n in rec.get("negatives", []) if n and n.strip()]
        if not q or not p or not negs:
            # skip malformed entries
            continue
        queries.append(q)
        positives.append(p)
        negatives_flat.extend(negs)
        neg_counts.append(len(negs))

    if not queries:
        print("[ERROR] No valid (query, positive, negatives) triplets found.")
        sys.exit(1)

    # Embed in batches
    print(f"[INFO] Embedding {len(queries)} queries, {len(positives)} positives, {len(negatives_flat)} negatives...")
    Q = embed_texts(model, queries, batch_size, normalize)
    P = embed_texts(model, positives, batch_size, normalize)
    N = embed_texts(model, negatives_flat, batch_size, normalize)

    # Compute similarities
    # Query-to-Positive: diagonal of cosine between Q and P
    qp_sims = np.sum(Q * P, axis=1)  # because both are L2-normalized if normalize=True

    # Query-to-Negatives: need to align by record
    # We'll compute cosine similarities per-record using slices of N
    results: List[Dict[str, Any]] = []
    neg_offset = 0

    correct = 0
    pos_sims_accum: List[float] = []
    max_neg_sims_accum: List[float] = []
    margins_accum: List[float] = []

    for i, q_vec in enumerate(Q):
        k = neg_counts[i]
        neg_slice = N[neg_offset:neg_offset + k]  # (k, D)
        neg_offset += k

        # Cosine similarities q_vec (1, D) vs neg_slice (k, D)
        # If normalized, dot = cosine; otherwise, compute full cosine via dot of normalized vectors
        if normalize:
            sims_neg = np.dot(neg_slice, q_vec)  # (k,)
        else:
            # do the safe path (this also works if normalize=True)
            qv = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            nv = l2_normalize(neg_slice)
            sims_neg = np.dot(nv, qv)

        pos_sim = float(qp_sims[i]) if normalize else float(
            np.dot(q_vec, P[i]) / ((np.linalg.norm(q_vec) + 1e-12) * (np.linalg.norm(P[i]) + 1e-12))
        )
        max_neg = float(np.max(sims_neg))
        margin = pos_sim - max_neg
        hit = pos_sim > max_neg

        if hit:
            correct += 1
        pos_sims_accum.append(pos_sim)
        max_neg_sims_accum.append(max_neg)
        margins_accum.append(margin)

        example = {
            "id": i,
            "query": queries[i],
            "positive_sim": round(pos_sim, 4),
            "max_negative_sim": round(max_neg, 4),
            "margin": round(margin, 4),
            "hit": bool(hit),
        }
        results.append(example)

    # Print per-example (compact)
    print("\n===== Per-Example Cosine Similarities =====")
    for r in results:
        status = "✓" if r["hit"] else "✗"
        if verbose:
            print(f"[{status}] #{r['id']:03d}  pos={r['positive_sim']:.4f}  "
                  f"max_neg={r['max_negative_sim']:.4f}  margin={r['margin']:.4f}  | {r['query']}")
        else:
            print(f"[{status}] #{r['id']:03d}  pos={r['positive_sim']:.4f}  "
                  f"max_neg={r['max_negative_sim']:.4f}  margin={r['margin']:.4f}")

    # Summary
    total = len(results)
    acc = (correct / total) if total > 0 else 0.0
    mean_pos = float(np.mean(pos_sims_accum)) if pos_sims_accum else 0.0
    mean_max_neg = float(np.mean(max_neg_sims_accum)) if max_neg_sims_accum else 0.0
    mean_margin = float(np.mean(margins_accum)) if margins_accum else 0.0
    med_margin = float(np.median(margins_accum)) if margins_accum else 0.0
    p95_margin = float(np.percentile(margins_accum, 95)) if margins_accum else 0.0

    print("\n===== Summary =====")
    print(f"Model                : {model_name}")
    print(f"Normalize embeddings : {normalize}")
    print(f"Examples evaluated   : {total}")
    print(f"Accuracy (pos > max neg): {acc:.4f}")
    print(f"Mean pos sim         : {mean_pos:.4f}")
    print(f"Mean max-neg sim     : {mean_max_neg:.4f}")
    print(f"Mean margin          : {mean_margin:.4f}")
    print(f"Median margin        : {med_margin:.4f}")
    print(f"95th pct margin      : {p95_margin:.4f}")


def main():
    args = parse_args()
    # To mirror pipeline defaults, we normalize by default (cosine == dot on unit vectors)
    evaluate(
        model_name=args.model,
        path=args.path,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
