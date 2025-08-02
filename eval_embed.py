#!/usr/bin/env python3
"""
eval_embed.py
-------------
Step-2 improvements for the FinAuxi RAG pipeline:
  • Embedding quality diagnostics (cosine histograms, basic recall@k)
  • Light domain fine-tuning on (query, positive_chunk, hard_neg) triplets
  • Saves the fine-tuned model to ./ft_bge

Usage
-----
python eval_embed.py --mode eval (only for model evaluation)
python eval_embed.py --mode fit --epochs 3 --lr 2e-5 (only for fine tuning)
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, evaluation
from sentence_transformers.readers import InputExample
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 1. COMMON CONSTANTS
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDINGS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 2. EMBEDDING QUALITY EVALUATION
# ---------------------------------------------------------------------------
def load_qa_pairs(path: str) -> List[dict]:
    """Expects JSONL: {"query": "...", "positive": "...", "negatives": [...]}"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def plot_cosine_distribution(
    model: SentenceTransformer, qa_pairs: List[dict], out: str = "cosine_dist.png"
):
    pos_sims, neg_sims = [], []
    for row in tqdm(qa_pairs, desc="Embedding pairs"):
        q_emb = model.encode([row["query"]], normalize_embeddings=True)
        pos_emb = model.encode([row["positive"]], normalize_embeddings=True)
        # Extract scalar from 2D array to avoid deprecation warning
        pos_sims.append(float(cosine_similarity(q_emb, pos_emb)[0][0]))

        for neg in row.get("negatives", []):
            neg_emb = model.encode([neg], normalize_embeddings=True)
            neg_sims.append(float(cosine_similarity(q_emb, neg_emb)[0][0]))

    plt.figure(figsize=(6, 4))
    plt.hist(pos_sims, bins=30, alpha=0.5, label="Positives", color="g")
    plt.hist(neg_sims, bins=30, alpha=0.5, label="Negatives", color="r")
    plt.title("Cosine similarity distribution")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    print(f"[eval] Cosine histogram saved to {out}")


def recall_at_k(
    model: SentenceTransformer,
    qa_pairs: List[dict],
    k: int = 5,
    index_path: str = FAISS_INDEX_PATH,
):
    """
    Very small sanity check:
    We embed the query and see if the *exact* positive chunk is in top-k.
    NOTE: assumes the positive chunk is already in the FAISS index.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    # Subclass to override embed_query properly
    class CustomHFEmbeddings(HuggingFaceEmbeddings):
        def __init__(self):
            super().__init__(
                model_name=BASE_MODEL_NAME,
                model_kwargs={"device": EMBEDDINGS_DEVICE},
            )

        def embed_query(self, texts):
            # Return numpy float32 normalized embeddings
            embs = model.encode(texts, normalize_embeddings=True)
            return embs.astype("float32")

    hf_emb = CustomHFEmbeddings()
    vs = FAISS.load_local(index_path, hf_emb, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": k})

    hits = 0
    for row in tqdm(qa_pairs, desc=f"R@{k}"):
        docs = retriever.get_relevant_documents(row["query"])
        if any(row["positive"].strip() == d.page_content.strip() for d in docs):
            hits += 1

    recall = hits / len(qa_pairs)
    print(f"[eval] Recall@{k} = {recall:.3f}")
    return recall


# ---------------------------------------------------------------------------
# 3. FINE-TUNING
# ---------------------------------------------------------------------------
def build_triplet_dataset(qa_pairs: List[dict]) -> Dataset:
    examples = []
    for row in qa_pairs:
        anchor = row["query"]
        positive = row["positive"]
        # pick first negative as the "hard" one
        if row.get("negatives"):
            negative = random.choice(row["negatives"])
            examples.append(InputExample(texts=[anchor, positive, negative]))
    dataset = Dataset.from_list([{"anchor": ex.texts[0], "positive": ex.texts[1], "negative": ex.texts[2]} for ex in examples])
    return dataset


def finetune(
    qa_pairs_path: str,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 8,
    model_out: str = "ft_bge",
):
    qa_pairs = load_qa_pairs(qa_pairs_path)
    dataset = build_triplet_dataset(qa_pairs)

    model = SentenceTransformer(BASE_MODEL_NAME, device=EMBEDDINGS_DEVICE)
    train_loss = losses.TripletLoss(model=model)

    evaluator = evaluation.TripletEvaluator.from_input_examples(
        [InputExample(texts=[r["query"], r["positive"], random.choice(r["negatives"])]) for r in qa_pairs[:50]],
        name="dev",
    )
   #SentenceTransformer.fit() do not accepts the batch_size parameter directly. 
   # This parameter should instead be passed as part of the train_dataloader_opts.
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    training_args = SentenceTransformerTrainingArguments(
    output_dir=model_out,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=lr,
    warmup_ratio=0.1,
    eval_strategy="epoch",      # ← changed from evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
)

    eval_dataset = dataset.shuffle(seed=42).select(range(min(50, len(dataset))))
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
    )
    trainer.train()
    trainer.save_model(model_out)

    print(f"[fit] Fine-tuned model saved to {model_out}")


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "fit"], required=True)
    parser.add_argument("--qa_pairs", default="data/qa_pairs.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    if not os.path.isfile(args.qa_pairs):
        print(f"[warn] QA pairs file not found at {args.qa_pairs}")
        return

    qa_pairs = load_qa_pairs(args.qa_pairs)

    if args.mode == "eval":
        model = SentenceTransformer(BASE_MODEL_NAME, device=EMBEDDINGS_DEVICE)
        plot_cosine_distribution(model, qa_pairs)
        recall_at_k(model, qa_pairs, k=5)
    elif args.mode == "fit":
        finetune(args.qa_pairs, args.epochs, args.lr)


if __name__ == "__main__":
    main()
