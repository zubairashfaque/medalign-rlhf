"""Build hybrid FAISS + BM25 index over PubMedQA abstracts (laptop, CPU)."""
from __future__ import annotations
import json, pickle, yaml
from pathlib import Path
import numpy as np


def main():
    cfg = yaml.safe_load(Path("configs/rag.yaml").read_text())
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss

    ds = load_dataset(cfg["corpus"]["dataset_id"], cfg["corpus"]["subset"], split="train")
    texts = []
    for row in ds:
        ctx = row.get(cfg["corpus"]["text_field"])
        if isinstance(ctx, dict):
            ctx = " ".join(ctx.get("contexts", []))
        if ctx:
            texts.append(ctx)
    print(f"Indexing {len(texts)} passages")

    out_dir = Path("data/rag")
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = SentenceTransformer(cfg["embedding"]["model"])
    embs = embedder.encode(
        texts,
        batch_size=cfg["embedding"]["batch_size"],
        normalize_embeddings=cfg["embedding"]["normalize"],
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, cfg["index"]["faiss_path"])

    bm25 = BM25Okapi([t.lower().split() for t in texts])
    with open(cfg["index"]["bm25_path"], "wb") as f:
        pickle.dump(bm25, f)

    with open(cfg["index"]["metadata_path"], "w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": i, "text": t}) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
