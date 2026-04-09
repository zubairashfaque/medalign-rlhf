"""Hybrid dense (FAISS) + sparse (BM25) retriever with RRF fusion + cross-encoder rerank."""
from __future__ import annotations
import pickle, json
from pathlib import Path
import numpy as np


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank_list in rankings:
        for rank, doc_id in enumerate(rank_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, cfg: dict):
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.embedder = SentenceTransformer(cfg["embedding"]["model"])
        self.faiss = faiss.read_index(cfg["index"]["faiss_path"])
        with open(cfg["index"]["bm25_path"], "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        self.meta = [json.loads(l) for l in Path(cfg["index"]["metadata_path"]).read_text().splitlines()]
        self.reranker = None
        if cfg["reranker"]["enabled"]:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(cfg["reranker"]["model"])

    def search(self, query: str) -> list[dict]:
        r = self.cfg["retrieval"]
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        _, dense_ids = self.faiss.search(np.array(q_emb, dtype="float32"), r["top_k_dense"])
        dense_list = dense_ids[0].tolist()

        bm25_scores = self.bm25.get_scores(query.lower().split())
        sparse_list = np.argsort(bm25_scores)[::-1][: r["top_k_sparse"]].tolist()

        fused = reciprocal_rank_fusion([dense_list, sparse_list], k=r["rrf_k"])
        candidate_ids = [doc_id for doc_id, _ in fused[: max(20, r["final_top_k"] * 4)]]
        candidates = [self.meta[i] for i in candidate_ids]

        if self.reranker is not None:
            pairs = [(query, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            order = np.argsort(scores)[::-1][: r["final_top_k"]]
            return [candidates[i] for i in order]
        return candidates[: r["final_top_k"]]
