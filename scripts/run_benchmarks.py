"""Benchmark base / SFT / DPO / DPO+RAG on PubMedQA, MedQA-USMLE, MedMCQA.

Run: poetry run python scripts/run_benchmarks.py
"""
from __future__ import annotations
import argparse, csv, yaml
from pathlib import Path


def load_pubmedqa():
    from datasets import load_dataset
    return load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")


def load_medqa():
    from datasets import load_dataset
    return load_dataset("GBaker/MedQA-USMLE-4-options", split="test")


def load_medmcqa():
    from datasets import load_dataset
    return load_dataset("openlifescienceai/medmcqa", split="validation")


def evaluate(generate_fn, dataset, prompt_fn, gold_fn) -> float:
    correct = 0
    total = 0
    for ex in dataset:
        pred = generate_fn(prompt_fn(ex))
        if gold_fn(ex).lower() in pred.lower():
            correct += 1
        total += 1
    return correct / max(total, 1)


def make_llama_cpp_generator(gguf_path: str):
    from llama_cpp import Llama
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_gpu_layers=0)
    def gen(prompt: str) -> str:
        out = llm(prompt, max_tokens=128, temperature=0.0, stop=["</s>", "<|im_end|>"])
        return out["choices"][0]["text"]
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="Path to a quantized GGUF to benchmark")
    ap.add_argument("--use-rag", action="store_true")
    ap.add_argument("--out", default="outputs/benchmarks.csv")
    args = ap.parse_args()

    gen = make_llama_cpp_generator(args.gguf)
    retriever = None
    if args.use_rag:
        cfg = yaml.safe_load(Path("configs/rag.yaml").read_text())
        from medalign.rag.hybrid import HybridRetriever
        retriever = HybridRetriever(cfg)

    def with_rag(question: str) -> str:
        if retriever is None:
            return f"Question: {question}\nAnswer:"
        ctx = "\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(retriever.search(question)))
        return f"Context:\n{ctx}\nQuestion: {question}\nAnswer:"

    rows = []
    pubmed = load_pubmedqa()
    acc = evaluate(
        gen,
        pubmed.select(range(min(200, len(pubmed)))),
        prompt_fn=lambda ex: with_rag(ex["question"]),
        gold_fn=lambda ex: ex["final_decision"],
    )
    rows.append(("PubMedQA", acc))

    medqa = load_medqa()
    acc = evaluate(
        gen,
        medqa.select(range(min(200, len(medqa)))),
        prompt_fn=lambda ex: with_rag(ex["question"] + " Options: " + " ".join(ex["options"].values())),
        gold_fn=lambda ex: ex["answer"],
    )
    rows.append(("MedQA-USMLE", acc))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "accuracy"])
        w.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
