"""Build a medical-text calibration corpus for llama.cpp imatrix quantization."""
from __future__ import annotations
from pathlib import Path
from datasets import load_dataset


def main(out_path: str = "data/calibration_medical.txt", n_samples: int = 2000):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pubmed = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    medqa = load_dataset("medalpaca/medical_meadow_medqa", split="train")

    chunks = []
    for row in pubmed.select(range(min(n_samples, len(pubmed)))):
        ctx = row.get("context")
        if isinstance(ctx, dict):
            ctx = " ".join(ctx.get("contexts", []))
        if ctx:
            chunks.append(ctx)
    for row in medqa.select(range(min(n_samples, len(medqa)))):
        chunks.append(f"Q: {row['instruction']}\nA: {row['output']}")

    Path(out_path).write_text("\n\n".join(chunks))
    print(f"Wrote {len(chunks)} chunks → {out_path}")


if __name__ == "__main__":
    main()
