"""Pareto chart: GGUF quantization level vs PubMedQA accuracy vs file size."""
from __future__ import annotations
import csv
from pathlib import Path


def benchmark_gguf_variants(gguf_dir: str, eval_fn, output_csv: str) -> list[dict]:
    """eval_fn(gguf_path) -> accuracy float."""
    rows = []
    for path in sorted(Path(gguf_dir).glob("model-*.gguf")):
        size_mb = path.stat().st_size / 1e6
        acc = eval_fn(str(path))
        rows.append({"variant": path.stem, "size_mb": round(size_mb, 1), "accuracy": round(acc, 4)})
        print(rows[-1])

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "size_mb", "accuracy"])
        w.writeheader()
        w.writerows(rows)
    return rows


def plot_pareto(rows: list[dict], out_png: str = "outputs/pareto.png"):
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["size_mb"])
    xs = [r["size_mb"] for r in rows]
    ys = [r["accuracy"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, "o-")
    for r in rows:
        plt.annotate(r["variant"], (r["size_mb"], r["accuracy"]), fontsize=8)
    plt.xlabel("Model size (MB)")
    plt.ylabel("PubMedQA accuracy")
    plt.title("MedAlign: quantization Pareto frontier")
    plt.grid(True, alpha=0.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
