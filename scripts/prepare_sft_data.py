"""Download medical instruction datasets, dedup, format as ChatML, push to HF Hub.

Run on laptop:
    poetry run python scripts/prepare_sft_data.py --hub-repo <user>/medalign-sft
"""
from __future__ import annotations
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets
from medalign.data.format import to_chatml, dedup_minhash


SOURCES = [
    ("medalpaca/medical_meadow_medqa", "instruction", "output"),
    ("medalpaca/medical_meadow_wikidoc", "instruction", "output"),
]


def load_all() -> Dataset:
    parts = []
    for repo, instr_col, resp_col in SOURCES:
        ds = load_dataset(repo, split="train")
        ds = ds.map(
            lambda ex: {"instruction": ex[instr_col], "response": ex[resp_col]},
            remove_columns=ds.column_names,
        )
        parts.append(ds)
    return concatenate_datasets(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-repo", required=False, help="HF Hub dataset repo to push to")
    ap.add_argument("--max-samples", type=int, default=50000)
    args = ap.parse_args()

    ds = load_all()
    print(f"Loaded {len(ds)} raw samples")

    keep = dedup_minhash(ds["instruction"], threshold=0.85)
    ds = ds.select(keep[: args.max_samples])
    print(f"After dedup + cap: {len(ds)}")

    ds = ds.map(lambda ex: {"text": to_chatml(ex["instruction"], ex["response"])})
    ds = ds.remove_columns(["instruction", "response"])

    out_dir = "data/sft"
    ds.save_to_disk(out_dir)
    print(f"Saved to {out_dir}")

    if args.hub_repo:
        ds.push_to_hub(args.hub_repo, private=False)
        print(f"Pushed to {args.hub_repo}")


if __name__ == "__main__":
    main()
