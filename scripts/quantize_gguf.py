"""Merge DPO LoRA into base, convert to GGUF, quantize at multiple levels with imatrix.

Requires llama.cpp built locally (see configs/quant.yaml).
Run: poetry run python scripts/quantize_gguf.py
"""
from __future__ import annotations
import subprocess, yaml
from pathlib import Path


def merge_adapter(base_model: str, adapter_repo: str, out_dir: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading {base_model} + adapter {adapter_repo}")
    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, adapter_repo).merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    return out_dir


def main():
    cfg = yaml.safe_load(Path("configs/quant.yaml").read_text())
    s, lc, q = cfg["source"], cfg["llama_cpp"], cfg["quantization"]

    merged_dir = merge_adapter(s["base_model"], s["dpo_adapter_repo"], s["merged_dir"])

    llama_dir = Path(lc["build_dir"])
    if not llama_dir.exists():
        subprocess.run(["git", "clone", lc["repo_url"], str(llama_dir)], check=True)
        subprocess.run(["make", "-C", str(llama_dir), "-j"], check=True)

    out = Path(q["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    f16_path = out / "model-f16.gguf"

    subprocess.run(
        ["python", str(llama_dir / "convert_hf_to_gguf.py"), merged_dir, "--outfile", str(f16_path), "--outtype", "f16"],
        check=True,
    )

    imatrix_path = out / "imatrix.dat"
    if q["imatrix"]["enabled"]:
        subprocess.run(
            [
                str(llama_dir / "llama-imatrix"),
                "-m", str(f16_path),
                "-f", q["imatrix"]["calibration_text"],
                "-o", str(imatrix_path),
            ],
            check=True,
        )

    for level in q["levels"]:
        out_file = out / f"model-{level}.gguf"
        cmd = [str(llama_dir / "llama-quantize")]
        if q["imatrix"]["enabled"]:
            cmd += ["--imatrix", str(imatrix_path)]
        cmd += [str(f16_path), str(out_file), level]
        print("→", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("Quantized variants:", list(out.glob("*.gguf")))


if __name__ == "__main__":
    main()
