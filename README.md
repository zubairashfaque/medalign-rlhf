# MedAlign — Domain-Adaptive Medical LLM (SFT → DPO → RAG → GGUF)

End-to-end medical LLM post-training pipeline: **QLoRA SFT → DPO alignment → Hybrid RAG → GGUF quantization → FastAPI/Gradio serving**.

📘 **Full implementation guide:** [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) — file-by-file breakdown of every module + step-by-step run instructions for all 10 phases.

## Execution Split

| Stage | Where | Why |
|---|---|---|
| Data curation | Laptop | CPU-bound, dedup + filter |
| **SFT QLoRA** | **Kaggle (T4x2)** | GPU-bound |
| **DPO pair generation** | **Kaggle (T4x2)** | GPU sampling from SFT model |
| **DPO alignment** | **Kaggle (T4x2)** | GPU-bound |
| RAG index build | Laptop | CPU |
| GGUF quantization (llama.cpp) | Laptop | CPU |
| Evaluation (PubMedQA / MedQA / MedMCQA / RAGAS) | Laptop | CPU/GPU optional |
| Serving (FastAPI + Gradio) | Laptop | — |

Trained adapters flow **Kaggle → HuggingFace Hub → Laptop**.

## Architecture
```
medical datasets → [SFT QLoRA] → SFT adapters
                                     ↓
                       sample + judge (GPT-4)
                                     ↓
                            DPO preference pairs
                                     ↓
                              [DPO alignment] → DPO adapters
                                                    ↓
                                       merge + llama.cpp → GGUF (Q8/Q6/Q5/Q4)
                                                    ↓
                              + Hybrid RAG (FAISS+BM25+reranker)
                                                    ↓
                                       FastAPI + Gradio
```

## Quick Start (laptop)
```bash
poetry install
poetry run python scripts/prepare_sft_data.py     # → pushes SFT dataset to HF Hub
# ... run the 3 Kaggle notebooks ...
poetry run python scripts/build_rag_index.py
poetry run python scripts/quantize_gguf.py
poetry run python scripts/run_benchmarks.py
poetry run python scripts/serve.py
```

## Quick Start (Kaggle)
1. Upload notebooks from `kaggle_notebooks/` to Kaggle.
2. Add `HF_TOKEN` (and optionally `OPENAI_API_KEY`) as Kaggle Secrets.
3. Enable GPU T4x2 accelerator.
4. Run top-to-bottom. Each notebook pushes its outputs to your HF Hub repo.

## Results (to be filled)

| Variant | PubMedQA | MedQA-USMLE | MedMCQA |
|---|---|---|---|
| Base (Mistral-7B-Instruct) | — | — | — |
| + SFT | — | — | — |
| + DPO | — | — | — |
| + DPO + RAG | — | — | — |

## Tech Stack
Mistral-7B-Instruct-v0.3 · Unsloth · QLoRA (NF4) · TRL DPOTrainer · BGE-large-en-v1.5 · FAISS · BM25 · ms-marco-MiniLM reranker · llama.cpp GGUF · FastAPI · Gradio
