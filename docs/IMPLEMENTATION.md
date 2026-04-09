# MedAlign — Implementation Guide

Comprehensive reference for what has been built and exactly how to run it end-to-end.

## Project Goal
Build a domain-adaptive medical LLM demonstrating ownership of the entire modern LLM post-training lifecycle: **SFT → DPO alignment → Hybrid RAG → GGUF quantization → FastAPI/Gradio serving**. Same pipeline shape as InstructGPT→RLHF / Llama post-training, scaled to a single 16 GB consumer GPU via QLoRA.

## Execution Split
| Stage | Where | Why |
|---|---|---|
| Data curation, dedup, ChatML formatting | Laptop | CPU-bound |
| **SFT QLoRA (4-bit NF4)** | **Kaggle T4x2** | GPU |
| **DPO preference-pair generation** | **Kaggle T4x2** | GPU sampling from SFT model |
| **DPO alignment (TRL DPOTrainer)** | **Kaggle T4x2** | GPU |
| RAG index build (BGE + BM25 + FAISS) | Laptop | CPU |
| GGUF quantization (llama.cpp) | Laptop | CPU |
| Benchmarks (PubMedQA / MedQA / MedMCQA / RAGAS / LLM-judge) | Laptop | CPU/GPU |
| Serving (FastAPI + Gradio) | Laptop | — |

Artifact flow: **Kaggle → HuggingFace Hub → Laptop**.

**Identifiers used in this repo:**
- GitHub: `zubairashfaque/medalign-rlhf`
- HF Hub user: `Zubairash`
- HF datasets/repos: `Zubairash/medalign-sft`, `Zubairash/medalign-sft-adapters`, `Zubairash/medalign-dpo-pairs`, `Zubairash/medalign-dpo-adapters`

---

## Repository Layout
```
medalign-rlhf/
├── README.md
├── pyproject.toml
├── Makefile
├── Dockerfile
├── .gitignore
├── .github/workflows/ci.yml
├── configs/
│   ├── sft.yaml          # LoRA r=16/α=32, lr=2e-4 cosine, NF4
│   ├── dpo.yaml          # β=0.1, frozen SFT ref, lr=5e-6
│   ├── rag.yaml          # BGE-large + BM25 + ms-marco rerank
│   └── quant.yaml        # Q8_0/Q6_K/Q5_K_M/Q4_K_M/Q4_K_S + imatrix
├── src/medalign/
│   ├── data/format.py
│   ├── training/sft.py
│   ├── training/dpo.py
│   ├── rag/hybrid.py
│   ├── eval/llm_judge.py
│   ├── eval/ragas_eval.py
│   └── quantization/benchmark.py
├── scripts/
│   ├── prepare_sft_data.py
│   ├── build_calibration.py
│   ├── build_rag_index.py
│   ├── quantize_gguf.py
│   ├── run_benchmarks.py
│   └── serve.py
├── kaggle_notebooks/
│   ├── kaggle_01_sft_qlora.ipynb
│   ├── kaggle_02_generate_dpo_pairs.ipynb
│   └── kaggle_03_dpo_alignment.ipynb
├── tests/unit/
│   ├── test_format.py
│   └── test_rrf.py
└── data/                  # .gitignored
```

---

## What Has Been Implemented (file by file)

### `src/medalign/training/sft.py`
`run_sft(config_path, hub_repo)`:
- `BitsAndBytesConfig`: NF4 4-bit, double-quant, bf16 compute.
- `prepare_model_for_kbit_training` then `LoraConfig` on `[q,k,v,o,gate,up,down]_proj`, r=16, α=32, dropout=0.05.
- TRL `SFTTrainer` over the HF dataset (`text` column = ChatML), packing off, paged AdamW 8-bit, cosine schedule, bf16.
- Pushes adapters + tokenizer to `hub_repo` on completion.

### `src/medalign/training/dpo.py`
`run_dpo(config_path, hub_repo)`:
- Loads base in 4-bit, attaches SFT adapters via `PeftModel.from_pretrained(..., is_trainable=True)`.
- `DPOConfig`: β=0.1, max_length=2048, max_prompt_length=1024.
- `ref_model=None` → TRL uses adapter-disabled base as the frozen reference (PEFT-DPO trick — saves a full model copy).
- Expects dataset columns: `prompt`, `chosen`, `rejected`.

### `src/medalign/data/format.py`
- `to_chatml(instruction, response, system?)` — Mistral ChatML with `<|im_start|>system/user/assistant`.
- `dedup_minhash(texts, threshold=0.85)` — `datasketch.MinHashLSH`, 128 perms, returns indices to keep.

### `src/medalign/rag/hybrid.py`
- `reciprocal_rank_fusion(rankings, k=60)` — pure-python RRF (unit-tested).
- `HybridRetriever`:
  1. BGE-large-en-v1.5 dense → FAISS top-50.
  2. BM25Okapi sparse top-50.
  3. RRF fusion → ~20 candidates.
  4. CrossEncoder `ms-marco-MiniLM-L-12-v2` rerank → final top-5.

### `src/medalign/eval/llm_judge.py`
- GPT-4o-mini in JSON mode. Rubric: accuracy / safety / completeness / citations (1–5 each) + rationale.
- `aggregate(scores)` for batch means.

### `src/medalign/eval/ragas_eval.py`
- Wraps `ragas.evaluate` with faithfulness, answer_relevancy, context_precision, context_recall.

### `src/medalign/quantization/benchmark.py`
- `benchmark_gguf_variants(gguf_dir, eval_fn, output_csv)` — file size + accuracy callback.
- `plot_pareto(rows, out_png)` — matplotlib Pareto frontier.

### `scripts/prepare_sft_data.py`
Concatenates `medalpaca/medical_meadow_medqa` + `medical_meadow_wikidoc`, MinHash dedup, caps at 50k, ChatML formats, saves to `data/sft/`, optionally pushes to `Zubairash/medalign-sft`.

### `scripts/build_calibration.py`
2k PubMedQA + 2k MedQA chunks → `data/calibration_medical.txt` for llama.cpp `imatrix`.

### `scripts/build_rag_index.py`
PubMedQA `pqa_artificial` → BGE embeddings (batch 64, normalized) → `IndexFlatIP`, BM25 pickle, JSONL metadata under `data/rag/`.

### `scripts/quantize_gguf.py`
1. Pulls `Zubairash/medalign-dpo-adapters`, merges via `merge_and_unload()`, saves merged HF model in fp16.
2. Clones + builds `llama.cpp` under `third_party/` if missing.
3. `convert_hf_to_gguf.py … --outtype f16` → `model-f16.gguf`.
4. `llama-imatrix` on calibration corpus.
5. `llama-quantize --imatrix` for each level in `[Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_K_S]`.

### `scripts/run_benchmarks.py`
`llama_cpp.Llama` generator over PubMedQA + MedQA-USMLE (200 samples each). Optional `--use-rag` enables `HybridRetriever` and citation-aware prompt. Writes CSV.

### `scripts/serve.py`
- FastAPI `POST /ask` (Pydantic `Query` model).
- Optional `--gradio` UI.
- `--use-rag` loads `HybridRetriever` and injects citations.
- ChatML prompt template with explicit "cite as [#]" instructions.

### Kaggle notebooks (thin — they `pip install` the GitHub repo + import `medalign.training`)
- **`kaggle_01_sft_qlora.ipynb`** — `run_sft('sft.yaml', hub_repo='Zubairash/medalign-sft-adapters')`.
- **`kaggle_02_generate_dpo_pairs.ipynb`** — Loads base+SFT in 4-bit, samples 2k prompts at T=0.3 and T=0.9, GPT-4o-mini judge picks A/B → `(prompt, chosen, rejected)` → push to `Zubairash/medalign-dpo-pairs`.
- **`kaggle_03_dpo_alignment.ipynb`** — `run_dpo('dpo.yaml', hub_repo='Zubairash/medalign-dpo-adapters')`.

### Infrastructure
- **`Dockerfile`** — Python 3.10 slim, Poetry install, exposes 8000, CMD: `serve.py --gguf /models/model-Q4_K_M.gguf --use-rag`.
- **`.github/workflows/ci.yml`** — ruff lint + pytest on push.
- **`tests/unit/`** — `test_format.py` (ChatML + dedup), `test_rrf.py` (RRF ordering).

---

## Step-by-Step Implementation Guide

### Phase 0 — One-time setup (laptop, ~10 min)
1. HF account at https://huggingface.co (username `Zubairash`).
2. HF token (write scope): https://huggingface.co/settings/tokens.
3. OpenAI API key (only for DPO judge + LLM-as-judge eval).
4. Empty GitHub repo: `https://github.com/zubairashfaque/medalign-rlhf`.
5. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`.
6. From repo root:
   ```bash
   cd "/home/zubair-ashfaque/GitHubProject/medalign-rlhf"
   git init && git add . && git commit -m "initial scaffolding"
   git remote add origin https://github.com/zubairashfaque/medalign-rlhf.git
   git push -u origin main
   poetry install
   poetry run huggingface-cli login   # paste HF token
   ```

### Phase 1 — Data preparation (laptop, ~15 min) ✅ COMPLETED

```bash
poetry run python scripts/prepare_sft_data.py --hub-repo Zubairash/medalign-sft
```

**Goal:** assemble a clean medical Q&A corpus, format it as Mistral ChatML, and push it to HF Hub so the Kaggle SFT notebook (Phase 2) can pull it down.

**Pipeline (5 steps):**
1. **Download** two medalpaca datasets — `medical_meadow_medqa` (USMLE multiple-choice, 10,178 rows) and `medical_meadow_wikidoc` (WikiDoc Q&A, 10,000 rows).
2. **Dedup** with MinHash LSH (Jaccard ≥ 0.85) — removes near-duplicate questions so the model doesn't over-memorize repeated text. Implementation: `medalign.data.format.dedup_minhash`.
3. **Cap** at 50,000 rows (`--max-samples`).
4. **Format** each row into ChatML via `medalign.data.format.to_chatml` — the prompt template Mistral-7B-Instruct expects.
5. **Save** to `data/sft/` as a HF Arrow dataset and **push** to `Zubairash/medalign-sft`.

**Final result:** 20,178 raw → **18,146** after dedup → pushed.

#### ⚠️ Bug found and fixed: medalpaca `instruction` column is a constant template

The medalpaca datasets store the **fixed prompt template** in the `instruction` column and the **actual question** in the `input` column:

| Dataset | `instruction` (constant for every row) | `input` (the real content) |
|---|---|---|
| medical_meadow_medqa | `"Please answer with one of the option in the bracket"` | `"A 23-year-old G1P0 woman at 22 weeks gestation presents with…"` |
| medical_meadow_wikidoc | `"Answer this question truthfully"` | `"What are the symptoms of pancreatitis?"` |

The original `prepare_sft_data.py` only read the `instruction` column, so MinHash dedup saw 20,178 copies of just **two unique strings** and kept exactly 2 rows. **Symptom in the logs:**
```
Loaded 20178 raw samples
After dedup + cap: 2     ← broken
```

**Fix** (`scripts/prepare_sft_data.py`): extend `SOURCES` with an explicit `input_col` and concatenate `instruction + input` before dedup:
```python
SOURCES = [
    ("medalpaca/medical_meadow_medqa", "instruction", "input", "output"),
    ("medalpaca/medical_meadow_wikidoc", "instruction", "input", "output"),
]
# in load_all():
ds = ds.map(lambda ex: {
    "instruction": (
        f"{ex[instr_col].strip()}\n\n{ex[input_col].strip()}"
        if ex.get(input_col)
        else ex[instr_col]
    ),
    "response": ex[resp_col],
}, remove_columns=ds.column_names)
```
After the fix: 20,178 → **18,146** kept (~10% removed as near-duplicates — healthy ratio).

#### Example of one finished training row (post-ChatML)
```
<|im_start|>system
You are a careful medical assistant. Be accurate and cite uncertainty.<|im_end|>
<|im_start|>user
Please answer with one of the option in the bracket

A 23-year-old G1P0 woman at 22 weeks gestation presents with vaginal bleeding…
Which of the following is the most likely diagnosis?
(A) Placenta previa  (B) Placental abruption  (C) Vasa previa  (D) Uterine rupture<|im_end|>
<|im_start|>assistant
(A) Placenta previa<|im_end|>
```
This `text` field is what `SFTTrainer` will tokenize in Phase 2.

#### ⚠️ Build issue: `llama-cpp-python` made optional
On this laptop `poetry install` failed building `llama-cpp-python==0.3.20` because Anaconda's `compiler_compat/ld` can't resolve OpenMP symbols (`GOMP_single_start@GOMP_1.0`, `GOMP_barrier@GOMP_1.0`) when linking `libggml-cpu.so`.

**Why it's safe to skip:** `llama-cpp-python` is only imported lazily in `scripts/run_benchmarks.py` and `scripts/serve.py` (Phases 7–8). `quantize_gguf.py` does **not** use the Python package — it shells out to llama.cpp binaries it builds itself. Phases 1–6 don't need it.

**Fix in `pyproject.toml`:** moved into a Poetry extra so default install skips it.
```toml
llama-cpp-python = { version = ">=0.2.90", optional = true }

[tool.poetry.extras]
serving = ["llama-cpp-python"]
```
**Before Phase 7**, install with the OpenMP-disabled workaround:
```bash
CMAKE_ARGS="-DGGML_OPENMP=OFF" poetry run pip install llama-cpp-python==0.3.20
```
`-DGGML_OPENMP=OFF` tells CMake to skip OpenMP entirely, sidestepping the linker issue. Tiny CPU-inference perf cost, no correctness impact.

- **Verify:** dataset visible at `https://huggingface.co/datasets/Zubairash/medalign-sft`; `data/sft/` exists locally; row count ≈ 18k.

### Phase 2 — SFT QLoRA on Kaggle (~3–6 h)
1. https://www.kaggle.com → New Notebook.
2. Settings → Accelerator **GPU T4 x2**, Internet **on**.
3. Add-ons → Secrets → `HF_TOKEN`.
4. Import `kaggle_notebooks/kaggle_01_sft_qlora.ipynb`. Run all.
5. Notebook installs the GitHub repo, wgets `configs/sft.yaml`, calls `run_sft(...)`.
- **Verify:** `Zubairash/medalign-sft-adapters` on HF; loss decreasing; trainable params ≈ 0.5–1%.

### Phase 3 — Generate DPO pairs on Kaggle (~2–4 h)
1. New notebook, T4x2, secrets `HF_TOKEN` + `OPENAI_API_KEY`.
2. Upload `kaggle_02_generate_dpo_pairs.ipynb`, run all.
- **Verify:** `Zubairash/medalign-dpo-pairs` has ~1.5–2k triples; spot-check `chosen` quality.

### Phase 4 — DPO alignment on Kaggle (~3–6 h)
1. New notebook, T4x2, `HF_TOKEN`.
2. Upload `kaggle_03_dpo_alignment.ipynb`, run all.
- **Verify:** `Zubairash/medalign-dpo-adapters` on HF; reward margin (`rewards/chosen - rewards/rejected`) > 0 and growing.

### Phase 5 — RAG index build (laptop, ~20 min)
```bash
poetry run python scripts/build_rag_index.py
```
- **Verify:** `data/rag/{faiss.index,bm25.pkl,meta.jsonl}` exist.

### Phase 6 — Quantization (laptop, ~1–2 h)
```bash
poetry run python scripts/build_calibration.py
poetry run python scripts/quantize_gguf.py
```
- **Verify:** `outputs/gguf/model-{Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q4_K_S}.gguf` (~7.7/5.9/5.1/4.4/4.1 GB for 7B).

### Phase 7 — Benchmarks (laptop)
```bash
poetry run python scripts/run_benchmarks.py --gguf outputs/gguf/model-Q4_K_M.gguf --out outputs/q4_norag.csv
poetry run python scripts/run_benchmarks.py --gguf outputs/gguf/model-Q4_K_M.gguf --use-rag --out outputs/q4_rag.csv
```
Aggregate into the README results table comparing base / SFT / DPO / DPO+RAG. Use `medalign.eval.llm_judge.score_answer` and `medalign.eval.ragas_eval.evaluate_rag` for qualitative metrics.
- **Verify:** monotonic improvement base < SFT < DPO < DPO+RAG.

### Phase 8 — Serving (laptop)
```bash
poetry run python scripts/serve.py --gguf outputs/gguf/model-Q4_K_M.gguf --use-rag
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What are first-line treatments for type 2 diabetes?"}'

poetry run python scripts/serve.py --gguf outputs/gguf/model-Q4_K_M.gguf --use-rag --gradio --port 7860
```
Optional: `docker build -t medalign . && docker run -p 8000:8000 -v $PWD/outputs/gguf:/models medalign`.

### Phase 9 — Deploy demo to HuggingFace Spaces
1. Create Space: SDK Gradio, hardware CPU basic.
2. Push the Q4_K_M GGUF + a small `app.py` that imports `build_app` from `serve.py`.
3. Link the Space from `README.md`.

### Phase 10 — Polish
- `make test` (unit tests).
- Architecture diagram (Mermaid/PNG) in README.
- Pareto chart from `medalign.quantization.benchmark.plot_pareto`.
- Final commit + push.

---

## Critical Files (for iteration)
| Purpose | Path |
|---|---|
| Training hyperparams | `configs/sft.yaml`, `configs/dpo.yaml` |
| RAG knobs | `configs/rag.yaml` |
| Quantization levels | `configs/quant.yaml` |
| Add SFT datasets | `scripts/prepare_sft_data.py::SOURCES` |
| Add benchmarks | `scripts/run_benchmarks.py` |

**Reusable utilities:** `medalign.data.format.to_chatml`, `medalign.data.format.dedup_minhash`, `medalign.rag.hybrid.reciprocal_rank_fusion`, `medalign.rag.hybrid.HybridRetriever`, `medalign.eval.llm_judge.score_answer`, `medalign.quantization.benchmark.benchmark_gguf_variants`.

---

## Verification Checklist
- [ ] `poetry install` succeeds
- [ ] `make test` → both unit tests pass
- [ ] `prepare_sft_data.py` produces dataset on HF Hub
- [ ] Kaggle 01 → SFT adapters on HF (sane loss curve)
- [ ] Kaggle 02 → ≥1k DPO triples on HF
- [ ] Kaggle 03 → DPO adapters on HF (positive reward margin)
- [ ] `build_rag_index.py` writes 3 RAG artifacts
- [ ] `quantize_gguf.py` produces 5 GGUF files
- [ ] `run_benchmarks.py` shows base < SFT < DPO < DPO+RAG
- [ ] `serve.py` answers `/ask` with citations
- [ ] HF Space loads and answers a sample query
- [ ] CI workflow green

---

## Out of Scope
- PDF Projects 2 (OpenPanoptic) and 3.
- Full RLHF with reward model + PPO — DPO replaces this and is Kaggle-feasible.
- Multi-node / multi-GPU beyond Kaggle T4x2.
- Continuous online learning with human annotators.
