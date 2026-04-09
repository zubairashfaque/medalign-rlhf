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
| **RAG index build (BGE + BM25 + FAISS)** | **Kaggle T4** | GPU embedding (laptop GPU OOMs on BGE-large) |
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

**Goal:** Get every credential and tool installed once so the rest of the pipeline runs without surprises. This is the boring-but-mandatory plumbing that connects laptop ↔ GitHub ↔ Hugging Face Hub ↔ Kaggle.

#### Why each piece exists

| Piece | Why we need it |
|---|---|
| **HF account + write-scope token** | Every artifact (datasets, adapters, GGUF, RAG index) is pushed to HF Hub with `push_to_hub()`. The token authenticates those calls. *Write* scope is required — read-only tokens can't push. |
| **OpenAI API key** | Only Phase 3 (`kaggle_02_generate_dpo_pairs.ipynb`) and the LLM-as-judge eval in Phase 7 use it. Cost ≈ $5–10 total with `gpt-4o-mini`. Skip if you don't run those steps. |
| **GitHub repo** | This is the real backbone of the artifact flow. The Kaggle notebooks `pip install git+https://github.com/zubairashfaque/medalign-rlhf` to get the source — so any code change you want Kaggle to see must be pushed to GitHub *first*. |
| **Poetry (not bare venv)** | `poetry.lock` pins exact versions of ~150 transitive deps so Kaggle and laptop run the same code. Python 3.10–3.12 only (transformers/trl/peft constraint). |
| **`huggingface-cli login`** | Writes the token to `~/.cache/huggingface/token`. Every subsequent `push_to_hub()` / `load_dataset()` call reads it from there — no need to set env vars. |

#### Step by step

```bash
# 1. Create the HF account at https://huggingface.co (username: Zubairash)
# 2. Generate a WRITE-scope token at https://huggingface.co/settings/tokens
# 3. (Optional) Get an OpenAI key if running Phases 3 / 7-judge

# 4. Install Poetry (one-time, system-wide)
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc

# 5. Initialize the repo (the git remote was created via the GitHub web UI)
cd "/home/zubair-ashfaque/GitHubProject/medalign-rlhf"
git init && git add . && git commit -m "initial scaffolding"
git remote add origin https://github.com/zubairashfaque/medalign-rlhf.git
git push -u origin main

# 6. Install all Python deps from poetry.lock
poetry install

# 7. Cache the HF token locally
poetry run huggingface-cli login   # paste the write token when prompted
```

#### Verify

```bash
poetry run python -c "import medalign; print('package import ok')"
poetry run huggingface-cli whoami        # should print: Zubairash
git remote -v                            # should list the github URL
```

#### Common pitfalls
- **Poetry not on PATH** → install put it in `~/.local/bin`; add that to `PATH` in `~/.bashrc`.
- **`poetry install` builds `llama-cpp-python` and fails** with anaconda `compiler_compat/ld` errors about `GOMP_*` symbols. This is the issue documented under Phase 1 — `llama-cpp-python` is now an optional `serving` extra so the default install skips it. If you see this error you're on an older `pyproject.toml` — pull the latest commit.
- **HF push fails with 403** → token wasn't write-scope. Generate a new one and re-run `huggingface-cli login`.

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

**Goal:** Take base Mistral-7B-Instruct-v0.3, freeze it in 4-bit NF4, and train tiny LoRA adapter matrices on our 18k medical Q&A corpus so the model learns the medical answering style. Output: a ~50–100 MB adapter file pushed to HF Hub.

#### Why QLoRA (in plain English)
- **Q (Quantize)**: store the 7B base model in 4-bit NF4 (NormalFloat4) instead of 16-bit. Cuts VRAM from ~14 GB → ~4 GB. Fits on a single T4 (16 GB) with room for activations.
- **LoRA**: instead of updating all 7B weights, we freeze them and add small trainable rank-16 matrices on top of every attention/MLP projection. Only ~0.5–1% of params are trainable.
- **Result:** full fine-tuning quality with consumer-grade hardware, and the artifact is a tiny LoRA delta you can swap on/off.

#### Hyperparameter walkthrough (`configs/sft.yaml`)
| Setting | Value | Why |
|---|---|---|
| `base_model` | `mistralai/Mistral-7B-Instruct-v0.3` | Already instruction-tuned; we're just specializing the domain |
| `bnb_4bit_quant_type` | `nf4` | NF4 = normal-distribution-aware 4-bit; better than fp4 for transformers |
| `bnb_4bit_compute_dtype` | `bfloat16` | Matmul happens in bf16 even though weights are 4-bit |
| `lora.r` | `16` | Rank of LoRA decomposition; 16 is the sweet spot for 7B |
| `lora.alpha` | `32` | Scaling = α/r = 2.0 — standard "double the rank" rule |
| `target_modules` | all 7 proj layers | Apply LoRA to q/k/v/o + gate/up/down (full coverage > attention-only) |
| `learning_rate` | `2e-4` | High because we're only training small LoRA matrices |
| `lr_scheduler_type` | `cosine` | Smooth decay to 0 — standard for SFT |
| `num_train_epochs` | `2` | 2 passes over 18k = ~36k examples; enough for domain shift, not so much we overfit |
| `per_device_train_batch_size` | `2` | Tight for 16 GB T4 with 2048 seq length |
| `gradient_accumulation_steps` | `8` | Effective batch = 2 × 8 × 2 GPUs = 32 |
| `max_seq_length` | `2048` | USMLE questions can be long; cap saves memory |
| `optim` | `paged_adamw_8bit` | bitsandbytes' paged optimizer — offloads to CPU RAM when VRAM is tight |

#### What `run_sft()` does step by step (`src/medalign/training/sft.py`)
1. **Load tokenizer** for Mistral; set `pad_token = eos_token` (Mistral has no pad token).
2. **Load model in 4-bit NF4** via `BitsAndBytesConfig`, double quantization on, bf16 compute dtype.
3. **`prepare_model_for_kbit_training`** — enables gradient checkpointing, casts layer norms to fp32, plumbs gradients through frozen 4-bit weights.
4. **Inject LoRA** via `LoraConfig` + `get_peft_model`. Calls `print_trainable_parameters()` — should show ~0.5–1% trainable.
5. **Load dataset** `Zubairash/medalign-sft` (the 18k rows we just pushed).
6. **`SFTTrainer`** from TRL handles tokenization of the `text` column, masking, packing=False (we keep one example per sequence so loss is computed only on the assistant response).
7. **Train** for 2 epochs with cosine LR + paged AdamW.
8. **Push** the adapter weights + tokenizer to `Zubairash/medalign-sft-adapters`.

#### Kaggle notebook execution
1. Go to https://www.kaggle.com → New Notebook.
2. **Settings → Accelerator → GPU T4 x2** (free tier — 30 hrs/week).
3. **Settings → Internet → On** (required for HF Hub push).
4. **Add-ons → Secrets → add `HF_TOKEN`** (write scope, from huggingface.co/settings/tokens).
5. **File → Import Notebook** → upload `kaggle_notebooks/kaggle_01_sft_qlora.ipynb`.
6. **Run All**. The notebook will:
   - `pip install` the repo from GitHub
   - `wget` the SFT config
   - Call `run_sft("sft.yaml", hub_repo="Zubairash/medalign-sft-adapters")`
7. Walk away for 3–6 hours.

**Verify:**
- `Zubairash/medalign-sft-adapters` exists on HF Hub
- Trainable params printed ≈ 0.5–1% (e.g. `trainable: 41,943,040 / 7,283,449,856 = 0.58%`)
- Training loss decreases monotonically from ~1.8 → ~0.9 over the 2 epochs
- No NaN/inf in `grad_norm`

### Phase 3 — Generate DPO preference pairs on Kaggle (~2–4 h)

**Goal:** Build a dataset of `(prompt, chosen_answer, rejected_answer)` triples that we'll use to teach the SFT model *which* of two plausible answers is better. This is the secret sauce of RLHF/DPO — the model learns from comparisons, not just imitation.

#### Why DPO instead of full RLHF/PPO
Classic RLHF: train reward model → run PPO. Two models, complex hyperparameters, GPU-hungry.
DPO (Direct Preference Optimization): one objective, no reward model, no PPO loop. Mathematically equivalent to PPO with a KL-constrained reward when β is chosen right. Fits on Kaggle T4x2.

#### How preference pairs are generated (`kaggle_notebooks/kaggle_02_generate_dpo_pairs.ipynb`)
For each prompt:
1. Load base + SFT adapters (4-bit) — same model from Phase 2.
2. Sample answer **A at temperature 0.3** (conservative, high-likelihood).
3. Sample answer **B at temperature 0.9** (creative, more diverse).
4. Send `{prompt, A, B}` to **GPT-4o-mini** as a judge (cheap: ~$0.01 per pair). Judge picks the more accurate/safer answer.
5. Write triple `(prompt, chosen=winner, rejected=loser)` and push the dataset to `Zubairash/medalign-dpo-pairs`.

We sample ~2k prompts → ~2k pairs (some get filtered if A==B or judge can't decide).

#### Why two temperatures?
At T=0.3 the model usually gives a safe boilerplate answer; at T=0.9 it explores more — sometimes brilliant, sometimes wrong. The contrast is what makes the preference signal learnable. If both samples were identical we'd have nothing to learn from.

#### Kaggle notebook execution
1. New notebook, **GPU T4 x2**, **Internet On**.
2. **Secrets:** `HF_TOKEN` **and** `OPENAI_API_KEY` (judge calls).
3. Import `kaggle_notebooks/kaggle_02_generate_dpo_pairs.ipynb`. Run All.

**Verify:**
- `Zubairash/medalign-dpo-pairs` has ~1.5k–2k rows with columns `prompt, chosen, rejected`
- Spot-check 5–10 rows: the `chosen` column should be visibly more accurate/safer than `rejected`
- Estimated GPT-4o-mini cost: ~$5–10

### Phase 4 — DPO alignment on Kaggle (~3–6 h)

**Goal:** Train the SFT model on the preference pairs so it shifts probability mass toward "chosen" answers and away from "rejected" ones. Result: a second LoRA adapter that stacks on top of the SFT one.

#### The DPO loss (intuition)
Without math: for every (prompt, chosen, rejected) triple, the loss pushes the model to make `chosen` more likely than `rejected` *relative to a frozen reference model* (the unmodified SFT model). The β parameter controls how aggressively — too high and the model collapses, too low and nothing changes.

`β = 0.1` is the standard well-tested value from the DPO paper.

#### The PEFT-DPO trick
Normally DPO needs **two copies** of the 7B model (policy + frozen reference) = ~28 GB VRAM. Won't fit on T4.
**Trick:** load the base once. The "policy" = base + trainable SFT adapters. The "reference" = base with adapters *disabled* (just toggle them off at runtime). This is what `ref_model=None` triggers in TRL — it tells DPOTrainer "use the adapter-disabled version of the policy as the reference." Saves ~14 GB.

#### Hyperparameters (`configs/dpo.yaml`)
| Setting | Value | Why |
|---|---|---|
| `beta` | `0.1` | DPO paper default; controls KL pressure |
| `loss_type` | `sigmoid` | Original DPO loss (vs. ipo/hinge variants) |
| `learning_rate` | `5e-6` | **40x lower than SFT** — DPO is fragile, large LRs collapse the model |
| `num_train_epochs` | `1` | Single pass; more epochs typically hurt |
| `warmup_ratio` | `0.1` | Longer warmup than SFT — gentler start |
| `max_length` | `2048` | Total prompt + response cap |
| `max_prompt_length` | `1024` | Half the budget reserved for the answer |

#### What `run_dpo()` does (`src/medalign/training/dpo.py`)
1. Load tokenizer + base model in 4-bit NF4.
2. **Attach SFT adapters** via `PeftModel.from_pretrained(base, "Zubairash/medalign-sft-adapters", is_trainable=True)`. The `is_trainable=True` flag is essential — without it the LoRA weights are frozen and DPO does nothing.
3. Load `Zubairash/medalign-dpo-pairs` (expects `prompt, chosen, rejected` columns).
4. `DPOConfig` + `DPOTrainer(ref_model=None)` → train.
5. Push merged-adapter weights to `Zubairash/medalign-dpo-adapters`.

#### Kaggle notebook execution
1. New notebook, **GPU T4 x2**, **Internet On**.
2. **Secret:** `HF_TOKEN`.
3. Import `kaggle_notebooks/kaggle_03_dpo_alignment.ipynb`. Run All.

**Verify:**
- `Zubairash/medalign-dpo-adapters` exists on HF Hub
- TRL logs `rewards/chosen - rewards/rejected` (the "reward margin"). This should be **positive and growing** over training. If it's negative or flat, DPO is failing — likely a too-high LR or bad pairs.
- Final loss < initial loss (DPO loss is ~0.69 = ln 2 at random init; should drop to ~0.4–0.5)

### Phase 5 — RAG index build

**Goal:** Build a searchable index of medical passages so that at inference time we can fetch relevant context and inject it into the prompt. This is **Retrieval-Augmented Generation (RAG)** — instead of relying on what the model memorized, it grounds answers in real documents and cites them.

#### Why hybrid (dense + sparse + rerank)?
| Method | Strength | Weakness |
|---|---|---|
| **Dense (BGE embeddings)** | Captures meaning ("chest pain" ≈ "thoracic discomfort") | Misses rare jargon and exact terminology |
| **Sparse (BM25 keyword)** | Catches exact medical terms, drug names, ICD codes | Blind to synonyms/paraphrases |
| **Cross-encoder reranker** | Joint scoring of (query, doc) — much more accurate | Too slow to run on every doc — only on the top-20 |

The pipeline runs dense + sparse in parallel, fuses with **Reciprocal Rank Fusion (RRF)** to get ~20 candidates, then reranks with a cross-encoder to pick the final top-5. This combines the best of all worlds.

#### Configuration (`configs/rag.yaml`)
| Knob | Value | Meaning |
|---|---|---|
| `corpus.dataset_id` | `qiaojin/PubMedQA` (`pqa_artificial`) | ~211k synthetic PubMed Q&A passages — our retrieval corpus |
| `embedding.model` | `BAAI/bge-large-en-v1.5` | Top open-source dense retriever (1024-dim) |
| `embedding.normalize` | `true` | Unit-length vectors → cosine ≡ inner product → use FAISS `IndexFlatIP` |
| `top_k_dense` / `top_k_sparse` | 50 / 50 | Each retriever returns its top-50 |
| `rrf_k` | 60 | RRF smoothing constant (paper default) |
| `final_top_k` | 5 | Top-5 fed into the prompt |
| `reranker.model` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Tiny but strong reranker |

#### What `build_rag_index.py` does (in 4 steps)
1. **Load** the PubMedQA `pqa_artificial` split. Each row's `context` field contains a list of passage strings — we join them.
2. **Embed** every passage with BGE-large in batches of 64, normalized. Output shape `(N, 1024) float32`.
3. **Build FAISS index**: `IndexFlatIP` (exact inner product, no quantization — fine at this scale on CPU). Save to `data/rag/faiss.index`.
4. **Build BM25 index**: tokenize each passage (lowercase split), feed to `BM25Okapi`, pickle to `data/rag/bm25.pkl`.
5. **Write metadata**: `data/rag/meta.jsonl` with `{id, text}` for each passage so we can recover the source text given an index ID.

These three files are everything `HybridRetriever` needs at serving time.

#### Where to run it — Kaggle (recommended)
The original plan was "laptop, ~20 min" but BGE-large is a 335M-param transformer; embedding 211k passages is GPU-bound. On a small consumer GPU it OOMs (e.g. 3.6 GB GPU: `CUDA out of memory. Tried to allocate 512.00 MiB.`); on CPU it takes 4–8 hours. **Kaggle T4 finishes the same job in ~10–20 min.**

So Phase 5 also moves to Kaggle, then the artifacts are mirrored to HF Hub and pulled back to the laptop where serving runs.

**Kaggle notebook execution:**
1. New notebook, **GPU T4 x1** (one is enough), **Internet On**.
2. **Secret:** `HF_TOKEN`.
3. Import `kaggle_notebooks/kaggle_04_build_rag_index.ipynb`. Run All.
4. The notebook:
   - Installs the repo + dependencies
   - Pulls `configs/rag.yaml` + `scripts/build_rag_index.py` from GitHub
   - Runs the build (BGE-large auto-uses CUDA)
   - Pushes the 3 files to a new HF dataset `Zubairash/medalign-rag-index`

**Then pull to laptop** (so Phases 7/8 can use it locally):
```bash
poetry run huggingface-cli download Zubairash/medalign-rag-index \
  --repo-type dataset --local-dir data/rag
ls -lh data/rag/
```

**Verify:** `data/rag/faiss.index` (~800 MB = 211k × 1024 × 4 bytes), `data/rag/bm25.pkl`, `data/rag/meta.jsonl` all present locally.

#### Laptop-only fallback (if you can't use Kaggle)
Force CPU encoding by editing `scripts/build_rag_index.py`:
```python
embedder = SentenceTransformer(cfg["embedding"]["model"], device="cpu")
```
Then `poetry run python scripts/build_rag_index.py`. Slow (~4–8 h) but no quality loss. Or swap in `BAAI/bge-small-en-v1.5` in `configs/rag.yaml` — fits any GPU and is much faster, with a small recall drop.

### Phase 6 — Quantization (laptop, ~1–2 h)

**Goal:** Take the trained DPO adapters, fuse them into the base model, and produce 5 quantized GGUF files at different size/quality trade-offs. The result is a single self-contained file per quantization level that runs anywhere `llama.cpp` runs — no Python, no GPU required.

#### Why GGUF
- **Single-file format** with weights + tokenizer + chat template baked in.
- **Memory-mapped** at load time → starts in seconds, not minutes.
- **Portable**: same `.gguf` runs on CPU (laptop), Metal (Mac), CUDA (server), Android, even browsers via WASM.
- **Quantizable** down to 2 bits per weight with minimal quality loss when paired with `imatrix`.

#### Why 5 levels (Pareto exploration)

| Level | Bits/weight | File size (7B) | Quality | When to use |
|---|---|---|---|---|
| `Q8_0` | 8.5 | ~7.7 GB | Near-lossless | Reference / quality ceiling |
| `Q6_K` | 6.6 | ~5.9 GB | Indistinguishable from fp16 in eval | Best quality you'd actually ship |
| `Q5_K_M` | 5.7 | ~5.1 GB | <1% perplexity hit | Good balance |
| **`Q4_K_M`** | 4.8 | ~4.4 GB | ~2% perplexity hit | **Default** — best size/quality knee |
| `Q4_K_S` | 4.6 | ~4.1 GB | ~3% perplexity hit | Smallest that still works on 8 GB RAM |

We build all five so Phase 7 can plot the Pareto curve and pick the right one for each deployment target.

#### What `imatrix` is, in plain words
An "importance matrix" is a record of which weights light up most when you pass *real* text through the unquantized model. The quantizer reads this and gives more bits to high-impact weights and fewer bits to dead ones. Without imatrix every weight gets the same coarse quantization → unnecessary accuracy loss. With imatrix, accuracy loss drops by ~30–50% at the same bit-width.

**Why a *medical* calibration corpus:** if we calibrated on Wikipedia, the imatrix would tell the quantizer "preserve weights that fire on Roman emperors and football scores" — irrelevant for our use case. By feeding it 2k PubMedQA + 2k MedQA chunks, we tell it "preserve weights that fire on drug names, anatomy, dosages." `scripts/build_calibration.py` produces `data/calibration_medical.txt`.

#### Configuration (`configs/quant.yaml`)
| Knob | Value | Meaning |
|---|---|---|
| `source.dpo_adapter_repo` | `Zubairash/medalign-dpo-adapters` | Phase 4 output to merge in |
| `source.merged_dir` | `outputs/merged` | Where the merged fp16 HF checkpoint lands |
| `llama_cpp.repo_url` | `https://github.com/ggerganov/llama.cpp` | Cloned + built on first run |
| `quantization.levels` | `[Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_K_S]` | Pareto sweep |
| `quantization.imatrix.calibration_text` | `data/calibration_medical.txt` | Corpus from `build_calibration.py` |

#### What the scripts do, step by step

**`scripts/build_calibration.py`** (1 file, runs first):
1. Download 2,000 PubMedQA `pqa_artificial` rows and 2,000 medalpaca `medical_meadow_medqa` rows.
2. From PubMedQA take the joined `context` field; from medqa format as `Q: …\nA: …`.
3. Concatenate with blank-line separators → write to `data/calibration_medical.txt` (~2 MB).

**`scripts/quantize_gguf.py`** (the main quantization pipeline):
1. **`merge_adapter()`** (`scripts/quantize_gguf.py:11`) — load base Mistral in fp16, attach DPO adapters via `PeftModel.from_pretrained(...)`, call `.merge_and_unload()` to bake the LoRA deltas into the base weights, save to `outputs/merged/`. *After this step the model is no longer LoRA — it's a regular HF checkpoint with the medical knowledge fused in.*
2. **Build llama.cpp** if `third_party/llama.cpp/` doesn't exist: `git clone` then `make -j`. Builds `llama-imatrix`, `llama-quantize`, and `convert_hf_to_gguf.py`.
3. **Convert HF → GGUF fp16**: `python convert_hf_to_gguf.py outputs/merged/ --outfile outputs/gguf/model-f16.gguf --outtype f16` → ~14 GB.
4. **Compute imatrix**: `llama-imatrix -m model-f16.gguf -f data/calibration_medical.txt -o outputs/gguf/imatrix.dat`. Takes ~10–20 min on CPU.
5. **Loop quantize**: for each level in the config, run `llama-quantize --imatrix imatrix.dat model-f16.gguf model-{LEVEL}.gguf {LEVEL}`.

#### Run

```bash
poetry run python scripts/build_calibration.py
poetry run python scripts/quantize_gguf.py
```

#### Verify
- `outputs/merged/` contains `pytorch_model*.safetensors` + `tokenizer.json`
- `outputs/gguf/` contains all 5 `.gguf` files plus `imatrix.dat` and `model-f16.gguf`
- Sizes ≈ 7.7 / 5.9 / 5.1 / 4.4 / 4.1 GB
- Sanity-check one file: `llama.cpp/llama-cli -m outputs/gguf/model-Q4_K_M.gguf -p "What is metformin?" -n 50`

#### Common pitfalls
- **`make` of llama.cpp fails with the same `GOMP_*` linker error from Phase 1.** Same root cause (anaconda `compiler_compat/ld`). Workarounds:
  ```bash
  cd third_party/llama.cpp && make GGML_OPENMP=0 -j
  # or in a clean shell:
  unset LD_LIBRARY_PATH; PATH=/usr/bin:$PATH make -j
  ```
- **Out of disk space**: the merged fp16 (~14 GB) + 5 quantized files + imatrix easily exceed 40 GB. Make sure `outputs/` lives on a partition with headroom.
- **`merge_and_unload` OOMs on a small GPU**: `quantize_gguf.py` loads the merge in fp16 on whatever device PyTorch picks. Force CPU if needed by editing `merge_adapter` to pass `device_map="cpu"`.

### Phase 7 — Benchmarks (laptop)

**Goal:** Numerically prove that each pipeline stage actually improved the model. Same test sets, same prompts, four model variants — and we want to see monotonic gains: **Base < SFT < DPO < DPO+RAG**.

#### How accuracy is measured
`scripts/run_benchmarks.py:25 evaluate()` does a naïve **substring match**: for each example, generate an answer, check if the gold string appears anywhere in the output. Fast, deterministic, noisy. Good for quick sanity checks. For higher-fidelity scoring, swap in the qualitative graders described below.

#### Eval sets
| Loader | Dataset | Subset | Format | Gold field |
|---|---|---|---|---|
| `load_pubmedqa` | `qiaojin/PubMedQA` | `pqa_labeled` (~1k) | yes/no/maybe | `final_decision` |
| `load_medqa` | `GBaker/MedQA-USMLE-4-options` | `test` | 4-option MCQ | `answer` |
| `load_medmcqa` | `openlifescienceai/medmcqa` | `validation` | MCQ | (loader exists, not wired into `main`) |

Each run benchmarks 200 samples per dataset (cap inside `main()`).

#### Generator wiring
`make_llama_cpp_generator(gguf_path)` (`scripts/run_benchmarks.py:36`) wraps `llama_cpp.Llama` with **greedy decoding** (`temperature=0.0`), `max_tokens=128`, stops at `</s>` and `<|im_end|>`. Greedy ensures the same model gives the same answer twice — required for fair comparison.

#### `--use-rag` flag
When set:
1. Load `configs/rag.yaml`.
2. Instantiate `HybridRetriever` (FAISS + BM25 + cross-encoder reranker).
3. For each question, fetch top-5 passages and prepend them as `[1] passage…\n[2] passage…\n` before the question.

The model is then expected to cite as `[1]`, `[2]` in its answer.

#### Concrete example

**Without RAG** (PubMedQA prompt):
```
Question: Does aspirin reduce the risk of myocardial infarction in healthy adults?
Answer:
```

**With RAG** (same question, top-2 context shown):
```
Context:
[1] A meta-analysis of 6 randomized trials found that low-dose aspirin reduced…
[2] The Physicians' Health Study showed a 44% reduction in first MI in men…
Question: Does aspirin reduce the risk of myocardial infarction in healthy adults?
Answer:
```

#### Qualitative metrics (optional, more expensive)
- **`medalign.eval.llm_judge.score_answer`** — calls GPT-4o-mini with a rubric rating accuracy / safety / completeness / citations on a 1–5 scale. Use for nuanced answers where substring match is too coarse.
- **`medalign.eval.ragas_eval.evaluate_rag`** — wraps RAGAS metrics: `faithfulness` (does the answer match the context?), `answer_relevancy`, `context_precision`, `context_recall`. Only meaningful when `--use-rag` is on.

#### Prerequisite
`llama-cpp-python` must be installed. Use the OpenMP-disabled command from Phase 1's known-issue block:
```bash
CMAKE_ARGS="-DGGML_OPENMP=OFF" poetry run pip install llama-cpp-python==0.3.20
```

#### Run

```bash
# Base / SFT / DPO would each require swapping the merged checkpoint and re-quantizing,
# so the typical workflow is to benchmark only the final DPO model with and without RAG.
poetry run python scripts/run_benchmarks.py \
  --gguf outputs/gguf/model-Q4_K_M.gguf --out outputs/q4_norag.csv

poetry run python scripts/run_benchmarks.py \
  --gguf outputs/gguf/model-Q4_K_M.gguf --use-rag --out outputs/q4_rag.csv
```

Aggregate the CSVs into the README results table comparing base / SFT / DPO / DPO+RAG.

#### Verify
- Both CSVs land in `outputs/`
- Each row: `benchmark, accuracy` with values in [0,1]
- Monotonic improvement: PubMedQA accuracy with RAG should be at least 5–10 points higher than without

### Phase 8 — Serving (laptop)

**Goal:** Wrap the quantized model + RAG retriever in a single process that exposes both an HTTP API (FastAPI) and an optional web UI (Gradio). Same `answer_fn` powers both, so behavior is identical.

#### Architecture

```
                  ┌──────────────────────────┐
HTTP POST /ask ──►│                          │
                  │      build_app()         │
Gradio UI     ───►│  ┌────────────────────┐  │
                  │  │     answer_fn      │  │
                  │  │  ─────────────     │  │
                  │  │  (optional) RAG    │  │
                  │  │  HybridRetriever   │  │
                  │  │       │            │  │
                  │  │       ▼            │  │
                  │  │ ChatML prompt      │  │
                  │  │       │            │  │
                  │  │       ▼            │  │
                  │  │  llama_cpp.Llama   │  │
                  │  │  (Q4_K_M.gguf)     │  │
                  │  └────────────────────┘  │
                  └──────────────────────────┘
```

`build_app(gguf_path, use_rag)` (`scripts/serve.py:7`) returns `(FastAPI app, answer_fn)`. The `answer_fn` is reused by both the FastAPI route and Gradio so a bug fix in one fixes both.

#### Prompt template

**Without RAG** (`scripts/serve.py:25`):
```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

**With RAG** (`scripts/serve.py:30–34`):
```
<|im_start|>system
Answer using ONLY the context. Cite as [#].<|im_end|>
<|im_start|>user
Context:
[1] {passage 1}
[2] {passage 2}
...
Question: {question}<|im_end|>
<|im_start|>assistant
```

The system instruction forces the model to ground its answer in the context and cite sources — which is exactly what we want for medical Q&A where hallucinations are dangerous.

#### Sampling
- `temperature=0.2` — slightly creative but mostly deterministic
- `max_tokens=512` — enough room for a paragraph + citations
- `stop=["<|im_end|>"]` — clean termination at the assistant's turn end

#### `/ask` endpoint
Pydantic schema:
```python
class Query(BaseModel):
    question: str
```
Returns:
```json
{
  "answer": "First-line treatment for type 2 diabetes is metformin [1]...",
  "sources": ["A meta-analysis of...", "Per ADA 2024 guidelines..."]
}
```
`sources` are 200-character previews of the retrieved passages (when RAG is on).

#### Gradio mode
Two `Textbox` outputs (Answer + Sources), launched on the same port. Same `answer_fn` invoked under the hood.

#### Docker
The `Dockerfile` uses Python 3.10-slim, runs `poetry install`, and the default CMD launches `serve.py --gguf /models/model-Q4_K_M.gguf --use-rag` on port 8000. Mount the GGUF directory at runtime:
```bash
docker build -t medalign .
docker run -p 8000:8000 -v $PWD/outputs/gguf:/models medalign
```

#### Run

```bash
# FastAPI only
poetry run python scripts/serve.py --gguf outputs/gguf/model-Q4_K_M.gguf --use-rag

# In another terminal:
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What are first-line treatments for type 2 diabetes?"}'

# Gradio UI
poetry run python scripts/serve.py \
  --gguf outputs/gguf/model-Q4_K_M.gguf --use-rag --gradio --port 7860
```

#### Verify
- HTTP 200 from `/ask`
- Response JSON has a non-empty `answer` and at least one `[1]` citation when `--use-rag` is on
- Latency ~5–15 sec on CPU laptop, sub-second on GPU

### Phase 9 — Deploy demo to HuggingFace Spaces

**Goal:** Put a public, shareable medical Q&A demo online with zero infrastructure cost. Reviewers click a URL and try the model — no install, no auth.

#### Why HF Spaces (vs. Render / Fly / Vercel)
- **Free CPU basic tier** (16 GB RAM, 2 vCPU) — enough for Q4_K_M GGUF at ~5–10 tok/s
- **Native Gradio SDK** — one-click deploy from a `gradio.Interface`
- **Lives next to the model** — `huggingface.co/spaces/Zubairash/...` pulls the GGUF straight from your HF Hub repo

#### What to push to the Space
| File | Purpose |
|---|---|
| `app.py` | 3-line wrapper: `from serve import build_app; _, answer_fn = build_app("model-Q4_K_M.gguf", use_rag=True); gr.Interface(...).launch()` |
| `requirements.txt` | `llama-cpp-python`, `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `gradio`, `huggingface_hub` |
| `model-Q4_K_M.gguf` | Via Git LFS, or pull at boot from `Zubairash/medalign-gguf` |
| `data/rag/*` | Pull from `Zubairash/medalign-rag-index` at boot to avoid bloating the Space repo |

#### Step by step
1. https://huggingface.co/new-space → name `medalign-demo`, SDK **Gradio**, hardware **CPU basic**.
2. `git clone https://huggingface.co/spaces/Zubairash/medalign-demo`
3. Copy `serve.py`, write the minimal `app.py`, commit + push (LFS for the GGUF).
4. Watch the Space build log; first build takes ~5 min.
5. Add the Space URL to `README.md`.

#### Verify
- Space URL loads the Gradio UI
- Sample question "What is metformin used for?" returns a coherent answer with at least one citation

### Phase 10 — Polish

**Goal:** Make the repo presentable as a portfolio piece. Tests pass, diagrams render, results table filled in, README is the front door.

#### Checklist with details

1. **Run tests** — `make test` or `poetry run pytest tests/unit/ -v`. Should be green:
   - `test_format.py` covers `to_chatml` + `dedup_minhash`
   - `test_rrf.py` covers `reciprocal_rank_fusion`

2. **Architecture diagram** — paste this Mermaid block into the README:
   ```mermaid
   flowchart LR
       A[medalpaca datasets] --> B[Phase 1: dedup + ChatML]
       B --> C[Phase 2: SFT QLoRA Kaggle]
       C --> D[Phase 3: DPO pair gen + GPT-4o judge]
       D --> E[Phase 4: DPO alignment Kaggle]
       E --> F[Phase 6: merge + GGUF quantize]
       G[PubMedQA corpus] --> H[Phase 5: BGE+BM25 RAG index]
       F --> I[Phase 8: FastAPI + Gradio]
       H --> I
       I --> J[Phase 9: HF Space demo]
   ```

3. **Pareto chart** — after Phase 7, generate the size-vs-accuracy plot:
   ```python
   from medalign.quantization.benchmark import benchmark_gguf_variants, plot_pareto
   rows = benchmark_gguf_variants("outputs/gguf", eval_fn=my_eval_fn, output_csv="outputs/pareto.csv")
   plot_pareto(rows, out_png="docs/pareto.png")
   ```
   Embed `docs/pareto.png` in the README.

4. **Fill the results table** in `README.md` from the Phase 7 CSVs:
   ```
   | Variant            | PubMedQA | MedQA-USMLE | MedMCQA |
   | Base (Mistral-7B)  | …        | …           | —       |
   | + SFT              | …        | …           | —       |
   | + DPO              | …        | …           | —       |
   | + DPO + RAG        | …        | …           | —       |
   ```

5. **Final commit + push** — `git add . && git commit -m "Phase 10: polish (diagrams, results, tests)" && git push origin main`.

#### Verify
- `make test` exits 0
- README renders the Mermaid diagram, Pareto image, and a complete results table
- HF Space demo URL is linked in the README and works
- CI workflow on the latest commit is green

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
