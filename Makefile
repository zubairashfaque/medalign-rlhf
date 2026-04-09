.PHONY: data rag-index quantize eval serve test

data:
	poetry run python scripts/prepare_sft_data.py

rag-index:
	poetry run python scripts/build_rag_index.py

quantize:
	poetry run python scripts/quantize_gguf.py

eval:
	poetry run python scripts/run_benchmarks.py

serve:
	poetry run python scripts/serve.py

test:
	poetry run pytest tests/unit/ -v
