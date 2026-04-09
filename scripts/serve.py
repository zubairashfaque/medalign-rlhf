"""FastAPI + Gradio serving for a quantized GGUF MedAlign model with optional RAG."""
from __future__ import annotations
import argparse, yaml
from pathlib import Path


def build_app(gguf_path: str, use_rag: bool):
    from fastapi import FastAPI
    from pydantic import BaseModel
    from llama_cpp import Llama

    llm = Llama(model_path=gguf_path, n_ctx=4096, n_gpu_layers=0)
    retriever = None
    if use_rag:
        from medalign.rag.hybrid import HybridRetriever
        retriever = HybridRetriever(yaml.safe_load(Path("configs/rag.yaml").read_text()))

    app = FastAPI(title="MedAlign")

    class Query(BaseModel):
        question: str

    def answer(question: str) -> dict:
        sources = []
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        if retriever is not None:
            docs = retriever.search(question)
            sources = [d["text"][:200] for d in docs]
            ctx = "\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
            prompt = (
                "<|im_start|>system\nAnswer using ONLY the context. Cite as [#].<|im_end|>\n"
                f"<|im_start|>user\nContext:\n{ctx}\n\nQuestion: {question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        out = llm(prompt, max_tokens=512, temperature=0.2, stop=["<|im_end|>"])
        return {"answer": out["choices"][0]["text"], "sources": sources}

    @app.post("/ask")
    def ask(q: Query):
        return answer(q.question)

    return app, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--use-rag", action="store_true")
    ap.add_argument("--gradio", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app, answer_fn = build_app(args.gguf, args.use_rag)

    if args.gradio:
        import gradio as gr
        def ui(q):
            r = answer_fn(q)
            return r["answer"], "\n\n".join(r["sources"])
        gr.Interface(
            ui,
            inputs=gr.Textbox(label="Medical question"),
            outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources")],
            title="MedAlign",
        ).launch(server_port=args.port)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
