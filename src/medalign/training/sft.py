"""SFT QLoRA training entrypoint. Designed to be called from a Kaggle notebook.

Usage (in Kaggle):
    from medalign.training import run_sft
    run_sft("configs/sft.yaml", hub_repo="user/medalign-sft-adapters")
"""
from __future__ import annotations
import yaml
from pathlib import Path


def run_sft(config_path: str, hub_repo: str | None = None) -> str:
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    cfg = yaml.safe_load(Path(config_path).read_text())
    m, l, t, d = cfg["model"], cfg["lora"], cfg["training"], cfg["data"]

    bnb = BitsAndBytesConfig(
        load_in_4bit=m["load_in_4bit"],
        bnb_4bit_quant_type=m["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, m["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(m["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        m["base_model"],
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        target_modules=l["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    dataset = load_dataset(d["dataset_id"], split="train")

    args = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        optim=t["optim"],
        report_to="none",
        dataset_text_field="text",
        max_seq_length=t["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(t["output_dir"])

    if hub_repo:
        model.push_to_hub(hub_repo, private=False)
        tokenizer.push_to_hub(hub_repo)
    return t["output_dir"]
