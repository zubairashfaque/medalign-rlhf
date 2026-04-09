"""DPO alignment entrypoint. Uses TRL DPOTrainer with frozen SFT reference."""
from __future__ import annotations
import yaml
from pathlib import Path


def run_dpo(config_path: str, hub_repo: str | None = None) -> str:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig
    from trl import DPOTrainer, DPOConfig

    cfg = yaml.safe_load(Path(config_path).read_text())
    m, dpo_c, t, d = cfg["model"], cfg["dpo"], cfg["training"], cfg["data"]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(m["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        m["base_model"],
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # Policy = base + SFT adapters (trainable LoRA on top)
    policy = PeftModel.from_pretrained(base, m["sft_adapter_repo"], is_trainable=True)

    dataset = load_dataset(d["dataset_id"], split="train")
    # Expected columns: prompt, chosen, rejected

    args = DPOConfig(
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
        beta=dpo_c["beta"],
        loss_type=dpo_c["loss_type"],
        max_length=dpo_c["max_length"],
        max_prompt_length=dpo_c["max_prompt_length"],
        report_to="none",
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,  # PEFT path: TRL uses adapter-disabled base as reference
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(t["output_dir"])

    if hub_repo:
        policy.push_to_hub(hub_repo, private=False)
        tokenizer.push_to_hub(hub_repo)
    return t["output_dir"]
