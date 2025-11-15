# src/train.py
import os
import argparse
import logging
import torch
import mlflow
import mlflow.transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ------- CONFIG -------
MODEL_NAME = "google/gemma-2b"   # gated model
PROCESSED_DATA_DIR = "data/processed"
MODEL_OUTPUT_DIR = "model-output"

# ------- Logging -------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma-2B QLoRA fine-tuning")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)  # ✅ T4 safe default
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--mlflow-experiment", type=str, default="SumThisConvo-Finetuning")

    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")

    return parser.parse_args()


def load_jsonl(split):
    """Loads train/validation processed JSONL and renames formatted_text → text."""
    path = os.path.join(PROCESSED_DATA_DIR, f"{split}_processed.jsonl")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    ds = load_dataset("json", data_files=path)
    ds_split = ds[list(ds.keys())[0]]

    if "formatted_text" in ds_split.column_names:
        ds_split = ds_split.rename_column("formatted_text", "text")
    return ds_split


def main():
    args = parse_args()
    logger.info("=== Starting Phase 2: LoRA Training on Gemma-2B ===")

    # -------- Load Dataset --------
    logger.info("Loading dataset...")
    train_ds = load_jsonl("train")
    eval_ds = load_jsonl("validation")

    # -------- BitsAndBytes (4-bit QLoRA) --------
    logger.info("Configuring BitsAndBytes 4-bit quantization (QLoRA)...")
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # -------- Load Model + Tokenizer --------
    logger.info(f"Loading model: {MODEL_NAME}")
    trust = args.trust_remote_code

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=trust,
    )
    model.config.use_cache = False  # required for training

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=trust)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # ✅ required for Gemma chat models

    # -------- Prepare model for LoRA --------
    logger.info("Applying LoRA to QLoRA model...")
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # ✅ correct for Gemma
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    # -------- MLflow --------
    logger.info(f"Setting MLflow experiment → {args.mlflow_experiment}")
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.transformers.autolog()

    # -------- TrainingArguments --------
    logger.info("Configuring TrainingArguments for Colab T4...")

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_OUTPUT_DIR, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,

        gradient_accumulation_steps=2,         # ✅ helps T4 memory
        gradient_checkpointing=True,           # ✅ HUGE VRAM saver
        optim="paged_adamw_32bit",

        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.01,

        max_grad_norm=0.3,
        max_steps=-1,

        fp16=True,   # ✅ T4 supports fp16
        bf16=False,  # ✅ T4 does NOT support bf16

        logging_steps=50,
        save_steps=500,
        eval_steps=500,

        evaluation_strategy="steps",
        report_to="mlflow",

        remove_unused_columns=False,
        push_to_hub=False,
    )

    # -------- Trainer --------
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,   # ✅ must remain False for Gemma
        args=training_args,
    )

    # -------- Training --------
    logger.info("=== Starting Training on GPU ===")
    trainer.train()
    logger.info("=== Training Finished ===")

    # -------- Save LoRA adapter --------
    final_path = os.path.join(MODEL_OUTPUT_DIR, "final_adapter")
    os.makedirs(final_path, exist_ok=True)

    logger.info(f"Saving LoRA adapter to {final_path}")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # -------- Optional: Push to hub --------
    if args.push_to_hub:
        if not args.hub_repo_id:
            logger.error("push-to-hub requested but --hub-repo-id missing.")
        else:
            from huggingface_hub import upload_folder, HfApi
            hf_token = os.environ.get("HF_TOKEN")

            if not hf_token:
                logger.error("HF_TOKEN missing. Cannot push.")
            else:
                logger.info(f"Pushing adapter to HF Hub → {args.hub_repo_id}")
                upload_folder(
                    folder_path=final_path,
                    repo_id=args.hub_repo_id,
                    token=hf_token
                )

    logger.info("=== Phase 2 Completed Successfully ===")


if __name__ == "__main__":
    main()
