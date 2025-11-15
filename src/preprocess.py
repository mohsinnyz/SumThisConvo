# src/preprocess.py
import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

# --- Constants ---
# Using a model on the Hub to ensure we have the correct tokenizer
TOKENIZER_NAME = "google/gemma-2b"

# Define the prompt template for summarization
# Gemma uses a specific chat template (<start_of_turn> / <end_of_turn>)
PROMPT_TEMPLATE = """<start_of_turn>user
Summarize the following conversation:
{dialogue}
<end_of_turn>
<start_of_turn>model
{summary}
<end_of_turn>"""

# Define paths (relative to project root)
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"


def fetch_and_save_raw_data():
    """
    Downloads the SAMSum dataset and saves the splits
    as .jsonl files in the data/raw directory.
    """
    print("-> Fetching raw SAMSum dataset...")
    
    # Create raw data directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    try:
        # Load the dataset from Hugging Face Hub
        dataset = load_dataset("knkarthick/samsum")
        
        # Define which splits to save
        splits = {"train": dataset["train"], 
                  "validation": dataset["validation"], 
                  "test": dataset["test"]}
        
        for split_name, data in splits.items():
            file_path = os.path.join(RAW_DATA_DIR, f"{split_name}.jsonl")
            
            # Convert to pandas DataFrame, then to .jsonl
            df = data.to_pandas()
            df.to_json(file_path, orient="records", lines=True)
            print(f"   Saved {split_name} split to {file_path}")
            
    except Exception as e:
        print(f"Error fetching or saving raw data: {e}")
        return False
        
    return True


def process_and_tokenize_data():
    """
    Loads the raw .jsonl files, applies the prompt template,
    tokenizes the data, and saves to data/processed.
    """
    print(f"\n-> Processing and tokenizing data using '{TOKENIZER_NAME}'...")
    
    # Create processed data directory
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        # Set padding token if not present (Gemma often doesn't have one)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        splits = ["train", "validation", "test"]
        
        for split_name in splits:
            raw_file_path = os.path.join(RAW_DATA_DIR, f"{split_name}.jsonl")
            processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}_tokenized.json")
            
            print(f"   Processing {raw_file_path}...")
            
            # Load the raw .jsonl
            df = pd.read_json(raw_file_path, lines=True)
            
            # --- Apply the prompt template ---
            # We filter out any missing data just in case
            df = df.dropna(subset=["dialogue", "summary"])
            
            # This is where we format the data for the model
            df["formatted_text"] = df.apply(
                lambda row: PROMPT_TEMPLATE.format(
                    dialogue=row["dialogue"], 
                    summary=row["summary"]
                ),
                axis=1
            )
            
            # --- Tokenize the data ---
            # We'll tokenize the formatted text.
            # This is a basic example; for training, we'd add truncation
            # and padding strategies in the training script itself.
            # For now, let's just save the text.
            
            # For simplicity, we'll save the formatted text.
            # The 'train.py' script will be responsible for loading this
            # and tokenizing on the fly (which is more efficient).
            
            processed_data = df[["formatted_text"]]
            
            # Save the processed data
            processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}_processed.jsonl")
            processed_data.to_json(processed_file_path, orient="records", lines=True)
            
            print(f"   Saved processed data to {processed_file_path}")

    except Exception as e:
        print(f"Error processing data: {e}")
        return False
        
    return True

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Data pipeline for SAMSum summarization.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step (if raw data already exists)"
    )
    args = parser.parse_args()

    print("--- Starting Data Pipeline (Phase 1) ---")
    
    if not args.skip_download:
        if not fetch_and_save_raw_data():
            print("Failed to fetch raw data. Exiting.")
            return
    else:
        print("-> Skipping raw data download.")
    
    if not process_and_tokenize_data():
        print("Failed to process data. Exiting.")
        return
        
    print("\n--- Data Pipeline (Phase 1) Completed Successfully ---")


if __name__ == "__main__":
    main()