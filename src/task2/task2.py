import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate # Hugging Face Evaluate library
import nltk
import numpy as np

# --- Configuration ---
# BASE_DIR = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Correctly resolve the data directory
DATA_DIR = os.path.join(BASE_DIR, "dataset/task2")

# DATA_DIR = os.path.join(BASE_DIR, '../../dataset/task2/') # Directory containing train/val CSVs

MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models') # Directory to save trained models
RANDOM_SEED = 42

# Choose model variants - smaller ones suitable for Colab/Kaggle free tier
# Other options: 'facebook/bart-large', 't5-base', 't5-large' (if resources allow)
#                 'google/flan-t5-small', 'google/flan-t5-base' (often perform well)
#                 'distilbart-cnn-6-6' (smaller BART variant)
BART_MODEL_NAME = "facebook/bart-base"
T5_MODEL_NAME = "t5-small" # Consider 'google/flan-t5-small' as a strong alternative

# Training Hyperparameters (adjust based on your GPU memory)
MAX_SOURCE_LENGTH = 512  # Max length of input sequence (Social Media Post)
MAX_TARGET_LENGTH = 128  # Max length of output sequence (Normalized Claim)
# Reduce batch size if you encounter CUDA out-of-memory errors
PER_DEVICE_TRAIN_BATCH_SIZE = 4 # Try 2 or 1 if 4 is too high
PER_DEVICE_EVAL_BATCH_SIZE = 8  # Can usually be higher than train batch size
# Increase accumulation steps to simulate larger batch size if lowering batch size
GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch size = batch_size * accumulation_steps
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3 # Start with a small number, increase if needed/possible
WEIGHT_DECAY = 0.01
FP16 = torch.cuda.is_available() # Use mixed precision if CUDA is available

# --- Ensure NLTK data is downloaded (for ROUGE metric) ---
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt", quiet=True)

# --- Load Data ---
print("Loading preprocessed data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_data.csv'))
# test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv')) # Load if needed for final eval

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
# test_dataset = Dataset.from_pandas(test_df)

# Optional: Create a DatasetDict if you want to group them
# dataset_dict = DatasetDict({
#     'train': train_dataset,
#     'validation': val_dataset,
#     'test': test_dataset
# })

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# --- Metric ---
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    """Computes ROUGE scores for sequence-to-sequence models."""
    predictions, labels = eval_pred
    # Decode generated summaries, replacing -100 in the labels as it's used for padding.
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE scores
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Extract specific ROUGE scores
    result = {key: value * 100 for key, value in result.items()} # Convert to percentage

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# --- Tokenization Function ---
# Global tokenizer variable to be set within the training loop
tokenizer = None

def preprocess_function(examples):
    """Tokenizes the input and target texts."""
    prefix = "normalize claim: "  # Task-specific prefix for T5 models
    inputs = [prefix + str(doc) if doc is not None else "" for doc in examples['Social Media Post']]
    targets = [str(doc) if doc is not None else "" for doc in examples['Normalized Claim']]

    # Tokenize source texts
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, truncation=True, padding="max_length")

    # Tokenize target texts (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Training Function ---
def train_model(model_name, train_data, val_data, output_dir_suffix):
    global tokenizer # Allow modification of the global tokenizer variable

    print(f"\n--- Training {model_name} ---")
    output_dir = os.path.join(MODEL_OUTPUT_DIR, f"{model_name.replace('/', '_')}_{output_dir_suffix}")
    logging_dir = os.path.join(output_dir, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # 1. Load Tokenizer and Model
    print(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 2. Tokenize Data
    print("Tokenizing datasets...")
    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_val = val_data.map(preprocess_function, batched=True)

    # Remove original text columns to free up memory and avoid Trainer issues
    tokenized_train = tokenized_train.remove_columns(train_data.column_names)
    tokenized_val = tokenized_val.remove_columns(val_data.column_names)


    # 3. Data Collator
    # Dynamically pads sequences to the longest sequence in a batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest" # Pad to longest in batch, more efficient than padding to max_length
    )

    # 4. Training Arguments
    print("Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",       # Evaluate at the end of each epoch
        save_strategy="epoch",             # Save a checkpoint at the end of each epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        predict_with_generate=True,        # Important for Seq2Seq evaluation
        fp16=FP16,                         # Use mixed precision if available
        logging_dir=logging_dir,           # Directory for TensorBoard logs
        logging_steps=100,                 # Log training loss every 100 steps
        load_best_model_at_end=True,       # Load the best model based on metric at the end
        metric_for_best_model="rougeL",    # Use ROUGE-L score to find the best model
        greater_is_better=True,            # Higher ROUGE is better
        report_to="tensorboard",           # Log to TensorBoard (can add "wandb" if using Weights & Biases)
        # Optional: Add gradient checkpointing if still running out of memory (slows training)
        # gradient_checkpointing=True,
        seed=RANDOM_SEED,
    )

    # 5. Trainer Initialization
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. Start Training
    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    # 7. Save Training Stats and Final Model
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print(f"Saving best model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir) # Saves the best model due to load_best_model_at_end=True
    tokenizer.save_pretrained(output_dir)

    print(f"--- Finished training {model_name} ---")
    return trainer # Return trainer if needed for further analysis/evaluation


# --- Run Training ---
if __name__ == "__main__":
    # Train BART model
    bart_trainer = train_model(
        model_name=BART_MODEL_NAME,
        train_data=train_dataset,
        val_data=val_dataset,
        output_dir_suffix="finetuned_claim_normalization"
    )
    
    # Clear CUDA cache before training the next model (important in limited memory environments)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

    # Train T5 model
    t5_trainer = train_model(
        model_name=T5_MODEL_NAME,
        train_data=train_dataset,
        val_data=val_dataset,
        output_dir_suffix="finetuned_claim_normalization"
    )

    print("\n--- All models trained successfully! ---")
    print(f"BART model saved in: {os.path.join(MODEL_OUTPUT_DIR, f'{BART_MODEL_NAME}_finetuned_claim_normalization')}")
    print(f"T5 model saved in: {os.path.join(MODEL_OUTPUT_DIR, f'{T5_MODEL_NAME}_finetuned_claim_normalization')}")