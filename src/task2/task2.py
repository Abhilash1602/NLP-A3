import pandas as pd
import os
import logging
from evaluate import load as load_metric
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import nltk
import numpy as np

# --- Configuration ---
# BASE_DIR = os.getcwd()
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Correctly resolve the data directory
DATA_DIR = os.path.join(BASE_DIR, "dataset/task2")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models') # Directory to save trained models
RANDOM_SEED = 42

# Ensure the model output directory exists
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(MODEL_OUTPUT_DIR, "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger()

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
NUM_TRAIN_EPOCHS = 50 # Start with a small number, increase if needed/possible
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
rouge = load_metric("rouge")
bleu = load_metric("bleu")
bertscore = load_metric("bertscore")

def compute_metrics(eval_pred):
    """Computes ROUGE, BLEU-4, and BERTScore."""
    predictions, labels = eval_pred

    # Ensure token IDs are valid
    predictions = np.where((predictions >= 0) & (predictions < tokenizer.vocab_size), predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_results = {key: value.mid.fmeasure * 100 for key, value in rouge_scores.items()}

    # BLEU-4
    decoded_labels_bleu = [[label] for label in decoded_labels]
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    bleu_results = {"bleu4": bleu_score["bleu"] * 100}

    # BERTScore
    bertscore_results = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    bertscore_f1 = {"bertscore_f1": np.mean(bertscore_results["f1"]) * 100}

    # Combine all metrics
    return {**rouge_results, **bleu_results, **bertscore_f1}

# Redirect stdout to log file
import sys
sys.stdout = open(os.path.join(MODEL_OUTPUT_DIR, "training_output.log"), "w")

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
    global tokenizer  # Allow modification of the global tokenizer variable

    logger.info(f"\n--- Training {model_name} ---")
    output_dir = os.path.join(MODEL_OUTPUT_DIR, f"{model_name.replace('/', '_')}_{output_dir_suffix}")
    logging_dir = os.path.join(output_dir, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # 1. Load Tokenizer and Model
    logger.info(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 2. Tokenize Data
    logger.info("Tokenizing datasets...")
    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_val = val_data.map(preprocess_function, batched=True)

    # Remove original text columns to free up memory and avoid Trainer issues
    tokenized_train = tokenized_train.remove_columns(train_data.column_names)
    tokenized_val = tokenized_val.remove_columns(val_data.column_names)

    # 3. Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest"  # Pad to longest in batch, more efficient than padding to max_length
    )

    # 4. Training Arguments
    logger.info("Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",       # Evaluate at the end of each epoch
        save_strategy="epoch",             # Save a checkpoint at the end of each epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,  # Train for 50 epochs
        predict_with_generate=True,        # Important for Seq2Seq evaluation
        generation_max_length=120,         # Set max length of generated response to 120
        fp16=FP16,                         # Use mixed precision if available
        logging_dir=logging_dir,           # Directory for TensorBoard logs
        logging_steps=100,                 # Log training loss every 100 steps
        load_best_model_at_end=True,       # Load the best model based on metric at the end
        metric_for_best_model="rougeL",    # Use ROUGE-L score to find the best model
        greater_is_better=True,            # Higher ROUGE is better
        report_to="tensorboard",           # Log to TensorBoard
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
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training finished.")

    # 7. Save Training Stats and Final Model
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"Saving best model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)  # Save the best model due to load_best_model_at_end=True
    tokenizer.save_pretrained(output_dir)

    # Save the last model
    last_model_dir = os.path.join(output_dir, "last_model")
    os.makedirs(last_model_dir, exist_ok=True)
    logger.info(f"Saving last model and tokenizer to {last_model_dir}...")
    model.save_pretrained(last_model_dir)
    tokenizer.save_pretrained(last_model_dir)

    logger.info(f"--- Finished training {model_name} ---")
    return trainer  # Return trainer if needed for further analysis/evaluation


# --- Inference Function ---
def generate_response(model, tokenizer, input_text, max_length=120, temperature=0.6):
    """Generates a response using the trained model."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_SOURCE_LENGTH)
    input_ids = inputs["input_ids"].to(model.device)

    # Generate response
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,  # Set temperature to 0.6
        num_beams=4,              # Use beam search for better results
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- Run Training ---
if __name__ == "__main__":
    logger.info("Starting training...")
    # Train BART model
    bart_trainer = train_model(
        model_name=BART_MODEL_NAME,
        train_data=train_dataset,
        val_data=val_dataset,
        output_dir_suffix="finetuned_claim_normalization"
    )
    logger.info("Finished training BART model.")

    # Clear CUDA cache before training the next model (important in limited memory environments)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared.")

    # Train T5 model
    t5_trainer = train_model(
        model_name=T5_MODEL_NAME,
        train_data=train_dataset,
        val_data=val_dataset,
        output_dir_suffix="finetuned_claim_normalization"
    )
    logger.info("Finished training T5 model.")

    logger.info("\n--- All models trained successfully! ---")
    logger.info(f"BART model saved in: {os.path.join(MODEL_OUTPUT_DIR, f'{BART_MODEL_NAME}_finetuned_claim_normalization')}")
    logger.info(f"T5 model saved in: {os.path.join(MODEL_OUTPUT_DIR, f'{T5_MODEL_NAME}_finetuned_claim_normalization')}")