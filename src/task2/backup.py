import pandas as pd
import os
import logging
from evaluate import load as load_metric
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
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
import csv  # Import CSV module for saving metrics

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
PER_DEVICE_TRAIN_BATCH_SIZE = 16 # Try 2 or 1 if 4 is too high
PER_DEVICE_EVAL_BATCH_SIZE = 16  # Can usually be higher than train batch size
# Increase accumulation steps to simulate larger batch size if lowering batch size
GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch size = batch_size * accumulation_steps
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3 # Start with a small number, increase if needed/possible
WEIGHT_DECAY = 0.01
FP16 = torch.cuda.is_available() # Use mixed precision if CUDA is available

import re
from contractions import contractions_dict

'''
Preprocessing: We have three columns in the dataset: (['PID', 'Social Media Post', 'Normalized Claim'], dtype='object')
- PID: Integer number representing the post ID
- Social Media Post: The text of the post
- Normalized Claim: The claim in the post, which is a string of text
'''

def expand_contractions(text, contractions_dict):
    """
    Expands contractions in the given text using the provided contractions dictionary.
    """
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group()], text)

def clean_text(text):
    """
    Cleans the text by removing links, special characters, and extra whitespace, and converts to lowercase.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers (except spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(data):
    """
    Preprocesses the dataset by expanding contractions and cleaning text.
    """
    # Apply preprocessing to the 'Social Media Post' column
    data['Social Media Post'] = data['Social Media Post'].apply(lambda x: expand_contractions(x, contractions_dict))
    data['Social Media Post'] = data['Social Media Post'].apply(clean_text)
    data['Normalized Claim'] = data['Normalized Claim'].apply(lambda x: expand_contractions(x, contractions_dict))
    data['Normalized Claim'] = data['Normalized Claim'].apply(clean_text)
    return data

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
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv')) # Load if needed for final eval

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

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

def compute_metrics(eval_pred, tokenizer):
    """Computes ROUGE, BLEU-4, and BERTScore."""
    predictions, labels = eval_pred

    # Ensure token IDs are valid
    predictions = np.where((predictions >= 0) & (predictions < tokenizer.vocab_size), predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print(f"Decoded Predictions: {decoded_preds[:5]}")
    print(f"Decoded Labels: {decoded_labels[:5]}")

    # Filter out empty predictions and references
    filtered_preds, filtered_labels = [], []
    for pred, label in zip(decoded_preds, decoded_labels):
        if pred.strip() and label.strip():  # Only include non-empty predictions and references
            filtered_preds.append(pred)
            filtered_labels.append(label)

    # If no valid predictions or references, return 0 for all metrics
    if not filtered_preds or not filtered_labels:
        print("Warning: No valid predictions or references for metric computation.")
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0, "bleu4": 0, "bertscore_f1": 0}

    # Ensure references are in the correct format for BLEU and ROUGE
    decoded_labels_bleu = [[label] for label in filtered_labels]  # BLEU expects a list of lists
    decoded_labels_rouge = filtered_labels  # ROUGE expects a flat list of strings

    # ROUGE
    rouge_scores = rouge.compute(predictions=filtered_preds, references=decoded_labels_rouge, use_stemmer=True)
    rouge_results = {
        "rouge1": rouge_scores["rouge1"] * 100,
        "rouge2": rouge_scores["rouge2"] * 100,
        "rougeL": rouge_scores["rougeL"] * 100,
    }

    # BLEU-4
    bleu_score = bleu.compute(predictions=filtered_preds, references=decoded_labels_bleu)
    bleu_results = {"bleu4": bleu_score["bleu"] * 100}

    # BERTScore
    bertscore_results = bertscore.compute(predictions=filtered_preds, references=filtered_labels, lang="en")
    bertscore_f1 = {"bertscore_f1": np.mean(bertscore_results["f1"]) * 100}

    # Combine all metrics
    return {**rouge_results, **bleu_results, **bertscore_f1}

# --- Tokenization Function ---
# Global tokenizer variable to be set within the training loop
tokenizer = None

def preprocess_function(examples, tokenizer):
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
    tokenized_train = train_data.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    tokenized_val = val_data.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

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
    print("Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",       # Evaluate at the end of each epoch
        save_strategy="epoch",             # Save checkpoints at the end of each epoch
        save_total_limit=1,                # Keep only the best checkpoint
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,  # Train for specified epochs
        predict_with_generate=True,        # Important for Seq2Seq evaluation
        generation_max_length=120,         # Set max length of generated response to 120
        fp16=FP16,                         # Use mixed precision if available
        logging_dir=logging_dir,           # Directory for TensorBoard logs
        logging_steps=100,                 # Log training loss every 100 steps
        load_best_model_at_end=True,       # Load the best model based on metric at the end
        metric_for_best_model="rougeL",    # Use ROUGE-L score to find the best model
        greater_is_better=True,            # Higher ROUGE is better
        report_to="none",                  # Do not report to TensorBoard
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
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    # 6. Text File for Metrics
    metrics_file = os.path.join(output_dir, "training_metrics.txt")
    with open(metrics_file, mode="w") as file:
        file.write("Epoch-wise Metrics:\n")

    # 7. Start Training
    print("Starting training...")

    # Train the model
    train_result = trainer.train()
    train_metrics = train_result.metrics

    # Log training metrics to the shell
    print(f"Training Metrics: {train_metrics}")

    # Evaluate the model after training
    eval_metrics = trainer.evaluate()
    print(f"Evaluation Metrics: {eval_metrics}")

    # Save evaluation metrics to the text file
    with open(metrics_file, mode="a") as file:
        file.write("Final Metrics:\n")
        file.write(f"  ROUGE-1: {eval_metrics.get('eval_rouge1', 0):.2f}\n")
        file.write(f"  ROUGE-2: {eval_metrics.get('eval_rouge2', 0):.2f}\n")
        file.write(f"  ROUGE-L: {eval_metrics.get('eval_rougeL', 0):.2f}\n")
        file.write(f"  BLEU-4: {eval_metrics.get('eval_bleu4', 0):.2f}\n")
        file.write(f"  BERTScore F1: {eval_metrics.get('eval_bertscore_f1', 0):.2f}\n")
        file.write("\n")

    print("Training finished.")

    # 8. Save Best and Last Models
    print(f"Saving best model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)  # Save the best model due to load_best_model_at_end=True
    tokenizer.save_pretrained(output_dir)

    last_model_dir = os.path.join(output_dir, "last_model")
    os.makedirs(last_model_dir, exist_ok=True)
    print(f"Saving last model and tokenizer to {last_model_dir}...")
    model.save_pretrained(last_model_dir)
    tokenizer.save_pretrained(last_model_dir)

    print(f"--- Finished training {model_name} ---")
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


# --- Evaluation Function ---
def evaluate_model_on_test(model_name, test_data, output_dir_suffix):
    """
    Evaluates the stored model on the test dataset.
    """
    print(f"\n--- Evaluating {model_name} on Test Dataset ---")
    model_dir = os.path.join(MODEL_OUTPUT_DIR, f"{model_name.replace('/', '_')}_{output_dir_suffix}")
    
    # Load the model and tokenizer
    print(f"Loading model and tokenizer from {model_dir}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Tokenize the test dataset
    print("Tokenizing test dataset...")
    tokenized_test = test_data.map(
        lambda examples: preprocess_function(examples, tokenizer),  # Pass tokenizer explicitly
        batched=True,
        remove_columns=test_data.column_names
    )
    
    # Initialize the trainer for evaluation
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),  # Pass tokenizer explicitly
        args=Seq2SeqTrainingArguments(
            output_dir=model_dir,
            predict_with_generate=True,  # Enable generation during evaluation
            generation_max_length=MAX_TARGET_LENGTH,  # Set max length for generated responses
            generation_num_beams=4  # Use beam search for better results
        )
    )
    
    # Evaluate the model on the test dataset
    print("Evaluating...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test)
    print(f"Test Metrics: {test_metrics}")
    
    # Save test metrics to a file
    test_metrics_file = os.path.join(model_dir, "test_metrics.txt")
    with open(test_metrics_file, mode="w") as file:
        file.write("Test Metrics:\n")
        file.write(f"  ROUGE-1: {test_metrics.get('eval_rouge1', 0):.2f}\n")
        file.write(f"  ROUGE-2: {test_metrics.get('eval_rouge2', 0):.2f}\n")
        file.write(f"  ROUGE-L: {test_metrics.get('eval_rougeL', 0):.2f}\n")
        file.write(f"  BLEU-4: {test_metrics.get('eval_bleu4', 0):.2f}\n")
        file.write(f"  BERTScore F1: {test_metrics.get('eval_bertscore_f1', 0):.2f}\n")
        file.write("\n")
    
    print(f"Test metrics saved to {test_metrics_file}")
    return test_metrics


# --- Run Training ---
if __name__ == "__main__":
    # print("Starting training...")
    # # Train BART model
    # bart_trainer = train_model(
    #     model_name=BART_MODEL_NAME,
    #     train_data=train_dataset,
    #     val_data=val_dataset,
    #     output_dir_suffix="finetuned_claim_normalization"
    # )
    # print("Finished training BART model.")

    # # Clear CUDA cache before training the next model (important in limited memory environments)
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     print("CUDA cache cleared.")

    # Train T5 model
    # t5_trainer = train_model(
    #     model_name=T5_MODEL_NAME,
    #     train_data=train_dataset,
    #     val_data=val_dataset,
    #     output_dir_suffix="finetuned_claim_normalization"
    # )
    # print("Finished training T5 model.")

    # print("\n--- All models trained successfully! ---")
    # print(f"BART model saved in: {os.path.join(MODEL_OUTPUT_DIR, f'{BART_MODEL_NAME}_finetuned_claim_normalization')}")
    # print(f"T5 model saved in: {os.path.join(MODEL_OUTPUT_DIR, f'{T5_MODEL_NAME}_finetuned_claim_normalization')}")

    # Evaluate BART model on test dataset
    bart_test_metrics = evaluate_model_on_test(
        model_name=BART_MODEL_NAME,
        test_data=test_dataset,
        output_dir_suffix="finetuned_claim_normalization"
    )
    print("Finished evaluating BART model on test dataset.")

    # Evaluate T5 model on test dataset
    # t5_test_metrics = evaluate_model_on_test(
    #     model_name=T5_MODEL_NAME,
    #     test_data=test_dataset,
    #     output_dir_suffix="finetuned_claim_normalization"
    # )
    # print("Finished evaluating T5 model on test dataset.")