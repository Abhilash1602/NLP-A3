# -*- coding: utf-8 -*-
"""
MuSE Model Testing Script
Loads the best trained model and evaluates it on the test set.
All outputs are stored in the src/task3/outputs directory.
"""

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from evaluate import load as load_metric
import nltk
import numpy as np
import json
from typing import Dict, List, Optional, Tuple

# Import needed classes from task3.py
from task3 import SarcasmDataset, MuSEModel, compute_metrics, MuSEFusionModule

# --- Configuration ---
# Model Identifiers
VIT_MODEL_ID = "google/vit-base-patch16-224-in21k"
BART_MODEL_ID = "facebook/bart-base"

# Test Data Paths
TEST_TSV_PATH = r"dataset\task3\test.tsv"
TEST_DESC_PKL = r"dataset\task3\D_test.pkl"
TEST_OBJ_PKL = r"dataset\task3\O_test.pkl"
IMAGE_DIR = r"dataset\task3\images"

# Model and Output Paths
MODEL_DIR = "./muse_results"  # Where the trained model is saved
OUTPUT_DIR = r"src\task3\outputs"  # Where to save test outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test Parameters
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
EVAL_BATCH_SIZE = 16
FUSION_DIM = 768

def test_model():
    print("Starting MuSE Model Testing...")
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Download NLTK data if needed
    for resource in ['punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)
    
    # Initialize Tokenizer and Feature Extractor
    print("Initializing Tokenizer and Feature Extractor...")
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_ID)
    feature_extractor = AutoFeatureExtractor.from_pretrained(VIT_MODEL_ID)
    
    # Add [SEP] token if not present
    if "[SEP]" not in tokenizer.additional_special_tokens:
        print("Adding [SEP] token to tokenizer.")
        tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]']})
    
    # --- Load Test Data ---
    print("Loading test dataset...")
    test_dataset = SarcasmDataset(
        tsv_path=TEST_TSV_PATH, desc_pkl_path=TEST_DESC_PKL, obj_pkl_path=TEST_OBJ_PKL,
        image_dir=IMAGE_DIR, tokenizer=tokenizer, feature_extractor=feature_extractor,
        max_source_length=MAX_SOURCE_LENGTH, max_target_length=MAX_TARGET_LENGTH, dataset_type="test"
    )
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None, label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )
    
    # --- Load Best Model ---
    print(f"Loading best model from {MODEL_DIR}...")
    try:
        # Initialize model with the same configuration as training
        model = MuSEModel(
            vit_model_id=VIT_MODEL_ID, bart_model_id=BART_MODEL_ID, fusion_dim=FUSION_DIM
        )
        # Load the trained weights
        model_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Resize token embeddings if needed
        model.bart.resize_token_embeddings(len(tokenizer))
        
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # --- Setup Trainer for Evaluation ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # --- Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    if len(test_dataset) > 0:
        print(f"Evaluating on {len(test_dataset)} test samples...")
        test_results = trainer.evaluate(
            eval_dataset=test_dataset,
            metric_key_prefix="test"
        )
        
        print("\nTest Set Evaluation Metrics:")
        # Format metrics for better readability
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")
        
        # Save test metrics to file
        metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        print(f"Test metrics saved to {metrics_path}")
        
        # --- Generate Predictions on Test Set ---
        print("\n--- Generating Predictions on Test Set ---")
        sample_predictions = []
        sample_count = min(5, len(test_dataset))  # Save examples for 5 samples or all if less
        
        for i in range(sample_count):
            sample = test_dataset[i]
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
            labels = sample['labels'].unsqueeze(0).to(device)
            
            # Generate explanation
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode input, ground truth, and prediction
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            labels_decoded = labels.clone()
            labels_decoded[labels_decoded == -100] = tokenizer.pad_token_id
            ground_truth = tokenizer.decode(labels_decoded[0], skip_special_tokens=True)
            prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            sample_predictions.append({
                "sample_id": i,
                "input_text": input_text,
                "ground_truth": ground_truth,
                "prediction": prediction
            })
        
        # Save sample predictions
        predictions_path = os.path.join(OUTPUT_DIR, "sample_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(sample_predictions, f, indent=4)
        print(f"Sample predictions saved to {predictions_path}")
        
    else:
        print("Test dataset is empty. Cannot perform evaluation.")

if __name__ == "__main__":
    test_model()
