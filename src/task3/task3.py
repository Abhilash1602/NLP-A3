# -*- coding: utf-8 -*-
"""
Multimodal Sarcasm Explanation (MuSE) Model Implementation
Based on ViT, BART, and a custom fusion mechanism.

Updates:
- Uses validation set during training.
- Performs final evaluation on test set after training.
- Logs epoch-wise metrics to CSV.
- Generates metric plots.
"""

import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torchvision.io import read_image # Using PIL instead for wider compatibility
from PIL import Image
import matplotlib.pyplot as plt # Added for plotting
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModel, # For ViT
    AutoModelForSeq2SeqLM, # For BART
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback # Added for potential custom logging if needed
)
from evaluate import load as load_metric # Uses HF evaluate library
import nltk
import numpy as np
from typing import Dict, List, Optional, Tuple
import math # For checking float comparison

# --- Configuration ---
# Model Identifiers (Hugging Face Hub)
VIT_MODEL_ID = "google/vit-base-patch16-224-in21k"
BART_MODEL_ID = "facebook/bart-base"

# File Paths (Adjust these paths according to your setup)
TRAIN_TSV_PATH = r"dataset\task3\train_df.tsv"
VAL_TSV_PATH = r"dataset\task3\val_df.tsv"   
TRAIN_DESC_PKL = r"dataset\task3\D_train.pkl"
VAL_DESC_PKL = r"dataset\task3\D_val.pkl"     
TRAIN_OBJ_PKL = r"dataset\task3\O_train.pkl"
VAL_OBJ_PKL = r"dataset\task3\O_val.pkl"       
IMAGE_DIR = r"dataset\task3\images" 

# Output Directories and Files
OUTPUT_DIR = "./muse_results"
LOGGING_DIR = "./muse_logs"
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, "epoch_metrics.csv") # <-- New: Path for metrics CSV
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots") # <-- New: Directory for plots

# Training Hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 8 # Adjust based on GPU memory
LEARNING_RATE = 5e-5
MAX_SOURCE_LENGTH = 512 # Max length for concatenated input text
MAX_TARGET_LENGTH = 128 # Max length for generated explanation
EVAL_BATCH_SIZE = 16

# Fusion Parameters
FUSION_DIM = 768 # BART-base and ViT-base hidden dim

# --- Data Loading and Preprocessing ---

class SarcasmDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MORE+ data for MuSE.
    Handles loading images, text from TSV, descriptions, objects, and tokenization.
    """
    def __init__(self, tsv_path: str, desc_pkl_path: str, obj_pkl_path: str,
                 image_dir: str, tokenizer, feature_extractor,
                 max_source_length: int, max_target_length: int, dataset_type: str = "train"):
        """
        Args:
            tsv_path (str): Path to the TSV file (e.g., train_df.tsv).
                           Expected columns: 'image_id', 'caption', 'sarcasm_target', 'sarcasm_explanation'
            desc_pkl_path (str): Path to the pickle file containing image descriptions.
                                 Expected format: {image_id: description_string}
            obj_pkl_path (str): Path to the pickle file containing detected objects.
                                Expected format: {image_id: objects_string}
            image_dir (str): Path to the directory containing images.
            tokenizer: Hugging Face tokenizer (for BART).
            feature_extractor: Hugging Face feature extractor (for ViT).
            max_source_length (int): Maximum sequence length for the encoder input.
            max_target_length (int): Maximum sequence length for the decoder output.
            dataset_type (str): Label for the dataset type (e.g., "train", "validation", "test") for logging.
        """
        try:
            self.data = pd.read_csv(tsv_path, sep='\t')
            print(f"Loaded {len(self.data)} samples from {tsv_path} ({dataset_type} set)")
             # Print actual columns to see what we're working with
            print(f"Available columns in {tsv_path}: {list(self.data.columns)}")
        except FileNotFoundError:
            print(f"ERROR: TSV file not found at {tsv_path}. Please check the path.")
            raise
        
        # Add column mapping
        column_mapping = {
            'pid': 'image_id',
            'text': 'caption',
            'target_of_sarcasm': 'sarcasm_target',
            'explanation': 'sarcasm_explanation'
        }

        # Rename columns based on mapping
        for old_name, new_name in column_mapping.items():
            if old_name in self.data.columns:
                self.data[new_name] = self.data[old_name]

        # Check if required columns exist after mapping
        required_cols = ['image_id', 'caption', 'sarcasm_target', 'sarcasm_explanation']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"TSV file '{tsv_path}' must contain columns: {required_cols}")
        
        try:
            with open(desc_pkl_path, 'rb') as f:
                self.descriptions = pickle.load(f)
            print(f"Loaded descriptions from {desc_pkl_path}")
        except FileNotFoundError:
            print(f"Warning: Description file not found at {desc_pkl_path}. Descriptions will be empty.")
            self.descriptions = {}

        try:
            with open(obj_pkl_path, 'rb') as f:
                self.objects = pickle.load(f)
            print(f"Loaded objects from {obj_pkl_path}")
        except FileNotFoundError:
            print(f"Warning: Objects file not found at {obj_pkl_path}. Objects will be empty.")
            self.objects = {}

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.dataset_type = dataset_type

        # Ensure required columns exist
        required_cols = ['image_id', 'caption', 'sarcasm_target', 'sarcasm_explanation']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"TSV file '{tsv_path}' must contain columns: {required_cols}")

        # Check if image directory exists
        if not os.path.isdir(self.image_dir):
             print(f"Warning: Image directory not found at {self.image_dir}. Image loading will fail.")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_id = sample['image_id']
        caption = str(sample['caption'])
        sarcasm_target = str(sample['sarcasm_target'])
        explanation = str(sample['sarcasm_explanation'])

        # --- Load Image ---
        image_filename = f"{image_id}.jpg" if not str(image_id).endswith(('.jpg', '.png', '.jpeg')) else str(image_id)
        image_path = os.path.join(self.image_dir, image_filename)
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        except FileNotFoundError:
            # print(f"Warning: Image file not found at {image_path}. Using zero tensor.") # Reduced verbosity
            try:
                dummy_size = (3, self.feature_extractor.size['height'], self.feature_extractor.size['width'])
            except:
                dummy_size = (3, 224, 224)
            pixel_values = torch.zeros(dummy_size)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Using zero tensor.")
            try:
                dummy_size = (3, self.feature_extractor.size['height'], self.feature_extractor.size['width'])
            except:
                dummy_size = (3, 224, 224)
            pixel_values = torch.zeros(dummy_size)

        # --- Load Precomputed Textual Data ---
        description = self.descriptions.get(image_id, "")
        objects = self.objects.get(image_id, "")

        # --- Construct Input Text ---
        input_text = (
            f"Caption: {caption} [SEP] "
            f"Description: {description} [SEP] "
            f"Objects: {objects} [SEP] "
            f"Target: {sarcasm_target}"
        )

        # --- Tokenize Text ---
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                explanation,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        labels["input_ids"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100

        item = {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "labels": labels["input_ids"].squeeze(0)
        }
        # Add image_id for potential debugging or reference if needed later
        # item["image_id"] = image_id
        return item

# --- MuSE Model Definition (Identical to previous version) ---

class MuSEFusionModule(nn.Module):
    """Implements the Shared Fusion Mechanism for MuSE."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.text_self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.img_self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.cross_attn_vt = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.cross_attn_tv = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.gate_t = nn.Linear(embed_dim, embed_dim)
        self.gate_v = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.norm_t = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_fuse = nn.LayerNorm(embed_dim)

    def forward(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor,
                text_attention_mask: Optional[torch.Tensor] = None,
                image_attention_mask: Optional[torch.Tensor] = None):
        text_key_padding_mask = (text_attention_mask == 0) if text_attention_mask is not None else None
        img_key_padding_mask = (image_attention_mask == 0) if image_attention_mask is not None else None

        attn_output_t, _ = self.text_self_attn(text_embeddings, text_embeddings, text_embeddings,
                                              key_padding_mask=text_key_padding_mask)
        At = self.norm_t(text_embeddings + attn_output_t)

        attn_output_v, _ = self.img_self_attn(image_embeddings, image_embeddings, image_embeddings)
        Av = self.norm_v(image_embeddings + attn_output_v)

        cross_attn_output_tv, _ = self.cross_attn_tv(query=At, key=Av, value=Av,
                                                     key_padding_mask=img_key_padding_mask)
        F_tv = cross_attn_output_tv

        cross_attn_output_vt, _ = self.cross_attn_vt(query=Av, key=At, value=At,
                                                     key_padding_mask=text_key_padding_mask)
        F_vt = cross_attn_output_vt

        gate_weights_t = self.sigmoid(self.gate_t(text_embeddings))
        gate_weights_v = self.sigmoid(self.gate_v(image_embeddings))

        pooled_F_vt = F_vt.mean(dim=1)
        pooled_gate_v = gate_weights_v.mean(dim=1)
        expanded_pooled_F_vt = pooled_F_vt.unsqueeze(1).expand_as(F_tv)
        expanded_pooled_gate_v = pooled_gate_v.unsqueeze(1).expand_as(gate_weights_t)

        fused_cross_modal = (gate_weights_t * F_tv) + (expanded_pooled_gate_v * expanded_pooled_F_vt)
        fused_representation = self.norm_fuse(text_embeddings + fused_cross_modal)
        return fused_representation

class MuSEModel(nn.Module):
    """
    Multimodal Sarcasm Explanation (MuSE) Model.
    Combines ViT, BART, and the MuSEFusionModule.
    """
    def __init__(self, vit_model_id: str, bart_model_id: str, fusion_dim: int):
        super().__init__()
        print("Initializing MuSEModel...")
        self.vit = AutoModel.from_pretrained(vit_model_id)
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(bart_model_id)
        # Add generation_config attribute from BART model
        self.generation_config = self.bart.generation_config
        self.bart_embed_dim = self.bart.config.hidden_size
        self.vit_embed_dim = self.vit.config.hidden_size

        if self.vit_embed_dim != fusion_dim:
            print(f"Projecting ViT output from {self.vit_embed_dim} to {fusion_dim}")
            self.vit_proj = nn.Linear(self.vit_embed_dim, fusion_dim)
        else:
            self.vit_proj = nn.Identity()

        if self.bart_embed_dim != fusion_dim:
             raise ValueError(f"BART embedding dimension ({self.bart_embed_dim}) must match fusion dimension ({fusion_dim}).")

        print("Initializing Fusion Module...")
        self.fusion_module = MuSEFusionModule(embed_dim=fusion_dim)
        print("MuSEModel initialization complete.")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Dict:
        # Freeze ViT? Consider adding self.vit.eval() or using no_grad based on fine-tuning strategy
        # with torch.no_grad(): # Example: uncomment if ViT should not be trained
        vit_outputs = self.vit(pixel_values=pixel_values)
        image_embeddings = vit_outputs.last_hidden_state
        image_embeddings_proj = self.vit_proj(image_embeddings)

        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False
        )
        text_embeddings = encoder_outputs.last_hidden_state

        fused_encoder_representation = self.fusion_module(
            text_embeddings=text_embeddings, image_embeddings=image_embeddings_proj,
            text_attention_mask=attention_mask
        )

        modified_encoder_outputs = (fused_encoder_representation,) + encoder_outputs[1:]

        outputs = self.bart(
            attention_mask=attention_mask, decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=modified_encoder_outputs, labels=labels, return_dict=True,
        )
        return outputs

    # Generate method remains the same as it uses the trained weights
    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        pixel_values: torch.Tensor, **generation_kwargs
    ) -> torch.Tensor:
        self.eval()
        vit_outputs = self.vit(pixel_values=pixel_values)
        # Handle both object and dictionary return types
        image_embeddings = vit_outputs.last_hidden_state if hasattr(vit_outputs, 'last_hidden_state') else vit_outputs['last_hidden_state']
        image_embeddings_proj = self.vit_proj(image_embeddings)

        encoder_outputs = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Handle both object and dictionary return types
        text_embeddings = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs['last_hidden_state']

        fused_encoder_representation = self.fusion_module(
            text_embeddings=text_embeddings, image_embeddings=image_embeddings_proj,
            text_attention_mask=attention_mask
        )
        
        # Create a proper encoder output object with the correct structure
        from transformers.modeling_outputs import BaseModelOutput
        modified_encoder_outputs = BaseModelOutput(
            last_hidden_state=fused_encoder_representation,
            hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, 'hidden_states') else encoder_outputs.get('hidden_states'),
            attentions=encoder_outputs.attentions if hasattr(encoder_outputs, 'attentions') else encoder_outputs.get('attentions')
        )
        
        # Pass the encoder outputs directly, not wrapped in another dictionary
        generated_ids = self.bart.generate(
            input_ids=None,  # Not needed when encoder_outputs is provided
            attention_mask=attention_mask,
            encoder_outputs=modified_encoder_outputs,
            **generation_kwargs
        )
        return generated_ids

    # Add these helper methods to support generation
    def get_encoder(self):
        """Returns the encoder part of the model."""
        return self.bart.get_encoder()

    def get_decoder(self):
        """Returns the decoder part of the model."""
        return self.bart.get_decoder()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation."""
        return self.bart.prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorders the cache for beam search."""
        return self.bart._reorder_cache(past_key_values, beam_idx)

# --- Evaluation Metrics Computation (Identical to previous version) ---

rouge_metric = load_metric("rouge")
bleu_metric = load_metric("bleu")
meteor_metric = load_metric("meteor")
bertscore_metric = load_metric("bertscore")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    decoded_preds_rouge = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels_rouge = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    rouge_result = rouge_metric.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)
    rouge_scores = {
        "rouge1": round(rouge_result["rouge1"] * 100, 2), "rouge2": round(rouge_result["rouge2"] * 100, 2),
        "rougeL": round(rouge_result["rougeL"] * 100, 2), "rougeLsum": round(rouge_result["rougeLsum"] * 100, 2),
    }

    decoded_labels_bleu = [[label] for label in decoded_labels]
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    bleu_scores = {
        "bleu1": round(bleu_result['precisions'][0] * 100, 2), "bleu2": round(bleu_result['precisions'][1] * 100, 2),
        "bleu3": round(bleu_result['precisions'][2] * 100, 2), "bleu4": round(bleu_result['precisions'][3] * 100, 2),
        "bleu": round(bleu_result['bleu'] * 100, 2)
    }

    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_score = {"meteor": round(meteor_result["meteor"] * 100, 2)}

    try:
        # Ensure model_type matches available models for bert_score
        bertscore_result = bertscore_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")#, model_type="microsoft/deberta-xlarge-mnli")
        bert_f1_avg = round(np.mean(bertscore_result['f1']) * 100, 2)
        bertscore_scores = {"bertscore_f1": bert_f1_avg}
    except Exception as e:
        print(f"Could not compute BERTScore: {e}")
        bertscore_scores = {"bertscore_f1": 0.0}

    metrics = {**rouge_scores, **bleu_scores, **meteor_score, **bertscore_scores}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    metrics["gen_len"] = np.mean(prediction_lens)
    return metrics


# --- Helper function for Processing Logs and Plotting ---

def process_logs_and_plot(log_history, csv_path, plots_dir):
    """
    Processes trainer log history to extract epoch-wise metrics, saves to CSV,
    and generates plots.
    """
    print("\nProcessing log history for metrics and plotting...")
    epochs = []
    train_losses = []
    eval_metrics_list = [] # List of dicts, one per eval epoch

    # Store metrics per epoch
    epoch_metrics = {} # Key: epoch number, Value: dict of metrics

    # Extract data from log history
    for log_entry in log_history:
        epoch = log_entry.get('epoch')
        if epoch is None: continue # Skip logs without epoch info

        # Round epoch to nearest int for grouping (Trainer sometimes logs fractional epochs)
        epoch_int = round(epoch)

        if 'loss' in log_entry: # Training loss log
            if epoch_int not in epoch_metrics: epoch_metrics[epoch_int] = {'train_loss': [], 'eval_metrics': {}}
            epoch_metrics[epoch_int]['train_loss'].append(log_entry['loss'])

        elif 'eval_loss' in log_entry: # Evaluation log
            if epoch_int not in epoch_metrics: epoch_metrics[epoch_int] = {'train_loss': [], 'eval_metrics': {}}
            # Store the entire eval metrics dict, excluding non-metric keys
            metrics_to_store = {k: v for k, v in log_entry.items() if k.startswith('eval_')}
            metrics_to_store['epoch'] = epoch_int # Ensure epoch is stored
            epoch_metrics[epoch_int]['eval_metrics'] = metrics_to_store


    # Consolidate metrics into lists for DataFrame and plotting
    sorted_epochs = sorted(epoch_metrics.keys())
    final_metrics_list = []

    for epoch in sorted_epochs:
        data = epoch_metrics[epoch]
        avg_train_loss = np.mean(data['train_loss']) if data['train_loss'] else None
        eval_data = data['eval_metrics']

        row = {'Epoch': epoch}
        if avg_train_loss is not None:
             row['Training Loss'] = avg_train_loss
        # Add eval metrics, renaming keys slightly for clarity (e.g., eval_rougeL -> Validation ROUGE-L)
        if eval_data:
            for key, value in eval_data.items():
                 if key.startswith('eval_'):
                     metric_name = "Validation " + key.split('eval_')[1].replace('_', ' ').title()
                     row[metric_name] = value
        final_metrics_list.append(row)


    if not final_metrics_list:
        print("Warning: No epoch-level metrics found in log history.")
        return

    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(final_metrics_list)
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        metrics_df.to_csv(csv_path, index=False)
        print(f"Epoch metrics saved to {csv_path}")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")


    # Generate Plots
    try:
        os.makedirs(plots_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-grid') # Use a nice style

        # Plot Training Loss
        if 'Training Loss' in metrics_df.columns and not metrics_df['Training Loss'].isnull().all():
            plt.figure(figsize=(10, 6))
            plt.plot(metrics_df['Epoch'], metrics_df['Training Loss'], marker='o', linestyle='-', label='Training Loss')
            plt.title('Training Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xticks(metrics_df['Epoch'].unique()) # Ensure integer ticks for epochs
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "training_loss_plot.png"))
            plt.close()
            print(f"Saved training loss plot to {plots_dir}")
        else:
             print("Skipping training loss plot (no data).")

        # Plot Validation Metrics (Example: ROUGE-L and METEOR)
        metrics_to_plot = ['Validation Rougel', 'Validation Meteor', 'Validation Bleu', 'Validation Bertscore F1'] # Adjust as needed
        plotable_metrics_found = [m for m in metrics_to_plot if m in metrics_df.columns and not metrics_df[m].isnull().all()]

        if plotable_metrics_found:
            plt.figure(figsize=(12, 7))
            for metric_name in plotable_metrics_found:
                 plt.plot(metrics_df['Epoch'], metrics_df[metric_name], marker='o', linestyle='-', label=metric_name)

            plt.title('Validation Metrics per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.xticks(metrics_df['Epoch'].unique())
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "validation_metrics_plot.png"))
            plt.close()
            print(f"Saved validation metrics plot to {plots_dir}")
        else:
             print("Skipping validation metrics plot (no data).")

    except Exception as e:
        print(f"Error generating plots: {e}")

# --- Main Execution Block ---

if __name__ == "__main__":
    print("Starting MuSE Training and Evaluation Pipeline...")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

    # Download NLTK data
    for resource in ['punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
        except LookupError:  # Changed from nltk.downloader.DownloadError
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

    # Initialize Tokenizer and Feature Extractor
    print("Initializing Tokenizer and Feature Extractor...")
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_ID)
    feature_extractor = AutoFeatureExtractor.from_pretrained(VIT_MODEL_ID)

    if "[SEP]" not in tokenizer.additional_special_tokens:
        print("Adding [SEP] token to tokenizer.")
        tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]']})
        # Model embedding resizing will happen after model init


    # --- Load Data ---
    print("Loading datasets...")
    train_dataset = SarcasmDataset(
        tsv_path=TRAIN_TSV_PATH, desc_pkl_path=TRAIN_DESC_PKL, obj_pkl_path=TRAIN_OBJ_PKL,
        image_dir=IMAGE_DIR, tokenizer=tokenizer, feature_extractor=feature_extractor,
        max_source_length=MAX_SOURCE_LENGTH, max_target_length=MAX_TARGET_LENGTH, dataset_type="train"
    )
    # --- Load Validation Data ---
    validation_dataset = SarcasmDataset(
        tsv_path=VAL_TSV_PATH, desc_pkl_path=VAL_DESC_PKL, obj_pkl_path=VAL_OBJ_PKL,
        image_dir=IMAGE_DIR, tokenizer=tokenizer, feature_extractor=feature_extractor,
        max_source_length=MAX_SOURCE_LENGTH, max_target_length=MAX_TARGET_LENGTH, dataset_type="validation"
    )

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None, label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    # --- Initialize Model ---
    model = MuSEModel(
        vit_model_id=VIT_MODEL_ID, bart_model_id=BART_MODEL_ID, fusion_dim=FUSION_DIM
    )
    # Resize BART embeddings if tokenizer was expanded
    model.bart.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # --- Training ---
    print("Setting up Training Arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_strategy="epoch", # Log training loss at each epoch end
        evaluation_strategy="epoch", # Evaluate on validation set each epoch
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True, # Load best model based on validation metric
        metric_for_best_model="eval_rougeL", # Use validation ROUGE-L to find best model
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        fp16=torch.cuda.is_available(),
        # report_to="none", # Disable default reporting like wandb if not used
        push_to_hub=False,
    )

    print("Initializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset, # <-- Use validation set for eval during training
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[MetricsLoggerCallback(METRICS_CSV_PATH)] # Use post-training log processing instead
    )

    print("Starting Training...")
    train_result = trainer.train()
    print("Training finished.")

    # Save final model (best one is already loaded if load_best_model_at_end=True)
    print("Saving the best model found during training...")
    trainer.save_model() # Saves the best model to output_dir
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state() # Saves trainer state, including log history

    # --- Process Logs, Save CSV, Generate Plots ---
    process_logs_and_plot(trainer.state.log_history, METRICS_CSV_PATH, PLOTS_DIR)

    print("\nMuSE Training Pipeline Complete.")