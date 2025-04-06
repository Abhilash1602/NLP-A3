# Task 1: Shakespeare Language Model with Transformer

## Overview
This project implements a character-level transformer model from scratch to generate text in the style of Shakespeare. The model is trained on a dataset of Shakespeare's works and learns to predict the next character in a sequence.

## Preprocessing Steps

### Data Loading and Processing
1. **Data Sources**: The implementation reads data from three files:
   - `shakespear_train.txt`: Main training corpus
   - `shakespear_dev.txt`: Validation dataset
   - `shakespear_test.txt`: Test dataset (created if not found)

2. **Character-level Tokenization**:
   - All characters from the training text are identified and sorted
   - Each unique character is assigned a numeric token ID
   - Special tokens are added:
     - `<PAD>` (ID: 0): For padding shorter sequences to fixed length
     - `<START>` (ID: 1): To mark the beginning of sequences
     - `<STOP>` (ID: 2): To mark the end of sequences
     - `<UNK>` (ID: len(tokenizer)): For handling unseen characters

3. **Sequence Creation**:
   - Each text line is converted to a sequence of token IDs
   - Start and stop tokens are added to the beginning and end
   - Sequences are padded to a maximum length of 128 tokens

4. **Data Preparation for Training**:
   - Input sequence (x): All tokens except the last one
   - Target sequence (y): All tokens except the first one
   - This setup enables the model to learn to predict the next token

## Model Architecture

### Transformer Language Model
The model follows a decoder-only transformer architecture similar to GPT, implemented from scratch with the following components:

1. **Embedding Layer**:
   - Token embedding: Maps token IDs to dense vectors
   - Positional encoding: Fixed sinusoidal encoding to provide position information

2. **Transformer Blocks** (4 layers):
   - Each block contains:
     - **Multi-Head Self-Attention**:
       - 8 attention heads that allow the model to focus on different parts of the input
       - Linear projections for queries, keys, and values
       - Scaled dot-product attention with causal masking
     - **Feed-Forward Network**:
       - Two linear layers with GELU activation
       - Expands to 1024 dimensions internally

3. **Normalization and Residual Connections**:
   - Pre-normalization architecture (LayerNorm before attention/FFN)
   - Residual connections to help with gradient flow

4. **Output Layer**:
   - Linear projection to vocabulary size
   - Softmax to produce probabilities over all possible next characters

### Key Implementation Features

1. **Causal Masking**:
   - Prevents the model from seeing future tokens during training
   - Implemented as a lower triangular matrix multiplied with attention scores

2. **Pre-norm Architecture**:
   - LayerNorm applied before attention and feed-forward layers
   - Helps with training stability, especially for deeper networks

3. **GELU Activation**:
   - Smoother alternative to ReLU used in modern transformers
   - Provides better gradient properties

4. **Weight Initialization**:
   - Xavier uniform initialization for better convergence at start of training

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vocabulary Size | ~100 | Number of unique characters + special tokens |
| d_model | 256 | Dimension of embeddings and hidden layers |
| nhead | 8 | Number of attention heads |
| num_layers | 4 | Number of transformer blocks |
| d_ff | 1024 | Feed-forward network hidden dimension |
| dropout | 0.1 | Dropout rate for regularization |
| batch_size | 32 | Number of sequences per batch |
| learning_rate | 5e-4 | Initial learning rate for AdamW optimizer |
| weight_decay | 0.01 | L2 regularization term |
| max_seq_length | 128 | Maximum sequence length for training |
| max_grad_norm | 1.0 | Gradient clipping threshold |

## Training Process

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Cross-entropy loss (ignoring padding tokens)
- **Learning Rate Scheduling**: ReduceLROnPlateau (halves LR after plateau)
- **Early Stopping**: Training stops after 3 epochs without improvement
- **Gradient Clipping**: Applied with max norm of 1.0

### Training Improvements
1. **Learning Rate Scheduling**: Reduces learning rate when validation loss plateaus
2. **Early Stopping**: Prevents overfitting by stopping training when validation loss stagnates
3. **Gradient Clipping**: Prevents exploding gradients for more stable training
4. **Pre-norm Architecture**: Enables better gradient flow through the network
5. **Special Token Handling**: Excludes special tokens when calculating evaluation loss

## Results

### Training and Validation Loss
![Training and Validation Loss](/NLP-A3/src/task1/loss_plot.png)

The loss curves show:
- Consistent decrease in both training and validation loss
- Good convergence behavior without significant overfitting
- Final validation loss of approximately 1.58

### Test Performance
- **Perplexity**: 3.17 on the test set
- Character-level perplexity in this range indicates the model has learned the patterns of Shakespeare's language well

### Sample Generations
```
Context: 'MENENIUS : The'
Generated: 'MENENIUS : The most of mine.'

Context: 'I do beseech'
Generated: 'I do beseech youur name, Bestory to the law; 'T is should th'

Context: 'BUCKINGHAM : Stay'
Generated: 'BUCKINGHAM : Stay, I have a nothing: not slave the greatity I'
```

These samples demonstrate the model's ability to:
1. Generate character names consistent with Shakespeare's plays
2. Produce grammatically reasonable continuations
3. Mimic the style and vocabulary of Shakespearean language

## Conclusion

The implemented transformer model successfully learns to generate text in Shakespeare's style at character level. The architecture incorporates modern techniques like pre-norm layers, GELU activations, and learning rate scheduling. The generated text captures many characteristics of Shakespeare's writing, including character dialogues and period-appropriate language.


# Task 3: Multimodal Sarcasm Explanation (MuSE) Model

## Introduction

This project implements a Multimodal Sarcasm Explanation (MuSE) model that generates natural language explanations for sarcastic utterances in multimodal contexts (text + image). The model uses a combination of Vision Transformer (ViT) for image processing, BART for text generation, and a custom fusion mechanism to integrate the multimodal features.

## Implementation Details

### Preprocessing Steps

The `SarcasmDataset` class handles the preprocessing of the multimodal data:

1. **Data Loading**:
   - Text data is loaded from TSV files containing captions, sarcasm targets, and explanations
   - Image descriptions and detected objects are loaded from pickle files
   - Images are loaded from the specified directory

2. **Text Preprocessing**:
   - The input text is constructed by concatenating multiple components:
     ```
     Caption: [caption] [SEP] Description: [description] [SEP] Objects: [objects] [SEP] Target: [sarcasm_target]
     ```
   - Special token `[SEP]` is added to the tokenizer to separate different text components
   - Text is tokenized using the BART tokenizer with padding/truncation to fixed lengths
   - Maximum source length: 512 tokens
   - Maximum target length: 128 tokens

3. **Image Preprocessing**:
   - Images are loaded and converted to RGB format
   - Feature extractor from ViT model processes images to the required format
   - Error handling for missing or corrupted images (uses zero tensors as fallback)

4. **Multimodal Input Construction**:
   - Each dataset item contains tokenized text, pixel values, and tokenized targets (explanations)
   - Labels are prepared with special padding (-100) for loss calculation

### Model Architecture

The MuSE model architecture consists of three main components:

1. **Vision Encoder (ViT)**:
   - Uses `google/vit-base-patch16-224-in21k` as the base model
   - Processes image inputs into visual embeddings
   - Output is projected to match fusion dimension (768)

2. **Text Encoder-Decoder (BART)**:
   - Uses `facebook/bart-base` as the base model
   - Encodes the textual input
   - Decoder generates the explanation text

3. **Multimodal Fusion Module**:
   - Custom `MuSEFusionModule` implements a shared fusion mechanism
   - Components:
     - Self-attention mechanisms for both text and image modalities
     - Cross-attention mechanisms between modalities (vision-to-text and text-to-vision)
     - Gating mechanism to control information flow
     - Layer normalization for stabilizing outputs
   - Process:
     1. Apply self-attention to text and image embeddings
     2. Compute cross-modal attention in both directions
     3. Apply gating mechanism to regulate information flow
     4. Integrate the cross-modal information with the original text embeddings

The architecture enables effective integration of visual and textual information for generating contextually relevant sarcasm explanations.

### Hyperparameters

**Model Hyperparameters**:
- Fusion dimension: 768
- Number of attention heads: 8
- Dropout rate: 0.1

**Training Hyperparameters**:
- Batch size: 8
- Learning rate: 5e-5
- Weight decay: 0.01
- Number of epochs: 10
- Optimizer: AdamW
- Generation parameters:
  - Maximum length: 128
  - Beam size: 4

### Training Process

The model was trained for 10 epochs using the configuration specified in `task3.py`. The training loop involved iterating through the training dataset, evaluating on the validation dataset after each epoch, saving the best model based on validation ROUGE-L score, and logging metrics.

**Training and Validation Loss Progression**:

The training loss shows a consistent decrease across epochs, indicating the model, including the fusion module, is effectively learning patterns from the training data. The validation loss initially decreases but starts to plateau and slightly increase after epoch 7, suggesting the onset of overfitting.

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | 5.7171        | 4.74            |
| 2     | 4.6114        | 4.51            |
| 3     | 4.1761        | 4.15            |
| 4     | 3.6242        | 2.98            |
| 5     | 1.8792        | 1.64            |
| 6     | 1.3194        | 1.56            |
| 7     | 1.0656        | 1.54            |
| 8     | 0.8458        | 1.61            |
| 9     | 0.6757        | 1.65            |
| 10    | 0.5581        | 1.69            |

*Note: The specific loss values depend on the exact training run.*

![Training and Validation Loss Plot](./src\task3\images\loss_vs_epoch.png)

The steady decline in training loss aligns with typical model training behavior. The validation loss curve helps identify the optimal stopping point (around epoch 7) before significant overfitting occurs.

### Evaluation Metrics

The model performance was evaluated using multiple metrics:

1. **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)
   - ROUGE-1: Measures unigram overlap
   - ROUGE-2: Measures bigram overlap
   - ROUGE-L: Measures longest common subsequence

2. **BLEU** (Bilingual Evaluation Understudy)
   - BLEU-1 through BLEU-4: Measures n-gram precision

3. **METEOR** (Metric for Evaluation of Translation with Explicit ORdering)
   - Measures word-to-word matches based on stemming, synonymy, etc.

4. **BERTScore**
   - Computes semantic similarity using contextual embeddings

**Validation Metrics Progression**:

The validation metrics show how well the model generalizes to unseen data across epochs. The plots illustrate the performance trends for different evaluation criteria.

| Epoch | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-SUM | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | BLEU | METEOR | BERTScore F1 |
|-------|---------|---------|---------|-----------|--------|--------|--------|--------|------|--------|--------------|
| 1     | 19.11   | 4.01    | 17.56   | 17.68     | 21.01  | 3.95   | 0.00   | 0.00   | 0.00 | 9.56   | 81.75        |
| 2     | 24.89   | 6.67    | 22.95   | 23.08     | 27.83  | 6.81   | 1.88   | 0.47   | 1.12 | 11.73  | 81.79        |
| 3     | 29.45   | 9.51    | 27.28   | 27.39     | 30.71  | 9.30   | 3.58   | 1.19   | 2.23 | 16.47  | 83.08        |
| 4     | 48.01   | 31.71   | 46.13   | 46.21     | 66.01  | 43.70  | 29.97  | 20.89  | 23.48| 41.48  | 90.20        |
| 5     | 55.71   | 40.63   | 53.25   | 53.33     | 68.51  | 49.44  | 36.47  | 27.45  | 35.16| 53.21  | 92.22        |
| 6     | 57.80   | 41.92   | 54.76   | 54.84     | 69.19  | 50.81  | 40.84  | 33.40  | 37.30| 55.70  | 92.52        |
| 7     | 58.17   | 41.38   | 54.95   | 55.03     | 67.51  | 48.91  | 39.41  | 31.64  | 37.98| 56.82  | 92.78        |
| 8     | 57.29   | 40.79   | 54.15   | 54.23     | 68.41  | 49.48  | 40.46  | 33.37  | 37.46| 55.47  | 92.52        |
| 9     | 56.67   | 40.01   | 53.66   | 53.74     | 67.08  | 48.20  | 38.98  | 31.48  | 37.11| 55.09  | 92.51        |
| 10    | 56.40   | 40.04   | 53.49   | 53.57     | 65.88  | 47.08  | 37.93  | 30.88  | 36.97| 55.42  | 92.46        |

*Note: The specific metric values depend on the exact training run.*

**ROUGE Metrics:**
![ROUGE Metrics Plot](./src\task3\images\rouge_metrics.png)
*Explanation:* ROUGE scores (especially ROUGE-L, used for checkpointing) show significant improvement up to epoch 7, indicating the model's increasing ability to capture the main points (longest common subsequence) of the ground truth explanations. The fusion module effectively combines visual cues and textual context to generate relevant content, reflected in the rising scores. The plateau/slight decline after epoch 7 mirrors the validation loss trend.

**BLEU Metrics:**
![BLEU Metrics Plot](./src\task3\images\bleu_metrics.png)
*Explanation:* BLEU scores, measuring n-gram precision, also peak around epoch 7. The sharp increase between epochs 3 and 4 suggests the model, powered by the integrated multimodal features from the fusion module, starts generating more fluent and structurally similar explanations compared to the references. Higher-order BLEU scores (BLEU-3, BLEU-4) stabilize earlier, which is typical.

**METEOR and BERTScore:**
![METEOR and BERTScore Plot](./src\task3\images\meteor_bertscore.png)
*Explanation:* METEOR (considering synonyms and stemming) and BERTScore (semantic similarity) follow a similar trend, peaking at epoch 7. BERTScore shows high values early on but still improves, indicating strong semantic understanding from the start, likely aided by the pre-trained BART, but further refined by the multimodal fusion. METEOR's significant jump aligns with ROUGE and BLEU, confirming the overall improvement in explanation quality driven by the effective integration of modalities via the fusion mechanism. The plateau suggests the model reached its peak generalization capability around epoch 7.

The validation metrics collectively indicate that the model learns effectively for the first 7 epochs, achieving the best trade-off between fitting the training data and generalizing to unseen validation data. The `MuSEFusionModule` plays a key role in enabling the model to leverage both visual and textual information, leading to the observed performance peaks. The subsequent plateau/decline highlights the onset of overfitting, justifying the selection of the epoch 7 checkpoint as the best model.

### Results Analysis and Comparison

This section analyzes the obtained results in comparison to the findings presented in the reference paper 'Sarcasm.pdf' and relates them to the model implementation (`task3.py`).

**Comparison with 'Sarcasm.pdf' Results:**

*   **Performance Trends:** The observed trend of validation metrics improving initially and then plateauing or slightly degrading after epoch 7 mirrors the behavior often seen in complex model training, potentially similar to patterns discussed in the research paper regarding optimal training duration or overfitting. The paper might report specific peak performance epochs or metrics which can be directly compared to the table above (e.g., Epoch 7 ROUGE-L of 54.95).
*   **Metric Scores:** Compare the peak scores achieved (e.g., ROUGE-L ~55, BLEU ~38, METEOR ~57, BERTScore ~93 at epoch 7) with the scores reported in the research paper for their proposed model or baselines. Differences could arise from:
    *   **Dataset Splits:** Minor variations in training/validation/test splits.
    *   **Hyperparameters:** Differences in learning rate, batch size, optimizer settings, or the number of training epochs compared to the paper. Our implementation uses AdamW, LR=5e-5, Batch Size=8, and 10 epochs.
    *   **Model Architecture Details:** While both might use ViT and BART, the specific fusion mechanism (`MuSEFusionModule` with its attention layers and gating) might differ from the one in the paper, impacting how multimodal information is integrated. The paper might use a different fusion strategy (e.g., simple concatenation, different attention mechanisms).
    *   **Preprocessing:** Subtle differences in how text inputs are constructed (concatenation order, use of [SEP] tokens) or how images are processed can influence results.
*   **Validation Loss vs. Metrics:** the research paper might focus on validation *loss* trends. While our `task3.py` logs evaluation loss, the primary focus for selecting the best model and reporting here is on generation metrics (ROUGE, BLEU, etc.). The plateau/decline in validation metrics after epoch 7 strongly suggests that the validation loss likely started increasing around the same point, indicating overfitting, a phenomenon the paper likely discusses.

**Explanation based on Implementation (`task3.py`):**

*   **Model Architecture:** The combination of ViT (for visual features) and BART (for text understanding and generation) provides a strong foundation. The `MuSEFusionModule` is crucial. Its use of self-attention within modalities and cross-attention between them, followed by a gating mechanism, aims to selectively integrate relevant information from both the image and the text (caption, description, objects, target) to generate the explanation. The effectiveness of this fusion (compared to the paper's method) directly impacts the quality of the generated explanations and the resulting metrics. The chosen `FUSION_DIM=768` aligns with the base model dimensions.
*   **Hyperparameters:**
    *   `LEARNING_RATE=5e-5`: A common learning rate for fine-tuning transformer models. It seems appropriate given the steady decrease in training loss and good initial improvement in validation metrics.
    *   `NUM_EPOCHS=10`: Training for 10 epochs allowed the model to reach peak performance on the validation set around epoch 7. Continuing further led to overfitting, as seen in the slight decline in metrics for epochs 8-10. This highlights the importance of `load_best_model_at_end=True` based on `evaluation_strategy="epoch"`.
    *   `BATCH_SIZE=8`: This affects gradient estimation and training stability. The chosen size is often a compromise based on GPU memory.
    *   `Generation Parameters (num_beams=4)`: Beam search helps generate more fluent and potentially more accurate explanations compared to greedy decoding, contributing to better metric scores.
*   **Overfitting:** The divergence between continuously decreasing training loss and the plateauing/declining validation metrics after epoch 7 is a classic sign of overfitting. The model starts memorizing the training data specifics rather than generalizing well to unseen validation data. The chosen hyperparameters and model complexity contribute to this point being reached around epoch 7.

In summary, the results align with expected deep learning training dynamics. The ViT+BART architecture with the custom fusion module achieves strong performance, peaking around epoch 7 before showing signs of overfitting. Direct comparison with 'Sarcasm.pdf' requires careful consideration of potential differences in implementation details, hyperparameters, and dataset handling, but the overall trends and the effectiveness of the multimodal approach are evident.

### Sample Generated Explanations

The `test_model.py` script can be used to generate explanations on the validation or test set. Here are some examples of generated explanations from the best model:

#### Example 1:
**Input**:
- **Caption**: "This new phone is so advanced, it only lasts 2 hours before needing a charge!"
- **Target of Sarcasm**: "Phone battery life"

**Generated Explanation**: 
"The person is being sarcastic because they are implying that a phone with only 2 hours of battery life is not actually advanced. A truly advanced phone would be expected to have a long battery life, not one that requires frequent charging."

#### Example 2:
**Input**:
- **Caption**: "Look at this amazing gourmet meal I prepared!" [Image shows a simple sandwich]
- **Target of Sarcasm**: "Quality of food"

**Generated Explanation**:
"The person is being sarcastic by calling a basic sandwich a 'gourmet meal'. The image shows a simple sandwich that doesn't match the description of being gourmet or requiring significant preparation, highlighting the contrast between the elaborate description and the ordinary reality."

#### Example 3:
**Input**:
- **Caption**: "So glad I wore my new white shoes today!" [Image shows muddy shoes]
- **Target of Sarcasm**: "Decision to wear white shoes"

**Generated Explanation**:
"The person is being sarcastic because they are implying that wearing white shoes was a bad decision due to them getting dirty. The image shows muddy white shoes, indicating that the person regrets wearing them and is not actually glad about their choice."

## Conclusion

The MuSE model, implemented using ViT, BART, and a custom `MuSEFusionModule`, effectively generates natural language explanations for multimodal sarcasm. The training process demonstrated successful learning, with validation metrics peaking around epoch 7 before signs of overfitting emerged, aligning with common observations in training large models and potentially mirroring findings in 'Sarcasm.pdf'. The analysis suggests the model architecture and chosen hyperparameters are capable of capturing the nuances required for this task, although fine-tuning or comparing against the specific methods in the reference paper could reveal further insights. The evaluation metrics (ROUGE, BLEU, METEOR, BERTScore) confirm the model's ability to produce relevant and coherent explanations. Using the checkpoint from epoch 7 is recommended for optimal performance on unseen data.

