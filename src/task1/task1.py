## Add imports here
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128  # Maximum sequence length for training

def initialise_projections(d_model, d_k):
    """
    create projections for Q, K, V.
    """
    return (
        nn.Linear(d_model, d_k, bias=False),
        nn.Linear(d_model, d_k, bias=False),
        nn.Linear(d_model, d_k, bias=False)
    )

def pairwise_similarities(Q, K):
    """
    Compute dot product attention.
    """
    # [batch, heads, seq_len, d_k] × [batch, heads, d_k, seq_len] -> [batch, heads, seq_len, seq_len]
    return torch.matmul(Q, K.transpose(-2, -1))

def attention_scaled(similarities, scale):
    """
    Scale the raw attention scores.
    """
    # Scale by sqrt(d_k)
    return similarities / scale

def attention_softmax(scaled_similarities, mask=None):
    """
    Normalize the scaled raw attention scores with softmax.
    """
    if mask is not None:
        scaled_similarities = scaled_similarities.masked_fill(mask == 0, -1e9)
    return F.softmax(scaled_similarities, dim=-1)

def compute_outputs(attention_weights, V):
    """
    Get outputs as a weighted sum of values by attention scores.
    """
    # [batch, heads, seq_len, seq_len] × [batch, heads, seq_len, d_k] -> [batch, heads, seq_len, d_k]
    return torch.matmul(attention_weights, V)

def make_causal_mask(seq_len):
    """
    Create a mask matrix that masks future context for the attention.
    """
    # Create a lower triangular matrix to mask future positions
    mask = torch.tril(torch.ones((seq_len, seq_len), device=DEVICE))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

def apply_causal_mask(similarities, mask):
    """
    Apply mask to attention.
    """
    return similarities.masked_fill(mask == 0, -1e9)

def split_heads(x, num_heads, depth):
    """
    Splitting the input across multiple heads.
    """
    batch_size, seq_length = x.shape[0], x.shape[1]
    
    # Reshape to [batch, seq_len, num_heads, depth]
    x = x.view(batch_size, seq_length, num_heads, depth)
    
    # Transpose to [batch, num_heads, seq_len, depth]
    return x.transpose(1, 2)

def merge_heads(x):
    """
    Reversing splitting action of function split_heads().
    """
    batch_size = x.shape[0]
    
    # Transpose back to [batch, seq_len, num_heads, depth]
    x = x.transpose(1, 2)
    
    # Merge the heads
    seq_length, num_heads, depth = x.shape[1], x.shape[2], x.shape[3]
    return x.contiguous().view(batch_size, seq_length, num_heads * depth)

def self_attention(Q, K, V, mask=None, scale=None):
    """
    Self-attention block.
    """
    # Compute raw attention scores
    similarities = pairwise_similarities(Q, K)
    
    # Scale the attention scores
    if scale is not None:
        similarities = attention_scaled(similarities, scale)
    
    # Apply causal mask if provided
    if mask is not None:
        similarities = apply_causal_mask(similarities, mask)
    
    # Apply softmax to get attention weights
    attention_weights = attention_softmax(similarities)
    
    # Compute the final attention output
    output = compute_outputs(attention_weights, V)
    
    return output, attention_weights

def split_heads_qkv(Q, K, V, num_heads, depth):
    """
    Split Q, K, V across multiple heads.
    """
    Q = split_heads(Q, num_heads, depth)
    K = split_heads(K, num_heads, depth)
    V = split_heads(V, num_heads, depth)
    return Q, K, V

def load_and_preprocess_data():
    # Try to use absolute paths for files first
    try:
        with open("./content/shakespear_train.txt", "r") as f:
            lines_train = f.readlines()
        print("Loaded train data locally")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find the train data file")
    
    try:
        with open("./content/shakespear_dev.txt", "r") as f:
            lines_dev = f.readlines()
        print("Loaded dev data locally")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find the dev data file")
    
    # For test data, try to load it or generate from train data
    try:
        with open("./content/shakespear_test.txt", "r") as f:
            lines_test = f.readlines()
        print("Loaded test data locally")
    except FileNotFoundError:
        print("Test file not found, generating from train data...")
        # Use a small portion of train data as test data
        random.seed(42)  # For reproducibility
        test_size = min(len(lines_train) // 20, 1000)  # 10% or max 100 lines
        test_indices = random.sample(range(len(lines_train)), test_size)
        lines_test = [lines_train[i] for i in test_indices]
        
        # Save the generated test data
        with open("./content/shakespear_test.txt", "w") as f:
            f.writelines(lines_test)
        print(f"Generated and saved {len(lines_test)} test examples")

    # Character-level tokenization
    all_text = ''.join(lines_train)
    chars = sorted(list(set(all_text)))
    
    # Create character-level tokenizer
    tokenizer = {c: i+3 for i, c in enumerate(chars)}  # Start from 3 to leave room for special tokens
    tokenizer['<PAD>'] = 0
    tokenizer['<START>'] = 1
    tokenizer['<STOP>'] = 2
    tokenizer['<UNK>'] = len(tokenizer)  # Add unknown token last

    # Create inverse tokenizer for decoding
    tokenizer_inv = {i: char for char, i in tokenizer.items()}

    # Prepare datasets
    data_train = lines_train
    data_val = lines_dev

    # Create input-output pairs
    class TextDataset(Dataset):
        def __init__(self, data, tokenizer, max_len=MAX_LEN):
            self.data = data
            self.tokenizer = tokenizer
            self.max_len = max_len
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            tokens = tokenize(self.data[idx], pad_to_len=self.max_len, tokenizer=self.tokenizer)
            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            return x, y

    train_dataset = TextDataset(data_train, tokenizer, MAX_LEN)
    val_dataset = TextDataset(data_val, tokenizer, MAX_LEN)

    return train_dataset, val_dataset, tokenizer, tokenizer_inv

def pad_to_length(tokens, max_len, tokenizer):
    """
    Pad tokens to a fixed length.
    """
    if len(tokens) >= max_len:
        return tokens[:max_len]
    else:
        return tokens + [tokenizer["<PAD>"]] * (max_len - len(tokens))

def tokenize(sentence, pad_to_len=None, tokenizer=None, include_stop=True):
    """
    Tokenize a sentence at character level.
    """
    tokens = [tokenizer["<START>"]] + [tokenizer.get(c, tokenizer["<UNK>"]) for c in sentence]
    if include_stop:
        tokens.append(tokenizer["<STOP>"])
    
    # Pad tokens if requested
    if pad_to_len is not None:
        tokens = pad_to_length(tokens, pad_to_len, tokenizer)
    
    return tokens

def decode(tokens, tokenizer_inv, end_at_stop=True, omit_pad=True):
    """
    Decode tokens to text.
    """
    chars = []
    for token in tokens:
        char = tokenizer_inv[token.item() if torch.is_tensor(token) else token]
        if char == "<STOP>" and end_at_stop:
            break
        if char == "<PAD>" and omit_pad:
            continue
        if char not in ["<START>", "<PAD>"]:
            chars.append(char)
    return ''.join(chars)

@torch.no_grad()
def evaluate_losses(data, model, tokenizer, bs=32, progress=True, pad_to_len=MAX_LEN):
    it = range(0, len(data), bs)
    if progress:
        it = tqdm(it)

    out = []
    for b_start in it:
        batch = slice(b_start, b_start + bs)
        tokens = torch.tensor(
            [tokenize(t, pad_to_len=pad_to_len, tokenizer=tokenizer) for t in data[batch]], dtype=torch.long
        ).to(DEVICE)
        X_tokens, y_tokens = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

        model.eval()
        logits, _ = model(X_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        y_log_probs = torch.gather(log_probs, 2, y_tokens[..., None])[..., 0]

        for i in range(y_tokens.shape[0]):
            # Create a mask for all special tokens
            special_tokens = [tokenizer["<PAD>"], tokenizer["<START>"], tokenizer["<STOP>"]]
            if "<UNK>" in tokenizer:
                special_tokens.append(tokenizer["<UNK>"])
            
            # Mask out all special tokens
            not_special = torch.ones_like(y_tokens[i], dtype=torch.bool)
            for token_id in special_tokens:
                not_special = not_special & (y_tokens[i] != token_id)
            
            # Only calculate loss on non-special tokens
            if not_special.any():
                loss = -y_log_probs[i, not_special].mean()
                out.append(loss.item())

    return out

def generate_text(model, tokenizer, tokenizer_inv, context="<START>", gen_tokens=10, temperature=0.6):
    """
    Generate a fixed number of tokens using the trained model.
    """
    # Convert string context to token IDs if it's a special token
    if context == "<START>":
        token_ids = [tokenizer["<START>"]]
    else:
        # Otherwise tokenize as usual
        token_ids = tokenize(context, tokenizer=tokenizer, include_stop=False)
        
    tokens = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)

    model.eval()
    with torch.no_grad():
        for _ in range(gen_tokens):
            # Get predictions
            logits, _ = model(tokens)
            
            # Focus on the last token's predictions
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the context
            tokens = torch.cat((tokens, next_token), dim=1)
            
            # Stop if we generated a STOP token
            if next_token.item() == tokenizer["<STOP>"]:
                break
         
    # Convert back to text
    return decode(tokens[0], tokenizer_inv)

## Define the Transformer model
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, 
                 d_ff=1024, dropout=0.1, max_len=MAX_LEN):        
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(self._create_positional_encoding(max_len, d_model))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()

    def _create_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)  # [1, max_len, d_model]
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        # x is [batch_size, seq_len]
        seq_len = x.shape[1]
        
        # Get token embeddings and add positional encoding
        token_emb = self.embedding(x)  # [batch_size, seq_len, d_model]
        pos_emb = self.pos_encoding[:, :seq_len, :]  # [1, seq_len, d_model]
        
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Create causal attention mask
        mask = make_causal_mask(seq_len)
        
        # Apply transformer blocks
        attentions = []
        for block in self.transformer_blocks:
            x, attn = block(x, mask)
            attentions.append(attn)
            
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        
        return logits, attentions


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Using GELU activation
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-norm architecture
        normed_x = self.norm1(x)
        attn_output, attention_weights = self.attention(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attn_output)
        
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)
        
        return x, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k])).to(DEVICE)
        
        # Create projections for Q, K, V and output
        self.q_proj, self.k_proj, self.v_proj = initialise_projections(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Project inputs to queries, keys, and values
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Split into multiple heads
        Q, K, V = split_heads_qkv(Q, K, V, self.nhead, self.d_k)
        
        # Compute attention
        attn_output, attention_weights = self_attention(Q, K, V, mask, self.scale)
        
        # Merge the heads back
        attn_output = merge_heads(attn_output)
        
        # Final projection
        output = self.output_proj(attn_output)
        
        return output, attention_weights


## Training function
def train_model(model, train_loader, val_dataset, tokenizer, tokenizer_inv, 
                optimizer, criterion, scheduler=None, epochs=20, val_data=None, 
                save_path="transformer_model.pt", patience=3, max_grad_norm=1.0):
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Training loop
        for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(x)
            
            # Reshape for CrossEntropyLoss: [batch_size*seq_len, vocab_size]
            logits = logits.contiguous().view(-1, logits.shape[-1])
            y = y.contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            if val_data is None:
                val_data = [" ".join(line.split()) for line in val_dataset.data]
            
            val_loss = np.mean(evaluate_losses(val_data, model, tokenizer))
            val_losses.append(val_loss)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}")

        # Generate a sample text
        sample_text = generate_text(
            model, 
            tokenizer, 
            tokenizer_inv, 
            context="<START>", 
            gen_tokens=20
        )
        
        print(f"Sample text: {sample_text}")
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            
            # Save the best model
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }
            
            # Save the best model checkpoint
            torch.save(best_model_state, save_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            no_improve_count += 1
            print(f"Validation loss did not improve. Patience: {no_improve_count}/{patience}")
        
        # Check if we should stop early
        if no_improve_count >= patience:
            print(f"Early stopping after {epoch+1} epochs as validation loss hasn't improved for {patience} epochs.")
            break
    
    # If training completes without early stopping, ensure we use the best model
    if best_model_state is not None and no_improve_count < patience:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f"Loaded the best model from epoch {best_model_state['epoch']+1} with validation loss: {best_model_state['val_loss']:.4f}")
    
    return train_losses, val_losses

def main():
    ## Load and preprocess data
    train_dataset, val_dataset, tokenizer, tokenizer_inv = load_and_preprocess_data()
    
    ## Create data loaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    ## Model hyperparameters
    vocab_size = len(tokenizer)
    d_model = 256
    nhead = 8
    num_layers = 4
    d_ff = 1024
    dropout = 0.1
    
    ## Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(DEVICE)
    
    ## Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    ## Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer["<PAD>"])
    
    ## Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    ## Train the model
    save_path = "transformer_shakespeare.pt"
    max_grad_norm = 1.0  # Setting the gradient clipping threshold
    
    val_data = [" ".join(line.split()) for line in val_dataset.data]
    train_losses, val_losses = train_model(
        model, 
        train_loader,
        val_dataset,
        tokenizer,
        tokenizer_inv,
        optimizer,
        criterion,
        scheduler=scheduler,
        val_data=val_data,
        save_path=save_path,
        max_grad_norm=max_grad_norm  # Pass the gradient clipping parameter
    )
    
    ## Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

    ## Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'tokenizer_inv': tokenizer_inv,
        'hyperparams': {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'd_ff': d_ff,
            'dropout': dropout
        }
    }, save_path)

    ## Evaluate on test data
    with open("./content/shakespear_test.txt", "r") as f:
        lines_test = f.readlines()
    
    test_data = [" ".join(line.split()) for line in lines_test]
    test_losses = evaluate_losses(test_data, model, tokenizer)
    test_ppl = math.exp(np.mean(test_losses))
    
    print(f"\nTest perplexity: {test_ppl:.4f}")


if __name__ == "__main__":
    main()


def inference(model_path, test_file, tokenizer, tokenizer_inv, gen_tokens=10, temperature=0.6):
    ## Load the saved model
    checkpoint = torch.load(model_path)
    
    # Extract hyperparameters
    hyperparams = checkpoint['hyperparams']
    
    # Initialize model with saved hyperparameters
    model = TransformerLM(
        vocab_size=hyperparams['vocab_size'],
        d_model=hyperparams['d_model'],
        nhead=hyperparams['nhead'],
        num_layers=hyperparams['num_layers'],
        d_ff=hyperparams['d_ff'],
        dropout=hyperparams['dropout']
    ).to(DEVICE)
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    ## Read and process the input from test.txt
    with open(test_file, "r") as f:
        lines_test = f.readlines()
    
    test_data = [" ".join(line.split()) for line in lines_test]
    
    ## Generate text and calculate perplexity
    generated_texts = []
    
    # Generate text for each line in test file
    for line in test_data:
        # Use the first few words as context
        context = line.split()[:3]
        context_str = " ".join(context)
        
        # Generate continuation
        generated = generate_text(
            model,
            tokenizer,
            tokenizer_inv,
            context=context_str,
            gen_tokens=gen_tokens,
            temperature=temperature
        )
        
        generated_texts.append({
            'context': context_str,
            'generated': generated
        })
    
    # Calculate perplexity on test set
    test_losses = evaluate_losses(test_data, model, tokenizer)
    test_ppl = math.exp(np.mean(test_losses))
    
    return generated_texts, test_ppl


# load the checkpoint first to get tokenizer and tokenizer_inv
checkpoint = torch.load("transformer_shakespeare.pt")
tokenizer = checkpoint['tokenizer']
tokenizer_inv = checkpoint['tokenizer_inv']

# use these variables in the function call
model_path = "transformer_shakespeare.pt"
test_file = "./content/shakespear_test.txt" 
generated_texts, ppl = inference(model_path, test_file, tokenizer, tokenizer_inv, gen_tokens=50, temperature=0.8)

## Print the generated text and perplexity
print(f"Test Perplexity: {ppl:.4f}\n")
print("Generated Text Samples:")
for i, sample in enumerate(generated_texts[:5]):  # Show first 5 samples
    print(f"\nSample {i+1}:")
    print(f"Context: '{sample['context']}'")
    print(f"Generated: '{sample['generated']}'")