import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the base directory (script's location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the paths (Relative to the script's location)
csv_file_path = os.path.join(BASE_DIR, "muse_res_10/muse_results/epoch_metrics.csv")
plots_dir = os.path.join(BASE_DIR, "images")

# Create plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)
print(f"Plots will be saved to: {os.path.abspath(plots_dir)}")

# Function to apply consistent styling to all plots
def apply_plot_styling(ax, title):
    ax.set_facecolor('black')
    ax.grid(True, color='gray', linestyle='-', alpha=0.6, linewidth=0.8)
    ax.set_title(title, color='white', fontsize=14)
    ax.set_xlabel('Epoch', color='white', fontsize=12)
    ax.set_ylabel('Score', color='white', fontsize=12)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

try:
    # Read the CSV file
    data = pd.read_csv(csv_file_path)
    print(f"Successfully loaded data with {len(data)} epochs")
    
    # Set plot style for better visuals
    plt.style.use('dark_background')
    
    # Plot 1: Training Loss vs Validation Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(data['Epoch'], data['Training Loss'], 'b-', label='Training Loss', marker='o', linewidth=2)
    plt.plot(data['Epoch'], data['Validation Loss'], 'r-', label='Validation Loss', marker='x', linewidth=2)
    ax.set_ylabel('Loss', color='white', fontsize=12)
    apply_plot_styling(ax, 'Training and Validation Loss vs Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_vs_epoch.png'), dpi=300, facecolor='black')
    plt.close()
    
    # Plot 2: All ROUGE Metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    rouge_metrics = ['Validation Rouge1', 'Validation Rouge2', 'Validation Rougel', 'Validation Rougelsum']
    colors = ['b', 'r', 'g', 'orange']
    
    for i, metric in enumerate(rouge_metrics):
        plt.plot(data['Epoch'], data[metric], color=colors[i], marker='o', linewidth=2, 
                 label=metric.replace('Validation ', ''))
    
    apply_plot_styling(ax, 'ROUGE Metrics vs Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rouge_metrics.png'), dpi=300, facecolor='black')
    plt.close()
    
    # Plot 3: All BLEU Metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    bleu_metrics = ['Validation Bleu1', 'Validation Bleu2', 'Validation Bleu3', 'Validation Bleu4', 'Validation Bleu']
    colors = ['b', 'r', 'g', 'orange', 'purple']
    
    for i, metric in enumerate(bleu_metrics):
        plt.plot(data['Epoch'], data[metric], color=colors[i], marker='o', linewidth=2, 
                 label=metric.replace('Validation ', ''))
    
    apply_plot_styling(ax, 'BLEU Metrics vs Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bleu_metrics.png'), dpi=300, facecolor='black')
    plt.close()
    
    # Plot 4: METEOR and BERTScore
    fig, ax = plt.subplots(figsize=(12, 6))
    other_metrics = ['Validation Meteor', 'Validation Bertscore F1']
    colors = ['b', 'r']
    
    for i, metric in enumerate(other_metrics):
        plt.plot(data['Epoch'], data[metric], color=colors[i], marker='o', linewidth=2, 
                 label=metric.replace('Validation ', ''))
    
    apply_plot_styling(ax, 'METEOR and BERTScore vs Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'meteor_bertscore.png'), dpi=300, facecolor='black')
    plt.close()
    
    print(f"All plots have been saved to {os.path.abspath(plots_dir)}")
    # List files in the directory to confirm they were saved
    saved_files = os.listdir(plots_dir)
    print(f"Files in directory: {saved_files}")
    
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Uncomment the following line if you want to display plots instead of just saving them
# plt.show()
