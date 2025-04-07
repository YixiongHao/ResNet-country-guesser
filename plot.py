import os
import sys
import glob
import json
import re
import matplotlib.pyplot as plt

PLOT_DIR = 'graphs'

def parse_model_settings(filename):
    """
    Extracts model settings from a filename.
    Expected format: "resnet<depth>_residual=<True|False>_transfer=<True|False>_history"
    
    Args:
        filename (str): The filename string.
        
    Returns:
        dict: A dictionary with keys "depth", "residual", and "transfer",
              or None if the string doesn't match the expected format.
    """
    pattern = r"resnet(\d+)_residual=(True|False)_transfer=(True|False)_history"
    match = re.search(pattern, filename)
    if match:
        depth = int(match.group(1))
        residual = match.group(2) == "True"
        transfer = match.group(3) == "True"
        return {"depth": depth, "residual": residual, "transfer": transfer}
    return None

def plot_history(history, settings):
    """
    Plot loss and accuracy curves for a single model.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    depth = settings['depth']
    residual = settings['residual']
    transfer = settings['transfer']
    outfile = f"{PLOT_DIR}/resnet{depth}_residual={residual}_transfer={transfer}.png"
    plt.savefig(outfile)
    print(f"Saved single model plot to {outfile}")

def plot_multiple_histories(folder):
    """
    Plots the validation accuracy curves for multiple models.
    Files are grouped by their 'residual' setting and each group
    is plotted on a separate graph.
    """
    files = glob.glob(os.path.join(folder, "*_history.json"))
    groups = {"True": [], "False": []}
    
    for file in files:
        settings = parse_model_settings(file)
        if settings is None:
            print(f"Skipping file (invalid format): {file}")
            continue
        with open(file, 'r') as f:
            data = json.load(f)
        history = data['history']
        # Group by residual setting as a string ("True" or "False")
        res_key = "True" if settings['residual'] else "False"
        groups[res_key].append((history, settings))
    
    for res_key, items in groups.items():
        if not items:
            print(f"No models found for residual={res_key}")
            continue
        plt.figure(figsize=(8, 6))
        # sort by increasing depth
        items.sort(key=lambda x: x[1]['depth'])
        for history, settings in items:
            epochs = range(1, len(history['val_acc']) + 1)
            label = f"resnet{settings['depth']}"
            plt.plot(epochs, history['val_acc'], label=label)
        plt.title(f"Validation Accuracy (Residual={res_key})")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy (%)")
        plt.legend()
        outfile = f"{PLOT_DIR}/residual={res_key}.png"
        plt.savefig(outfile)
        print(f"Saved multi-model plot for residual={res_key} to {outfile}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot.py <history_json_file | history_directory>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    
    if os.path.isdir(input_path):
        # Multiple models mode
        plot_multiple_histories(input_path)
    else:
        # Single model mode
        settings = parse_model_settings(input_path)
        if not settings:
            print("Error: File name does not match expected format.")
            sys.exit(1)
        with open(input_path, 'r') as f:
            data = json.load(f)
        plot_history(data['history'], settings)