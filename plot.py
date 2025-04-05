import json
import sys
import matplotlib.pyplot as plt
import re

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
    # edit path as needed
    depth = settings['depth']
    residual = settings['residual']
    transfer = settings['transfer']
    plt.savefig(f'{PLOT_DIR}/resnet{depth}_residual={residual}_transfer={transfer}.png')

# Example usage:
if __name__ == '__main__':
    # Replace with your actual history dict
    if len(sys.argv) < 2:
        print("Usage: python plot.py <history_json_file>")
        sys.exit(1)
    json_file = sys.argv[1]
    settings = parse_model_settings(json_file)
    with open(json_file, 'r') as f:
        history = json.load(f)['history']
    plot_history(history, settings)