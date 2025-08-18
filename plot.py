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

def compare_histories(file1, file2):
    """
    Plots a comparison of the loss and accuracy for two history files.
    Two separate figures are produced: one for loss and one for accuracy.
    Legends indicate whether each model was trained with or without residual connections.
    """
    histories = []
    settings_list = []
    files = [file1, file2]
    
    for file in files:
        settings = parse_model_settings(file)
        if settings is None:
            print(f"Error: File {file} does not match expected format.")
            sys.exit(1)
        with open(file, 'r') as f:
            data = json.load(f)
        history = data['history']
        histories.append(history)
        settings_list.append(settings)
        
    # Assume histories have same number of epochs
    epochs = range(1, len(histories[0]['val_acc']) + 1)
    
    # Plot Loss Comparison
    plt.figure(figsize=(8, 6))
    for history, settings in zip(histories, settings_list):
        residual_label = "with residual" if settings['residual'] else "without residual"
        label_train = f"resnet{settings['depth']} ({residual_label}) - Train Loss"
        label_val = f"resnet{settings['depth']} ({residual_label}) - Val Loss"
        plt.plot(epochs, history['train_loss'], label=label_train)
        plt.plot(epochs, history['val_loss'], label=label_val)
    plt.title("Comparison of Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    outfile = f"{PLOT_DIR}/comparison_loss.png"
    plt.savefig(outfile)
    print(f"Saved comparison loss plot to {outfile}")
    
    # Plot Accuracy Comparison
    plt.figure(figsize=(8, 6))
    for history, settings in zip(histories, settings_list):
        residual_label = "with residual" if settings['residual'] else "without residual"
        label_train = f"resnet{settings['depth']} ({residual_label}) - Train Accuracy"
        label_val = f"resnet{settings['depth']} ({residual_label}) - Val Accuracy"
        plt.plot(epochs, history['train_acc'], label=label_train)
        plt.plot(epochs, history['val_acc'], label=label_val)
    plt.title("Comparison of Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    outfile = f"{PLOT_DIR}/comparison_accuracy.png"
    plt.savefig(outfile)
    print(f"Saved comparison accuracy plot to {outfile}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python plot.py single <history_json_file>")
        print("  python plot.py multiple <history_folder>")
        print("  python plot.py comparison <history_json_file1> <history_json_file2>")
        sys.exit(1)
        
    mode = sys.argv[1].lower()
    
    if mode == "single":
        input_path = sys.argv[2]
        settings = parse_model_settings(input_path)
        if not settings:
            print("Error: File name does not match expected format.")
            sys.exit(1)
        with open(input_path, 'r') as f:
            data = json.load(f)
        plot_history(data['history'], settings)
        
    elif mode == "multiple":
        folder = sys.argv[2]
        if not os.path.isdir(folder):
            print("Error: Provided path is not a directory.")
            sys.exit(1)
        plot_multiple_histories(folder)
        
    elif mode == "comparison":
        if len(sys.argv) != 4:
            print("Error: Comparison mode requires two history files.")
            sys.exit(1)
        file1 = sys.argv[2]
        file2 = sys.argv[3]
        compare_histories(file1, file2)
        
    else:
        print("Error: Mode must be one of 'single', 'multiple', or 'comparison'.")
        sys.exit(1)