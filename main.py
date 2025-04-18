import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import time
import json

### SETTINGS ###


# yall shouldnt need to change these two
MODEL_DIR = "/home/hice1/kboyce7/ResNet-country-guesser/config+history"
DATA_DIR = "~/scratch/resnet/data"
KAGGLE_DIR = "~/scratch/kaggle_dataset"
MIN_IMAGES_PER_CLASS = 50

# with H100s: python main.py --depth 18 --residual true --transfer false
# note: may need to try dropout + more augmentation to avoid overfitting

### SETTINGS ###


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet models on Country211 dataset"
    )
    parser.add_argument(
        "--depth",
        type=str,
        choices=["18", "34", "50", "101", "152"],
        default="18",
        help="ResNet depth (18, 34, 50, 101, 152)",
    )
    parser.add_argument(
        "--residual",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Use residual connections (true or false)",
    )
    parser.add_argument(
        "--transfer",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Use transfer learning (true or false)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["country", "geo"],
        default="country",
        help="which training set to use",
    )
    parser.add_argument(
        "--save",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Save the model parameters",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="Weight decay")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_dataset(dataset, train_transform):
    # Method to load dataset specified by args, returns -1 on failure, dataset on success
    print(f"Loading {dataset} dataset")
    if dataset == "country":
        try:
            # download=False after running code once
            country_dataset = datasets.Country211(
                root=DATA_DIR, download=False, transform=train_transform
            )
            return (country_dataset, 211)
        except Exception as e:
            print("Error: Country211 dataset not found in torchvision.")
            print(
                "You may need to use a custom loader or check your torchvision version."
            )
            print(e)
            return -1
    elif dataset == "geo":
        try:
            data = datasets.ImageFolder(KAGGLE_DIR, transform=train_transform)
            data = FilteredImageFolder(data, MIN_IMAGES_PER_CLASS)
            return (data, len(data.classes))
        except Exception as e:
            print("kaggle geoguesser data set failed to load")
            print(e)
            return -1


class FilteredImageFolder(ImageFolder):
    def __init__(self, base_dataset, min_samples=50):
        root = base_dataset.root
        transform = base_dataset.transform
        target_transform = base_dataset.target_transform
        loader = base_dataset.loader
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader)

        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        # Count label frequencies
        label_counts = defaultdict(int)
        for _, label in base_dataset.samples:
            label_counts[label] += 1

        # Keep only classes that have enough samples
        for label in base_dataset.classes:
            if label_counts[base_dataset.class_to_idx[label]] >= min_samples:
                self.class_to_idx[label] = len(self.classes)
                self.classes.append(label)

        # Fix samples list
        for path, label in base_dataset.samples:
            if label_counts[label] >= min_samples:
                original_class = base_dataset.classes[label]
                new_class = self.class_to_idx[original_class]
                self.samples.append((path, new_class))


# Custom ResNet without residual connections
class ResNetNoResidual(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ResNetNoResidual, self).__init__()
        # Copy the layers from the base model
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # Get convolutional blocks but modify to remove residual connections
        self.layer1 = self._modify_layer(base_model.layer1)
        self.layer2 = self._modify_layer(base_model.layer2)
        self.layer3 = self._modify_layer(base_model.layer3)
        self.layer4 = self._modify_layer(base_model.layer4)

        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def _modify_layer(self, layer):
        # Create a new Sequential container
        modified = nn.Sequential()

        # For each block in the layer
        for i, block in enumerate(layer):
            # Create a new block without skip connections
            mod_block = nn.Sequential()

            # Add all convolutions, batch norms, and activations
            # but skip the residual connection
            if hasattr(block, "conv1"):
                mod_block.add_module("conv1", block.conv1)
                mod_block.add_module("bn1", block.bn1)
                mod_block.add_module("relu1", nn.ReLU(inplace=True))

            if hasattr(block, "conv2"):
                mod_block.add_module("conv2", block.conv2)
                mod_block.add_module("bn2", block.bn2)
                mod_block.add_module("relu2", nn.ReLU(inplace=True))

            if hasattr(block, "conv3"):
                mod_block.add_module("conv3", block.conv3)
                mod_block.add_module("bn3", block.bn3)
                mod_block.add_module("relu3", nn.ReLU(inplace=True))

            modified.add_module(f"block{i}", mod_block)

        return modified

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def get_model(depth, use_residual, transfer_learning, num_classes):
    # Select the appropriate ResNet model based on depth
    if depth == "18":
        base_model = models.resnet18(
            weights="IMAGENET1K_V1" if transfer_learning else None
        )
    elif depth == "34":
        base_model = models.resnet34(
            weights="IMAGENET1K_V1" if transfer_learning else None
        )
    elif depth == "50":
        base_model = models.resnet50(
            weights="IMAGENET1K_V1" if transfer_learning else None
        )
    elif depth == "101":
        base_model = models.resnet101(
            weights="IMAGENET1K_V1" if transfer_learning else None
        )
    elif depth == "152":
        base_model = models.resnet152(
            weights="IMAGENET1K_V1" if transfer_learning else None
        )
    else:
        raise ValueError(f"Unsupported depth: {depth}")

    if use_residual:
        # If using residual connections, just modify the final layer
        model = base_model
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(model.fc.in_features, num_classes)
        )
    else:
        # If not using residual connections, create a custom model
        model = ResNetNoResidual(base_model, num_classes)

    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    # Training metrics
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"Epoch {epoch+1}/{epochs}")
        train_progress_bar = tqdm(train_loader, desc="Training")

        for inputs, targets in train_progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Update progress bar
            train_progress_bar.set_postfix(
                {
                    "loss": train_loss / train_total,
                    "acc": 100.0 * train_correct / train_total,
                }
            )

        train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc="Validation")

            for inputs, targets in val_progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                # Update progress bar
                val_progress_bar.set_postfix(
                    {
                        "loss": val_loss / val_total,
                        "acc": 100.0 * val_correct / val_total,
                    }
                )

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_epoch = epoch
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    return history, best_model_state, best_epoch


def main():
    args = parse_args()

    # Convert string arguments to appropriate types
    use_residual = args.residual.lower() == "true"
    transfer_learning = args.transfer.lower() == "true"
    save = args.save.lower() == "true"

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Data transformations
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if args.dataset == "geo":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Get dataset iterator
    dataset, num_classes = load_dataset(args.dataset, train_transform)
    if dataset == -1:
        return
    print(f"Loaded dataset: {len(dataset)} images, {len(dataset.classes)} classes")

    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Split dataset:")
    print(f"\tTraining set: {len(train_dataset)} images")
    print(f"\tValidation set: {len(val_dataset)} images")

    # Apply different transforms to validation set
    # if args.dataset == "country":
    #     val_dataset.dataset = datasets.Country211(
    #         root=DATA_DIR, download=False, transform=val_transform
    #     )
    # elif args.dataset == "geo":
    #     val_dataset.dataset = TransformedDataset(
    #         dataset, val_transform
    #     )
    val_dataset.dataset.transform = val_transform
    print(f"Applied transforms to validation dataset")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    # # Get the number of classes
    # num_classes = 211  # Country211 has 211 classes
    # if args.dataset == "geo":
    #     num_classes = 103  # Kaggle  has 211 classes

    # Create model
    model_name = (
        f"resnet{args.depth}_residual={use_residual}_transfer={transfer_learning}"
    )
    print(f"Creating model: {model_name}")
    model = get_model(args.depth, use_residual, transfer_learning, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Train the model
    print(f"Training model for {args.epochs} epochs...")
    start_time = time.time()
    history, best_model_state, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, args.epochs, device
    )
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    # Save the best model
    if save:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
        torch.save(best_model_state, model_path)
        print(f"Best model saved to {model_path}")

    # Save model configuration and history
    config = {
        "depth": args.depth,
        "residual": use_residual,
        "transfer_learning": transfer_learning,
        "num_classes": num_classes,
        "best_val_accuracy": max(history["val_acc"]),
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "training_time": elapsed_time,
        "history": history,
    }

    config_path = os.path.join(MODEL_DIR, f"{model_name}_history.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    print(f"Model configuration saved to {config_path}")


if __name__ == "__main__":
    main()
