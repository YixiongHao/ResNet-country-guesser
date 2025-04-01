import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import time
import json

### SETTINGS ###

#yall shouldnt need to change these two
MODEL_DIR = '~/scratch/resnet/models'
DATA_DIR = '~/scratch/resnet/data'

# with H100s: python main.py --depth 18 --residual true --transfer false
# note: may to try dropout + more augmentation to avoid overfitting

### SETTINGS ###

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet models on Country211 dataset')
    parser.add_argument('--depth', type=str, choices=['18', '34', '50'], default='18',
                        help='ResNet depth (18, 34, or 50)')
    parser.add_argument('--residual', type=str, choices=['true', 'false'], default='true',
                        help='Use residual connections (true or false)')
    parser.add_argument('--transfer', type=str, choices=['true', 'false'], default='true',
                        help='Use transfer learning (true or false)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

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
            if hasattr(block, 'conv1'):
                mod_block.add_module('conv1', block.conv1)
                mod_block.add_module('bn1', block.bn1)
                mod_block.add_module('relu1', nn.ReLU(inplace=True))
            
            if hasattr(block, 'conv2'):
                mod_block.add_module('conv2', block.conv2)
                mod_block.add_module('bn2', block.bn2)
                mod_block.add_module('relu2', nn.ReLU(inplace=True))
            
            if hasattr(block, 'conv3'):
                mod_block.add_module('conv3', block.conv3)
                mod_block.add_module('bn3', block.bn3)
                mod_block.add_module('relu3', nn.ReLU(inplace=True))
            
            modified.add_module(f'block{i}', mod_block)
            
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
        x = self.fc(x)
        
        return x

def get_model(depth, use_residual, transfer_learning, num_classes):
    # Select the appropriate ResNet model based on depth
    if depth == '18':
        base_model = models.resnet18(weights='IMAGENET1K_V1' if transfer_learning else None)
    elif depth == '34':
        base_model = models.resnet34(weights='IMAGENET1K_V1' if transfer_learning else None)
    elif depth == '50':
        base_model = models.resnet50(weights='IMAGENET1K_V1' if transfer_learning else None)
    else:
        raise ValueError(f"Unsupported depth: {depth}")
    
    if use_residual:
        # If using residual connections, just modify the final layer
        model = base_model
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # If not using residual connections, create a custom model
        model = ResNetNoResidual(base_model, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    # Training metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
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
            train_progress_bar.set_postfix({
                'loss': train_loss / train_total,
                'acc': 100. * train_correct / train_total
            })
        
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
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
                val_progress_bar.set_postfix({
                    'loss': val_loss / val_total,
                    'acc': 100. * val_correct / val_total
                })
        
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_epoch = epoch
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            
    return history, best_model_state, best_epoch

def main():
    args = parse_args()
    
    # Convert string arguments to appropriate types
    use_residual = args.residual.lower() == 'true'
    transfer_learning = args.transfer.lower() == 'true'
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Country211 dataset
    print("Loading Country211 dataset...")
    try:
        # download=False after running code once
        dataset = datasets.Country211(root=DATA_DIR, download=False, transform=train_transform)
    except:
        print("Error: Country211 dataset not found in torchvision.")
        print("You may need to use a custom loader or check your torchvision version.")
        return
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply different transforms to validation set
    val_dataset.dataset = datasets.Country211(root=DATA_DIR, download=False, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    # Get the number of classes
    num_classes = 211  # Country211 has 211 classes
    
    # Create model
    model_name = f"resnet{args.depth}_residual={use_residual}_transfer={transfer_learning}"
    print(f"Creating model: {model_name}")
    model = get_model(args.depth, use_residual, transfer_learning, num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    print(f"Training model for {args.epochs} epochs...")
    start_time = time.time()
    history, best_model_state, best_epoch = train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Save the best model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    torch.save(best_model_state, model_path)
    print(f"Best model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(MODEL_DIR, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")
    
    # Save model configuration
    config = {
        'depth': args.depth,
        'residual': use_residual,
        'transfer_learning': transfer_learning,
        'num_classes': num_classes,
        'best_val_accuracy': max(history['val_acc']),
        'best_epoch': best_epoch,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'training_time': elapsed_time
    }
    
    config_path = os.path.join(MODEL_DIR, f"{model_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"Model configuration saved to {config_path}")

if __name__ == "__main__":
    main()