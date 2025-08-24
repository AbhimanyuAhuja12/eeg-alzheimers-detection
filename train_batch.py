"""
EEGNet Training Script for Alzheimer's EEG Detection
---------------------------------------------------
Make sure you have:
- eeg_net.py (EEGNet model)
- eeg_dataset.py (dataset class)
- model-data/labels.json + EEG files
"""

import os
import json
import mne
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from eeg_net import EEGNet
from eeg_dataset import EEGDataset

# --------------------------
# Setup
# --------------------------
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable CUDA if available
try:
    mne.utils.set_config('MNE_USE_CUDA', 'true')
    mne.cuda.init_cuda(verbose=True)
except Exception as e:
    print("CUDA init failed, falling back to CPU:", e)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Create output folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("images", exist_ok=True)

# --------------------------
# Model Parameters (Fixed to match your data)
# --------------------------
num_chans = 19
timepoints = 1425  # Fixed to match actual data
num_classes = 3
F1, D, F2, dropout_rate = 152, 5, 760, 0.5

eegnet_model = EEGNet(
    num_channels=num_chans,
    timepoints=timepoints,
    num_classes=num_classes,
    F1=F1,
    D=D,
    F2=F2,
    dropout_rate=dropout_rate,
)

print(eegnet_model)
print(f"Model params: num_channels={num_chans}, timepoints={timepoints}, "
      f"num_classes={num_classes}, F1={F1}, D={D}, F2={F2}, dropout_rate={dropout_rate}")

# --------------------------
# Load Data with Proper Train/Test Split
# --------------------------
data_dir = "model-data"
data_file = "labels.json"

with open(os.path.join(data_dir, data_file), "r") as file:
    data_info = json.load(file)

# Check what types exist in your data
available_types = set(d.get("type", "unknown") for d in data_info)
print(f"Available data types in labels.json: {available_types}")

# Try to use existing train/test split if available
train_data = [d for d in data_info if d.get("type") == "train"]
test_data = [d for d in data_info if d.get("type") == "test"]

# If no test data exists, create a train/test split
if len(test_data) == 0:
    print("No test data found, creating train/test split from available data...")
    
    # Use all available data if no specific train data exists
    if len(train_data) == 0:
        all_data = data_info
        print(f"Using all {len(all_data)} samples for train/test split")
    else:
        all_data = train_data
        print(f"Using {len(train_data)} training samples for train/test split")
    
    # Stratified split to maintain class balance
    labels = [d["label"] for d in all_data]
    train_data, test_data = train_test_split(
        all_data, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    print(f"Created train/test split: {len(train_data)} train, {len(test_data)} test")

# Balance training set
labels = ["A", "C", "F"]
class_groups = {label: [d for d in train_data if d["label"] == label] for label in labels}
min_samples = min(len(v) for v in class_groups.values())
balanced_train_data = sum([random.sample(v, min_samples) for v in class_groups.values()], [])

print("Before balancing:", {k: len(v) for k, v in class_groups.items()})
print(f"After balancing: each class -> {min_samples}")
print(f"Total balanced training samples: {len(balanced_train_data)}")

# Test data class distribution
test_class_groups = {label: [d for d in test_data if d["label"] == label] for label in labels}
print("Test data distribution:", {k: len(v) for k, v in test_class_groups.items()})

# Dataset + Dataloaders
train_dataset = EEGDataset(data_dir, balanced_train_data)
test_dataset = EEGDataset(data_dir, test_data)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=SubsetRandomSampler(range(len(train_dataset))))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")

# Add debug info for data shapes
if len(train_dataset) > 0:
    sample_inputs, sample_labels = train_dataset[0]
    print(f"Sample input shape: {sample_inputs.shape}")
    print(f"Sample label: {sample_labels}")

# --------------------------
# Training Setup
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(eegnet_model.parameters(), lr=0.0007)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
epochs = 50  # Reduce for faster testing

eegnet_model.to(device)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
best_test_loss = float('inf')

# --------------------------
# Training Loop
# --------------------------
print(f"Starting training for {epochs} epochs...")
print("-" * 60)

for epoch in range(epochs):
    start_time = time.time()
    
    # Training phase
    eegnet_model.train()
    train_loss = 0.0
    train_correct, train_total = 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = eegnet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Evaluation phase
    if len(test_loader) > 0:  # Only evaluate if we have test data
        eegnet_model.eval()
        test_loss = 0.0
        test_correct, test_total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = eegnet_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': eegnet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'test_accuracy': test_accuracy
            }, "outputs/eegnet_best.pth")

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:5.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:5.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {time.time()-start_time:5.1f}s")
    else:
        # No test data available - only show training metrics
        test_losses.append(0)  # Placeholder
        test_accuracies.append(0)  # Placeholder
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:5.2f}% | "
              f"No Test Data | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {time.time()-start_time:5.1f}s")

print("-" * 60)
print("✅ Training complete!")

# --------------------------
# Save Results
# --------------------------

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss curves
ax1.plot(train_losses, label="Train Loss", color='blue')
if len(test_loader) > 0:
    ax1.plot(test_losses, label="Test Loss", color='red')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Test Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2.plot(train_accuracies, label="Train Accuracy", color='blue')
if len(test_loader) > 0:
    ax2.plot(test_accuracies, label="Test Accuracy", color='red')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training and Test Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/training_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print("📊 Training curves saved at images/training_curves.png")

# Save final model
torch.save({
    'epoch': epochs,
    'model_state_dict': eegnet_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies,
    'model_params': {
        'num_channels': num_chans,
        'timepoints': timepoints,
        'num_classes': num_classes,
        'F1': F1,
        'D': D,
        'F2': F2,
        'dropout_rate': dropout_rate
    }
}, "outputs/eegnet_final.pth")

print("💾 Final model saved at outputs/eegnet_final.pth")
if len(test_loader) > 0:
    print("💾 Best model saved at outputs/eegnet_best.pth")

# Print final statistics
print("\n" + "="*50)
print("FINAL TRAINING STATISTICS")
print("="*50)
print(f"Final Train Loss: {train_losses[-1]:.4f}")
print(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
if len(test_loader) > 0:
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Best Test Loss: {best_test_loss:.4f}")
else:
    print("No test data was available for evaluation")

print("\n" + "="*50)
print("HOW TO LOAD THE SAVED MODEL:")
print("="*50)
print("# For CPU loading:")
print("checkpoint = torch.load('outputs/eegnet_final.pth', map_location='cpu')")
print("model.load_state_dict(checkpoint['model_state_dict'])")
print("\n# For CUDA loading (if available):")
print("checkpoint = torch.load('outputs/eegnet_final.pth', map_location=device)")
print("model.load_state_dict(checkpoint['model_state_dict'])")