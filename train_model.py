"""
Main training script for Pneumonia Detection Model - Simplified version
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.dataset import ChestXRayDataset
from src.model import PneumoniaCNN, PneumoniaResNet
from src.train import train_model


def main():
    # Configuration
    MODEL_TYPE = "CNN"  # "CNN" or "ResNet"
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.001
    PATIENCE = 5
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*70)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ChestXRayDataset(split="train", use_augmentation=True)
    val_dataset = ChestXRayDataset(split="val", use_augmentation=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print()
    
    # Create model
    print(f"Creating {MODEL_TYPE} model...")
    
    if MODEL_TYPE == "CNN":
        model = PneumoniaCNN(in_channels=1, num_classes=2)
        print("Using custom 4-layer CNN")
    else:
        model = PneumoniaResNet(num_classes=2, freeze_backbone=True)
        print("Using ResNet18 with transfer learning")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Training configuration:")
    print(f"  Loss: CrossEntropyLoss")
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Early stopping patience: {PATIENCE}")
    print("="*70)
    print()
    
    # Train
    print("Starting training...")
    print()
    
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE
    )
    
    print()
    print("="*70)
    print("Training completed!")
    print("Model saved to: outputs/models/best_model.pth")
    print("Training log saved to: outputs/logs/training_log.txt")
    print("="*70)


if __name__ == "__main__":
    main()
