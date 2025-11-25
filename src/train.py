import torch
import os
from pathlib import Path
from tqdm import tqdm

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


def write_log(message, log_file):
    """Append message to the log file with timestamp."""
    from datetime import datetime
    os.makedirs(log_file.parent, exist_ok=True)
    with open(log_file, "a", encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")


def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs=20, log_path=None, patience=5):
    """
    Train the model with early stopping support.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epochs: Maximum number of epochs
        patience: Early stopping patience
    """
    if log_path is None:
        log_path = PROJECT_ROOT / "outputs" / "logs" / "training_log.txt"
    else:
        log_path = Path(log_path)
    
    model_save_path = PROJECT_ROOT / "outputs" / "models" / "best_model.pth"
    os.makedirs(model_save_path.parent, exist_ok=True)
    
    train_losses, val_losses = [], []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # ================== TRAINING ==================
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # ================== VALIDATION ==================
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs,labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # ================== LOGGING ==================
        msg = (f"Epoch [{epoch+1}/{epochs}] "
               f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
               f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(msg)
        write_log(msg, log_file=log_path)
        
        # ================== MODEL SAVING & EARLY STOPPING ==================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            torch.save(model.state_dict(), model_save_path)
            save_msg = f"Best model saved (Val Acc: {best_val_acc:.2f}%, Val Loss: {avg_val_loss:.4f})"
            print(save_msg)
            write_log(save_msg, log_file=log_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stop_msg = f"Early stopping triggered after {epoch+1} epochs (patience={patience})"
                print(early_stop_msg)
                write_log(early_stop_msg, log_file=log_path)
                break
        
        print("-" * 70)

    final_msg = f"Training completed. Best Val Acc: {best_val_acc:.2f}%, Best Val Loss: {best_val_loss:.4f}"
    write_log(final_msg, log_file=log_path)
    print(f"\n{final_msg}")
    
    return train_losses, val_losses
