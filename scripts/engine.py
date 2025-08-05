import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import mlflow 

def train_epoch(model, train_loader, val_loader, config, device, class_weights=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.get('scheduler_factor', 0.1),
        patience=config.get('scheduler_patience', 5)
    )
    
    best_f1 = float('-inf')
    best_val_acc = 0
    best_model_path = None
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        all_preds_train, all_labels_train = [], []

        for image, label in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            image, label = image.to(device), label.to(device).long()
            prediction = model(image)
            loss = criterion(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_preds_train.append(prediction.argmax(dim=1).cpu().numpy())
            all_labels_train.append(label.cpu().numpy())
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(np.concatenate(all_labels_train), np.concatenate(all_preds_train))
        train_f1 = f1_score(np.concatenate(all_labels_train), np.concatenate(all_preds_train), average='weighted')
        
        # --- Validation Loop ---
        model.eval()
        all_preds_val, all_labels_val = [], []
        with torch.no_grad():
            for image, label in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                image, label = image.to(device), label.to(device).long()
                prediction = model(image)
                all_preds_val.append(prediction.argmax(dim=1).cpu().numpy())
                all_labels_val.append(label.cpu().numpy())
                
        val_acc = accuracy_score(np.concatenate(all_labels_val), np.concatenate(all_preds_val))
        val_f1 = f1_score(np.concatenate(all_labels_val), np.concatenate(all_preds_val), average='weighted')

        scheduler.step(val_f1)

        # --- MLflow Logging (per epoch) ---
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("validation_accuracy", val_acc, step=epoch)
        mlflow.log_metric("validation_f1", val_f1, step=epoch)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        print(f"\nEpoch {epoch+1}/{config['epochs']} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # --- Early Stopping & Model Checkpointing ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_val_acc = val_acc # Store the accuracy associated with the best F1
            patience_counter = 0
            model_path = os.path.join(config['output_dir'], f"best_model_fold_{config['fold']}.pt")
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path # Update the path to the best model
            print(f"Validation F1 improved to {best_f1:.4f}. Saving model to {model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{config['patience']}")

        if patience_counter >= config['patience']:
            print("Early stopping triggered.")
            break

    return best_val_acc, best_f1, best_model_path