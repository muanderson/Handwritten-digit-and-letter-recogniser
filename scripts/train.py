import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import mlflow
import mlflow.pytorch
from model import CNN
from data_loader import EMNISTDataset, transforms
from engine import train_epoch
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.utils import class_weight

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything()

    config = {
        'output_dir': r'models',
        'csv_path': r'C:\Users\Matthew\Documents\EMNIST\emnist-byclass-train.csv',
        'learning_rate': 1e-3,
        'epochs': 100,
        'patience': 10,
        'scheduler_patience': 5,
        'scheduler_factor': 0.1,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'batch_size': 32,
        'n_splits': 5,
        'use_class_weights': False,  
    }

    mlflow.set_experiment("EMNIST Model Training Improved CNN grad-cam")
    os.makedirs(config['output_dir'], exist_ok=True)

    # Load dataset once
    full_data = pd.read_csv(config['csv_path'])
    labels = full_data.iloc[:, 0].values

    kf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(labels)), labels)):
        config['fold'] = fold + 1

        with mlflow.start_run(run_name=f"Fold_{config['fold']}"):
            print(f"\n===== Starting Fold {config['fold']}/{config['n_splits']} =====")
            mlflow.log_params(config)

            class_weights_tensor = None
            if config['use_class_weights']:
                # Extract labels for the training split to calculate class weights
                train_labels = labels[train_idx]
                
                # --- Class weight calculation ---
                class_weights = class_weight.compute_class_weight(
                    'balanced',
                    classes=np.unique(train_labels),
                    y=train_labels
                )
                class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(config['device'])
                
                # Log class weights to MLflow for tracking
                mlflow.log_param("class_weights", class_weights_tensor.tolist())
                print("Using class weights...")
            else:
                print("Not using class weights...")

            train_dataset = EMNISTDataset(
                csv_path=config['csv_path'],
                transform=transforms(is_training=True),
                subset_indices=train_idx
            )
            val_dataset = EMNISTDataset(
                csv_path=config['csv_path'],
                transform=transforms(is_training=False),
                subset_indices=val_idx
            )

            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

            model = CNN().to(config['device'])

            # Pass class_weights_tensor to the training function
            best_val_acc, best_val_f1, best_model_path = train_epoch(model, train_loader, val_loader, config, config['device'], class_weights=class_weights_tensor)

            if best_model_path and os.path.exists(best_model_path):
                mlflow.pytorch.log_model(model, "model", registered_model_name=f"emnist-cnn-fold-{config['fold']}")

            mlflow.log_metric("best_validation_accuracy", best_val_acc)
            mlflow.log_metric("best_validation_f1", best_val_f1)

            print(f"Finished Fold: {config['fold']}")
            print(f"Best Val Acc: {best_val_acc:.4f}")
            print(f"Best Val F1: {best_val_f1:.4f}")
            fold_results.append((best_val_acc, best_val_f1))

    if fold_results:
        avg_acc = np.mean([r[0] for r in fold_results])
        avg_f1 = np.mean([r[1] for r in fold_results])
        print('\n=== Cross-Validation Results ===')
        for i, (acc, f1) in enumerate(fold_results, 1):
            print(f'Fold {i}: Best Val Acc={acc:.4f}, Best Val F1={f1:.4f}')
        print(f'Average Best Val Acc: {avg_acc:.4f}, Average Best Val F1: {avg_f1:.4f}')
    else:
        print("No folds were completed successfully.")

if __name__ == "__main__":
    main()