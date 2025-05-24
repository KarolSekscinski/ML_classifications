# gcs/ft_transformer_pipeline.py
# ft_transformer_pipeline.py (Updated May 24, 2025)
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchmetrics
import rtdl  # Revisiting Tabular Deep Learning library
from sklearn.metrics import confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import time
from io import BytesIO
import json
import os
import optuna  # For hyperparameter tuning

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')


# --- Helper Functions (Reusing from previous scripts) ---

def load_data_from_gcs(gcs_bucket, gcs_path):
    """Loads CSV data from GCS into a pandas DataFrame. Feature names are ignored here."""
    logging.info(f"Loading data from: gs://{gcs_bucket}/{gcs_path}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, gcs_path)
        if data_bytes is None: raise FileNotFoundError(f"Could not download GCS file: gs://{gcs_bucket}/{gcs_path}")
        df = pd.read_csv(BytesIO(data_bytes))
        logging.info(f"Data loaded successfully from gs://{gcs_bucket}/{gcs_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from gs://{gcs_bucket}/{gcs_path}: {e}")
        raise


def save_plot_to_gcs(fig, gcs_bucket, gcs_blob_name):
    """Saves a matplotlib figure to GCS as PNG."""
    logging.info(f"Saving plot to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_bytes = buf.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, plot_bytes, gcs_blob_name, content_type='image/png')
        logging.info(f"Plot successfully saved to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
        plt.close(fig)
    except Exception as e:
        logging.error(f"ERROR saving plot to GCS ({gcs_blob_name}): {e}")
        plt.close(fig)


def save_model_to_gcs(model_state_dict, gcs_bucket, gcs_blob_name, metadata=None):
    """Saves a PyTorch model state dictionary to GCS, optionally with metadata."""
    logging.info(f"Saving model state_dict to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        with BytesIO() as buf:
            torch.save({'state_dict': model_state_dict, 'metadata': metadata or {}}, buf)
            buf.seek(0)
            model_bytes = buf.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model state_dict successfully saved to gs://{gcs_bucket}/{gcs_blob_name}")
        if metadata:
            logging.info(f"Model metadata included: {metadata}")
    except Exception as e:
        logging.error(f"ERROR saving model state_dict to GCS ({gcs_blob_name}): {e}")


# --- PyTorch Dataset Definition for FT-Transformer ---
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, Y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.int64)
        self.Y = torch.tensor(Y, dtype=torch.int64)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.Y[idx]


# --- Training & Evaluation Functions ---
def train_epoch_ft(model, loader, criterion, optimizer, device, epoch_num):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for i, (x_num, x_cat, targets) in enumerate(loader):
        x_num, x_cat, targets = x_num.to(device), x_cat.to(device), targets.to(device)
        targets_float = targets.float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(x_num, x_cat)
        loss = criterion(outputs, targets_float)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Reduced logging frequency for cleaner output during tuning
        # if (i + 1) % 100 == 0: logging.info(f'Epoch {epoch_num}, Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}')
    avg_loss = total_loss / len(loader)
    epoch_duration = time.time() - start_time
    # logging.info(f'Epoch {epoch_num} Training Summary: Average Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s')
    return avg_loss, epoch_duration


def evaluate_model_ft(model, loader, criterion, device, trial_num=None, epoch_num=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    metrics_collection = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="binary").to(device),
        'precision': torchmetrics.Precision(task="binary").to(device),
        'recall': torchmetrics.Recall(task="binary").to(device),
        'f1': torchmetrics.F1Score(task="binary").to(device),
        'roc_auc': torchmetrics.AUROC(task="binary").to(device),
        'pr_auc': torchmetrics.AveragePrecision(task="binary").to(device)
    }).to(device)

    with torch.no_grad():
        for x_num, x_cat, targets in loader:
            x_num, x_cat, targets = x_num.to(device), x_cat.to(device), targets.to(device)
            targets_float = targets.float().unsqueeze(1)
            targets_int = targets.int()
            outputs_logits = model(x_num, x_cat)
            loss = criterion(outputs_logits, targets_float)
            total_loss += loss.item()
            outputs_probs = torch.sigmoid(outputs_logits).squeeze()
            metrics_collection.update(outputs_probs, targets_int)
            outputs_preds = (outputs_probs > 0.5).int()
            all_preds.append(outputs_preds.cpu().numpy())
            all_targets.append(targets_int.cpu().numpy())
    avg_loss = total_loss / len(loader)
    final_metrics = metrics_collection.compute()
    all_preds_np = np.concatenate(all_preds) if all_preds else np.array([])
    all_targets_np = np.concatenate(all_targets) if all_targets else np.array([])

    cm_list = [[0, 0], [0, 0]]  # Default CM if no predictions
    if all_preds_np.size > 0 and all_targets_np.size > 0:
        cm = confusion_matrix(all_targets_np, all_preds_np)
        cm_list = cm.tolist()

    metrics_dict = {k: v.item() for k, v in final_metrics.items()}
    metrics_dict["confusion_matrix"] = cm_list

    log_prefix = ""
    if trial_num is not None: log_prefix += f"Trial {trial_num} "
    if epoch_num is not None: log_prefix += f"Epoch {epoch_num} "

    # logging.info(f"--- {log_prefix}Model Evaluation ---")
    # logging.info(f"Average Loss: {avg_loss:.4f}")
    # for name, value in metrics_dict.items():
    #     if name != "confusion_matrix": logging.info(f"{name.replace('_', ' ').title()}: {value:.4f}")
    # logging.info(f"Confusion Matrix:\n{cm_list}")
    return avg_loss, metrics_dict, (all_targets_np, torch.sigmoid(model(x_num.to(device), x_cat.to(
        device))).detach().cpu().numpy().squeeze() if x_num.nelement() > 0 else np.array([]))


# --- Plotting Function ---
def plot_evaluation_charts(y_true_np, y_pred_proba_np, cm, model_name, gcs_bucket, gcs_output_prefix):
    """Generates and saves Confusion Matrix, ROC, and PR curve plots."""
    if y_true_np.size == 0 or y_pred_proba_np.size == 0:
        logging.warning("Skipping plot generation due to empty true or predicted values.")
        return
    # 1. Confusion Matrix Plot
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_name} Confusion Matrix')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        save_plot_to_gcs(fig_cm, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    except Exception as e:
        logging.error(f"Failed to generate or save Confusion Matrix plot: {e}")
    # 2. ROC Curve Plot
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_roc, name=model_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
        ax_roc.set_title(f'{model_name} ROC Curve')
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_roc_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save ROC Curve plot: {e}")
    # 3. Precision-Recall Curve Plot
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_pr, name=model_name)
        ax_pr.set_title(f'{model_name} PR Curve')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_pr_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save PR Curve plot: {e}")


# --- Optuna Objective Function ---
def objective(trial, args, device, train_loader, test_loader, n_num_features, cat_cardinalities):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    d_token = trial.suggest_categorical("d_token", [64, 128, 192, 256])
    n_blocks = trial.suggest_int("n_blocks", 1, 4)
    attention_dropout = trial.suggest_float("attention_dropout", 0.0, 0.5)
    ffn_dropout = trial.suggest_float("ffn_dropout", 0.0, 0.5)
    residual_dropout = trial.suggest_float("residual_dropout", 0.0, 0.3)
    ffn_factor = trial.suggest_float("ffn_factor", 1.0, 2.0)  # (e.g. 4/3 for ReGLU)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    # Recalculate ffn_d_hidden based on current trial's d_token and ffn_factor
    calculated_ffn_d_hidden = int(d_token * ffn_factor)

    # Update DataLoaders if batch_size is tuned
    current_train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    current_test_loader = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities if cat_cardinalities else None,
        d_token=d_token,
        n_blocks=n_blocks,
        attention_dropout=attention_dropout,
        ffn_d_hidden=calculated_ffn_d_hidden,
        ffn_dropout=ffn_dropout,
        residual_dropout=residual_dropout,
        d_out=1
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    logging.info(f"Trial {trial.number}: Starting training with params: {trial.params}")

    for epoch in range(1, args.epochs + 1):  # Use fixed epochs for tuning, or tune epochs too
        train_loss, _ = train_epoch_ft(model, current_train_loader, criterion, optimizer, device, epoch)
        # Optional: Evaluate on a validation set here if available and report intermediate values
        # trial.report(validation_metric, epoch)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    # Evaluate on the test set for this trial
    _, test_metrics, _ = evaluate_model_ft(model, current_test_loader, criterion, device, trial_num=trial.number)

    # Optuna tries to minimize, so return a metric where lower is better (e.g. 1 - AUC)
    # Or maximize by returning AUC directly and setting direction='maximize' in study.
    metric_to_optimize = test_metrics.get(args.optimization_metric, 0.0)
    logging.info(f"Trial {trial.number}: Finished. Test {args.optimization_metric}: {metric_to_optimize:.4f}")
    return metric_to_optimize


# --- Main Execution ---
def main(args):
    logging.info("--- Starting FT-Transformer Training Pipeline with Hyperparameter Tuning ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')
    METADATA_URI = args.metadata_uri

    # --- 1. Setup Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Using CPU.")

    # --- 2. Load FT-Transformer Specific Metadata ---
    logging.info(f"Loading FT-Transformer specific metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata = gcs_utils.load_json_from_gcs(GCS_BUCKET, metadata_blob_name)
        if metadata is None: raise FileNotFoundError("Could not download or parse metadata file.")
        logging.info("Metadata loaded successfully.")

        data_paths = metadata.get("gcs_paths", {})
        x_train_num_path = data_paths.get("X_train_num_scaled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_train_cat_path = data_paths.get("X_train_cat_indices", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_train_path = data_paths.get("y_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_test_num_path = data_paths.get("X_test_num_scaled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_test_cat_path = data_paths.get("X_test_cat_indices", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_test_path = data_paths.get("y_test", "").replace(f"gs://{GCS_BUCKET}/", "")

        required_paths = [x_train_num_path, x_train_cat_path, y_train_path, x_test_num_path, x_test_cat_path,
                          y_test_path]
        if not all(required_paths):
            missing = [name for name, path in
                       zip(["x_train_num", "x_train_cat", "y_train", "x_test_num", "x_test_cat", "y_test"],
                           required_paths) if not path]
            raise ValueError(f"Missing required data paths in metadata: {missing}.")

        cat_cardinalities = metadata.get("cat_feature_cardinalities")
        if not cat_cardinalities: raise ValueError("Categorical cardinalities not found in metadata.")
        numerical_features_used = metadata.get("numerical_features_used", [])
        categorical_features_used = metadata.get("categorical_features_used", [])

    except Exception as e:
        logging.error(f"Failed to load or parse metadata from {METADATA_URI}: {e}")
        return

    # --- 3. Load Processed Data ---
    try:
        X_train_num_df = load_data_from_gcs(GCS_BUCKET, x_train_num_path)
        X_train_cat_df = load_data_from_gcs(GCS_BUCKET, x_train_cat_path)
        y_train_df = load_data_from_gcs(GCS_BUCKET, y_train_path)
        X_test_num_df = load_data_from_gcs(GCS_BUCKET, x_test_num_path)
        X_test_cat_df = load_data_from_gcs(GCS_BUCKET, x_test_cat_path)
        y_test_df = load_data_from_gcs(GCS_BUCKET, y_test_path)
    except Exception as e:
        logging.error(f"Failed to load data from GCS paths: {e}")
        return

    X_train_num_np = X_train_num_df.values.astype(np.float32)
    X_train_cat_np = X_train_cat_df.values.astype(np.int64)
    y_train_np = y_train_df.iloc[:, 0].values.astype(np.int64)
    X_test_num_np = X_test_num_df.values.astype(np.float32)
    X_test_cat_np = X_test_cat_df.values.astype(np.int64)
    y_test_np = y_test_df.iloc[:, 0].values.astype(np.int64)

    train_dataset = TabularDataset(X_train_num_np, X_train_cat_np, y_train_np)
    test_dataset = TabularDataset(X_test_num_np, X_test_cat_np, y_test_np)

    # Initial batch size, may be overridden by Optuna per trial
    # For Optuna's objective function, we pass the original datasets, not dataloaders tied to a specific batch size initially.
    # Or, pass dataloaders with a default batch_size and let Optuna create new ones if batch_size is tuned.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_default, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_default, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    n_num_features = X_train_num_np.shape[1]
    if len(cat_cardinalities) != X_train_cat_np.shape[1]:
        logging.error(
            f"Mismatch: cardinalities ({len(cat_cardinalities)}) vs cat features ({X_train_cat_np.shape[1]}).")
        return

    # --- 4. Hyperparameter Tuning with Optuna ---
    logging.info(f"--- Starting Hyperparameter Tuning (Optuna) for {args.n_trials} trials ---")
    study = optuna.create_study(direction="maximize")  # Maximize based on selected metric (e.g. roc_auc or pr_auc)
    study.optimize(
        lambda trial: objective(trial, args, device, train_loader, test_loader, n_num_features, cat_cardinalities),
        n_trials=args.n_trials)

    best_hyperparams = study.best_params
    best_metric_value = study.best_value
    logging.info(f"--- Hyperparameter Tuning Finished ---")
    logging.info(f"Best {args.optimization_metric}: {best_metric_value:.4f}")
    logging.info(f"Best Hyperparameters: {best_hyperparams}")

    # --- 5. Train Final Model with Best Hyperparameters ---
    logging.info("--- Training Final Model with Best Hyperparameters ---")

    # Get best params from study
    final_lr = best_hyperparams["lr"]
    final_weight_decay = best_hyperparams["weight_decay"]
    final_d_token = best_hyperparams["d_token"]
    final_n_blocks = best_hyperparams["n_blocks"]
    final_attention_dropout = best_hyperparams["attention_dropout"]
    final_ffn_dropout = best_hyperparams["ffn_dropout"]
    final_residual_dropout = best_hyperparams["residual_dropout"]
    final_ffn_factor = best_hyperparams["ffn_factor"]
    final_batch_size = best_hyperparams.get("batch_size", args.batch_size_default)  # Use tuned batch_size

    final_calculated_ffn_d_hidden = int(final_d_token * final_ffn_factor)

    # Re-create DataLoaders with the best batch size
    final_train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    final_test_loader = DataLoader(test_dataset, batch_size=final_batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    final_model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities if cat_cardinalities else None,
        d_token=final_d_token,
        n_blocks=final_n_blocks,
        attention_dropout=final_attention_dropout,
        ffn_d_hidden=final_calculated_ffn_d_hidden,
        ffn_dropout=final_ffn_dropout,
        residual_dropout=final_residual_dropout,
        d_out=1
    ).to(device)
    logging.info(f"Final FT-Transformer Model Architecture:\n{final_model}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=final_lr, weight_decay=final_weight_decay)

    total_train_time = 0
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):  # Use original epochs for final training
        avg_loss, duration = train_epoch_ft(final_model, final_train_loader, criterion, optimizer, device, epoch)
        epoch_losses.append(avg_loss)
        total_train_time += duration
        logging.info(f'Final Training Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Duration: {duration:.2f}s')
    logging.info(f"--- Final Model Training Finished --- Total Duration: {total_train_time:.2f}s")

    # --- 6. Evaluate Final Model on Test Set ---
    logging.info("Evaluating final model on test set...")
    test_loss, test_metrics, (y_true_np, y_pred_proba_np) = evaluate_model_ft(final_model, final_test_loader, criterion,
                                                                              device)

    logging.info("--- Final Model Evaluation (Test Set) ---")
    logging.info(f"Average Loss: {test_loss:.4f}")
    for name, value in test_metrics.items():
        if name != "confusion_matrix": logging.info(f"{name.replace('_', ' ').title()}: {value:.4f}")
    logging.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # --- 7. Generate and Save Plots for Final Model ---
    test_cm_np = np.array(test_metrics["confusion_matrix"])
    plot_evaluation_charts(y_true_np, y_pred_proba_np, test_cm_np, 'FT-Transformer (Best)', GCS_BUCKET,
                           GCS_OUTPUT_PREFIX)

    # --- 8. Save Final Model and Metadata ---
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/ft_transformer_best_model_state_dict.pt"
    model_metadata = {"best_hyperparameters": best_hyperparams, "best_metric_value": best_metric_value,
                      "optimized_metric": args.optimization_metric}
    save_model_to_gcs(final_model.state_dict(), GCS_BUCKET, model_blob_name, metadata=model_metadata)

    log_summary = {
        "model_type": "FT-Transformer",
        "tuning_args": {"n_trials": args.n_trials, "epochs_per_trial": args.epochs,
                        "optimization_metric": args.optimization_metric},
        "best_hyperparameters": best_hyperparams,
        "best_tuned_metric_value": best_metric_value,
        "device_used": str(device),
        "metadata_source": METADATA_URI,
        "n_numerical_features": n_num_features,
        "categorical_cardinalities": cat_cardinalities,
        "final_model_training_duration_seconds": total_train_time,
        "final_model_training_avg_loss_per_epoch": epoch_losses,
        "final_model_test_set_evaluation": {"loss": test_loss, "metrics": test_metrics},
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }
    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/ft_transformer_final_training_log.json"
    gcs_utils.save_json_to_gcs(log_summary, GCS_BUCKET, log_blob_name)
    logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")

    logging.info("--- FT-Transformer Training Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, tune, and evaluate an FT-Transformer model.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    parser.add_argument("--metadata-uri", type=str, required=True,
                        help="GCS URI of the preprocessing_ft_metadata.json file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")

    # Tuning and Training Control
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs FOR EACH TRIAL and for FINAL training.")  # Reduced for faster tuning example
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials for hyperparameter tuning.")
    parser.add_argument("--optimization-metric", type=str, default="roc_auc", choices=['roc_auc', 'pr_auc', 'f1'],
                        help="Metric to optimize during tuning.")
    parser.add_argument("--batch-size-default", type=int, default=256,
                        help="Default batch size if not tuned, or for initial loader setup.")

    # Fixed arguments (not tuned by default in this setup, but could be)
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    # Removed specific FT-Transformer hyperparams like --ft-d-token etc. as they are tuned now.
    # Removed SHAP arguments

    args = parser.parse_args()
    main(args)