# gcs/mlp_pipeline.py
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
from sklearn.metrics import confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import time
from io import BytesIO
import json
import os
import optuna  # For hyperparameter tuning

from ..gcs_utils import *

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')


# --- Helper Functions (Adapted for PyTorch & GCS) ---
def load_data_from_gcs(gcs_bucket, gcs_path, feature_names=None):
    logging.info(f"Loading data from: gs://{gcs_bucket}/{gcs_path}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, gcs_path)
        if data_bytes is None:
            raise FileNotFoundError(f"Could not download GCS file: gs://{gcs_bucket}/{gcs_path}")
        df = pd.read_csv(BytesIO(data_bytes))
        if feature_names is not None and not df.columns.equals(pd.Index(feature_names)):
            if len(feature_names) == len(df.columns):
                logging.warning(f"Assigning provided feature names to loaded data from {gcs_path}.")
                df.columns = feature_names
            else:
                logging.error(
                    f"Feature name count ({len(feature_names)}) does not match column count ({len(df.columns)}) in {gcs_path}.")
        logging.info(f"Data loaded successfully from gs://{gcs_bucket}/{gcs_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from gs://{gcs_bucket}/{gcs_path}: {e}")
        raise


def save_plot_to_gcs(fig, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving plot to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_bytes = buf.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, plot_bytes, gcs_blob_name, content_type='image/png')
        logging.info(f"Plot successfully saved to gs://{gcs_bucket}/{gcs_blob_name}")
        plt.close(fig)
    except Exception as e:
        logging.error(f"ERROR saving plot to GCS ({gcs_blob_name}): {e}")
        plt.close(fig)


def save_model_to_gcs(model_state_dict, gcs_bucket, gcs_blob_name, metadata=None):
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


# --- PyTorch Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout_rate=0.3):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- Training & Evaluation Functions ---
def train_epoch(model, loader, criterion, optimizer, device, epoch_num):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    epoch_duration = time.time() - start_time
    return avg_loss, epoch_duration


def evaluate_model(model, loader, criterion, device, trial_num=None, epoch_num=None):
    model.eval()
    total_loss = 0.0
    all_preds_list = []
    all_targets_list = []
    all_probs_list = []

    metrics_collection = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="binary").to(device),
        'precision': torchmetrics.Precision(task="binary").to(device),
        'recall': torchmetrics.Recall(task="binary").to(device),
        'f1': torchmetrics.F1Score(task="binary").to(device),
        'roc_auc': torchmetrics.AUROC(task="binary").to(device),
        'pr_auc': torchmetrics.AveragePrecision(task="binary").to(device)
    }).to(device)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets_dev = inputs.to(device), targets.to(device)
            targets_float = targets_dev.float().unsqueeze(1)
            targets_int = targets_dev.int()

            outputs_logits = model(inputs)
            loss = criterion(outputs_logits, targets_float)
            total_loss += loss.item()

            outputs_probs = torch.sigmoid(outputs_logits).squeeze()
            outputs_preds = (outputs_probs > 0.5).int()

            metrics_collection.update(outputs_probs, targets_int)
            all_preds_list.append(outputs_preds.cpu().numpy())
            all_targets_list.append(targets_int.cpu().numpy())  # targets were already on device, now to cpu
            all_probs_list.append(outputs_probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    final_metrics = metrics_collection.compute()

    all_preds_np = np.concatenate(all_preds_list) if all_preds_list else np.array([])
    all_targets_np = np.concatenate(all_targets_list) if all_targets_list else np.array([])
    all_probs_np = np.concatenate(all_probs_list) if all_probs_list else np.array([])

    cm_list = [[0, 0], [0, 0]]
    if all_preds_np.size > 0 and all_targets_np.size > 0:
        cm = confusion_matrix(all_targets_np, all_preds_np)
        cm_list = cm.tolist()

    metrics_dict = {k: v.item() for k, v in final_metrics.items()}
    metrics_dict["confusion_matrix"] = cm_list

    return avg_loss, metrics_dict, (all_targets_np, all_probs_np)


def plot_evaluation_charts(y_true_np, y_pred_proba_np, cm, model_name, gcs_bucket, gcs_output_prefix):
    if y_true_np.size == 0 or y_pred_proba_np.size == 0:
        logging.warning("Skipping plot generation due to empty true or predicted values.")
        return
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_name} Confusion Matrix')
        save_plot_to_gcs(fig_cm, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    except Exception as e:
        logging.error(f"Failed to generate or save CM plot: {e}")
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_roc, name=model_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
        ax_roc.set_title(f'{model_name} ROC Curve')
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_roc_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save ROC plot: {e}")
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_pr, name=model_name)
        ax_pr.set_title(f'{model_name} PR Curve')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_pr_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save PR plot: {e}")


# --- Optuna Objective Function ---
def objective(trial, args, device, train_loader, test_loader, input_dim):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(trial.suggest_categorical(f"h_dim_l{i}", [32, 64, 128, 256]))

    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

    current_train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    current_test_loader = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    model = SimpleMLP(input_dim, hidden_dims, dropout_rate=dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    logging.info(f"Trial {trial.number}: Starting training with params: {trial.params}")
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = train_epoch(model, current_train_loader, criterion, optimizer, device, epoch)

    _, test_metrics, _ = evaluate_model(model, current_test_loader, criterion, device, trial_num=trial.number)
    metric_to_optimize = test_metrics.get(args.optimization_metric, 0.0)
    logging.info(f"Trial {trial.number}: Finished. Test {args.optimization_metric}: {metric_to_optimize:.4f}")
    return metric_to_optimize


# --- Main Execution ---
def main(args):
    logging.info("--- Starting MLP Training Pipeline with Hyperparameter Tuning ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')
    METADATA_URI = args.metadata_uri

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Using CPU.")

    logging.info(f"Loading metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata = gcs_utils.load_json_from_gcs(GCS_BUCKET, metadata_blob_name)
        if metadata is None: raise FileNotFoundError("Could not download or parse metadata file.")
        processed_feature_names = metadata.get("processed_feature_names")
        if not processed_feature_names: raise ValueError("Processed feature names not found.")
        data_paths = metadata.get("gcs_paths", {})
        x_train_path = data_paths.get("X_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_train_path = data_paths.get("y_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_test_path = data_paths.get("X_test_processed", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_test_path = data_paths.get("y_test", "").replace(f"gs://{GCS_BUCKET}/", "")
        if not all([x_train_path, y_train_path, x_test_path, y_test_path]):
            raise ValueError("Missing data paths in metadata.")
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        return

    X_train_df = load_data_from_gcs(GCS_BUCKET, x_train_path, processed_feature_names)
    y_train_df = load_data_from_gcs(GCS_BUCKET, y_train_path)
    X_test_df = load_data_from_gcs(GCS_BUCKET, x_test_path, processed_feature_names)
    y_test_df = load_data_from_gcs(GCS_BUCKET, y_test_path)

    X_train_np = X_train_df.values.astype(np.float32)
    y_train_np = y_train_df.iloc[:, 0].values.astype(np.int64)
    X_test_np = X_test_df.values.astype(np.float32)
    y_test_np = y_test_df.iloc[:, 0].values.astype(np.int64)

    X_train_tensor = torch.from_numpy(X_train_np)
    y_train_tensor = torch.from_numpy(y_train_np)
    X_test_tensor = torch.from_numpy(X_test_np)
    y_test_tensor = torch.from_numpy(y_test_np)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_default, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_default, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    input_dim = X_train_tensor.shape[1]

    logging.info(f"--- Starting Hyperparameter Tuning (Optuna) for {args.n_trials} trials ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args, device, train_loader, test_loader, input_dim),
                   n_trials=args.n_trials)

    best_hyperparams = study.best_params
    best_metric_value = study.best_value
    logging.info(f"--- Hyperparameter Tuning Finished ---")
    logging.info(f"Best {args.optimization_metric}: {best_metric_value:.4f}")
    logging.info(f"Best Hyperparameters: {best_hyperparams}")

    logging.info("--- Training Final Model with Best Hyperparameters ---")
    final_lr = best_hyperparams["lr"]
    final_dropout = best_hyperparams["dropout_rate"]
    final_n_layers = best_hyperparams["n_layers"]
    final_hidden_dims = [best_hyperparams[f"h_dim_l{i}"] for i in range(final_n_layers)]
    final_batch_size = best_hyperparams.get("batch_size", args.batch_size_default)

    final_train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    final_test_loader = DataLoader(test_dataset, batch_size=final_batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    final_model = SimpleMLP(input_dim, final_hidden_dims, dropout_rate=final_dropout).to(device)
    logging.info(f"Final MLP Model Architecture:\n{final_model}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=final_lr)

    total_train_time = 0;
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        avg_loss, duration = train_epoch(final_model, final_train_loader, criterion, optimizer, device, epoch)
        epoch_losses.append(avg_loss);
        total_train_time += duration
        logging.info(f'Final Training Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Duration: {duration:.2f}s')
    logging.info(f"--- Final Model Training Finished --- Total Duration: {total_train_time:.2f}s")

    logging.info("Evaluating final model on test set...")
    test_loss, test_metrics, (y_true_np, y_pred_proba_np) = evaluate_model(final_model, final_test_loader, criterion,
                                                                           device)
    logging.info(f"Final Model Test Loss: {test_loss:.4f}, Metrics: {test_metrics}")

    test_cm_np = np.array(test_metrics["confusion_matrix"])
    plot_evaluation_charts(y_true_np, y_pred_proba_np, test_cm_np, 'MLP (Best)', GCS_BUCKET, GCS_OUTPUT_PREFIX)

    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/mlp_best_model_state_dict.pt"
    model_metadata = {"best_hyperparameters": best_hyperparams, "best_metric_value": best_metric_value,
                      "optimized_metric": args.optimization_metric}
    save_model_to_gcs(final_model.state_dict(), GCS_BUCKET, model_blob_name, metadata=model_metadata)

    log_summary = {
        "model_type": "MLP",
        "tuning_args": {"n_trials": args.n_trials, "epochs_per_trial": args.epochs,
                        "optimization_metric": args.optimization_metric},
        "best_hyperparameters": best_hyperparams,
        "best_tuned_metric_value": best_metric_value,
        "device_used": str(device),
        "metadata_source": METADATA_URI,
        "final_model_training_duration_seconds": total_train_time,
        "final_model_training_avg_loss_per_epoch": epoch_losses,
        "final_model_test_set_evaluation": {"loss": test_loss, "metrics": test_metrics},
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }
    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/mlp_final_training_log.json"
    gcs_utils.save_json_to_gcs(log_summary, GCS_BUCKET, log_blob_name)
    logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")

    logging.info("--- MLP Training Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, tune, and evaluate an MLP model.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    parser.add_argument("--metadata-uri", type=str, required=True,
                        help="GCS URI of the preprocessing_metadata.json file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")

    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs for each trial and final model.")  # Reduced for tuning example
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials.")
    parser.add_argument("--optimization-metric", type=str, default="roc_auc", choices=['roc_auc', 'pr_auc', 'f1'],
                        help="Metric to optimize.")
    parser.add_argument("--batch-size-default", type=int, default=512, help="Default batch size.")

    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    # Removed --mlp-hidden-dims, --mlp-dropout, --learning-rate as they are tuned.
    # Removed SHAP arguments

    args = parser.parse_args()
    main(args)