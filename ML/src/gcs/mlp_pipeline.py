# mlp_pipeline.py
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics # For efficient metric calculation on tensors
from sklearn.metrics import (confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split # For validation split if needed
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
from io import BytesIO, StringIO
import json
import os

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')

# --- Helper Functions (Adapted for PyTorch & GCS) ---

def load_data_from_gcs(gcs_bucket, gcs_path, feature_names=None):
    """Loads CSV data from GCS into a pandas DataFrame."""
    # (Identical to CPU scripts - could be in a shared utils file)
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
                 logging.error(f"Feature name count ({len(feature_names)}) does not match column count ({len(df.columns)}) in {gcs_path}.")
        logging.info(f"Data loaded successfully from gs://{gcs_bucket}/{gcs_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from gs://{gcs_bucket}/{gcs_path}: {e}")
        raise

def save_plot_to_gcs(fig, gcs_bucket, gcs_blob_name):
    """Saves a matplotlib figure to GCS as PNG."""
    # (Identical to CPU scripts)
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

def save_model_to_gcs(model_state_dict, gcs_bucket, gcs_blob_name):
    """Saves a PyTorch model state dictionary to GCS."""
    logging.info(f"Saving model state_dict to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        with BytesIO() as buf:
            torch.save(model_state_dict, buf)
            buf.seek(0)
            model_bytes = buf.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model state_dict successfully saved to gs://{gcs_bucket}/{gcs_blob_name}")
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
            layers.append(nn.BatchNorm1d(hidden_dim)) # Batch norm often helps
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        # No final activation here, as BCEWithLogitsLoss includes sigmoid

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

        # Ensure target is float and has the right shape for BCEWithLogitsLoss
        targets = targets.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 100 == 0: # Log progress every 100 batches
             logging.info(f'Epoch {epoch_num}, Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(loader)
    epoch_duration = time.time() - start_time
    logging.info(f'Epoch {epoch_num} Training Summary: Average Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s')
    return avg_loss, epoch_duration

def evaluate_model(model, loader, criterion, device, num_classes=2):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    # Initialize torchmetrics metrics, move them to the correct device
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
            inputs, targets = inputs.to(device), targets.to(device)
            targets_float = targets.float().unsqueeze(1) # For loss calculation
            targets_int = targets.int() # For torchmetrics

            outputs_logits = model(inputs)
            loss = criterion(outputs_logits, targets_float)
            total_loss += loss.item()

            # Get probabilities (apply sigmoid) and predicted class (threshold at 0.5)
            outputs_probs = torch.sigmoid(outputs_logits).squeeze()
            outputs_preds = (outputs_probs > 0.5).int()

            # Update metrics
            metrics_collection.update(outputs_probs, targets_int) # Use probs for AUC, preds implicit for others

            # Store for confusion matrix later (move to CPU)
            all_preds.append(outputs_preds.cpu().numpy())
            all_targets.append(targets_int.cpu().numpy())

    avg_loss = total_loss / len(loader)
    final_metrics = metrics_collection.compute() # Compute all metrics at once

    # Concatenate batch results for CM
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    cm = confusion_matrix(all_targets, all_preds)

    # Convert tensor metrics to float for logging/JSON
    metrics_dict = {k: v.item() for k, v in final_metrics.items()}
    metrics_dict["confusion_matrix"] = cm.tolist()

    logging.info("--- Model Evaluation (Test Set) ---")
    logging.info(f"Average Loss: {avg_loss:.4f}")
    for name, value in metrics_dict.items():
        if name != "confusion_matrix":
             logging.info(f"{name.replace('_', ' ').title()}: {value:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")

    return avg_loss, metrics_dict, cm # Return raw CM for plotting

def plot_evaluation_charts(y_true_np, y_pred_proba_np, cm, model_name, gcs_bucket, gcs_output_prefix):
    """Generates and saves Confusion Matrix, ROC, and PR curve plots."""
    # This function needs numpy arrays on CPU
    # 1. Confusion Matrix Plot
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_name} Confusion Matrix')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        save_plot_to_gcs(fig_cm, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_confusion_matrix.png")
    except Exception as e:
        logging.error(f"Failed to generate or save Confusion Matrix plot: {e}")

    # 2. ROC Curve Plot - Use sklearn for display from probabilities
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_roc, name=model_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
        ax_roc.set_title(f'{model_name} Receiver Operating Characteristic (ROC) Curve')
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_roc_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save ROC Curve plot: {e}")

    # 3. Precision-Recall Curve Plot - Use sklearn for display from probabilities
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_pr, name=model_name)
        ax_pr.set_title(f'{model_name} Precision-Recall (PR) Curve')
        save_plot_to_gcs(fig_pr, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_pr_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save PR Curve plot: {e}")


def perform_shap_analysis(model, X_train_tensor, X_test_tensor, feature_names, device, gcs_bucket, gcs_output_prefix, sample_size=100):
    logging.info("--- SHAP Analysis (MLP - DeepExplainer) ---")
    logging.info(f"Using background sample size: {min(sample_size, X_train_tensor.shape[0])}")
    logging.info(f"Explaining sample size: {min(sample_size, X_test_tensor.shape[0])}")

    try:
        model.eval() # Ensure model is in eval mode

        indices_bg = np.random.choice(X_train_tensor.shape[0], min(sample_size, X_train_tensor.shape[0]), replace=False)
        X_train_summary = X_train_tensor[indices_bg].to(device)

        indices_test = np.random.choice(X_test_tensor.shape[0], min(sample_size, X_test_tensor.shape[0]), replace=False)
        X_test_sample = X_test_tensor[indices_test].to(device)

        start_shap_time = time.time()
        logging.info("Initializing SHAP DeepExplainer...")
        explainer = shap.DeepExplainer(model, X_train_summary)

        logging.info(f"Calculating SHAP values for {X_test_sample.shape[0]} test samples...")
        shap_values_output = explainer.shap_values(X_test_sample) # Get the raw output
        end_shap_time = time.time()
        logging.info(f"SHAP values calculated in {end_shap_time - start_shap_time:.2f} seconds.")

        # Robustly handle SHAP output type
        shap_values_focused = None
        if isinstance(shap_values_output, list):
            # For binary MLP with 1 output logit, DeepExplainer often gives a list with 1 item.
            # If model had 2 output logits (e.g., for CrossEntropyLoss), shap_values_output[1] might be for positive class.
            if len(shap_values_output) > 0:
                shap_values_focused = shap_values_output[0] # Assuming first/only item is for the single logit
                if len(shap_values_output) > 1:
                     logging.warning(f"SHAP explainer returned a list of {len(shap_values_output)} items. Using the first one.")
            else:
                raise ValueError("SHAP explainer returned an empty list.")
        else: # Assuming it's a single tensor or numpy array
            shap_values_focused = shap_values_output

        # Now convert the focused SHAP values to numpy
        if torch.is_tensor(shap_values_focused):
            shap_values_np = shap_values_focused.cpu().numpy()
        elif isinstance(shap_values_focused, np.ndarray):
            shap_values_np = shap_values_focused # It's already a NumPy array
        else:
            raise TypeError(f"Unexpected type for SHAP values after focusing: {type(shap_values_focused)}")

        # X_test_sample is definitely a tensor, so .cpu().numpy() is safe
        X_test_sample_np = X_test_sample.cpu().numpy()
        X_test_sample_df = pd.DataFrame(X_test_sample_np, columns=feature_names)

        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values_np, X_test_sample_df, plot_type="dot", show=False)
        plt.title("SHAP Summary Plot (MLP)")
        try:
            plt.tight_layout()
        except Exception: # Can sometimes fail with certain backends/versions
            logging.warning("Could not apply tight_layout() to SHAP plot.")
        save_plot_to_gcs(fig_shap, gcs_bucket, f"{gcs_output_prefix}/plots/mlp_shap_summary.png")

        mean_abs_shap = np.mean(np.abs(shap_values_np), axis=0)
        feature_importance = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
        feature_importance = feature_importance.sort_values('mean_abs_shap', ascending=False)
        logging.info("Top 10 features by Mean Absolute SHAP value:\n" + feature_importance.head(10).to_string())

        return shap_values_np, feature_importance.to_dict('records')

    except Exception as e:
        logging.error(f"Failed during SHAP analysis: {e}")
        import traceback
        logging.error(traceback.format_exc()) # Log the full traceback for better debugging
        return None, None

# --- Main Execution ---

def main(args):
    """Main training and evaluation pipeline function."""
    logging.info("--- Starting MLP Training Pipeline ---")
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

    # --- 2. Load Metadata ---
    logging.info(f"Loading metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata_str = gcs_utils.download_blob_as_string(GCS_BUCKET, metadata_blob_name)
        if metadata_str is None: raise FileNotFoundError("Could not download metadata file.")
        metadata = json.loads(metadata_str)
        logging.info("Metadata loaded successfully.")
        processed_feature_names = metadata.get("processed_feature_names")
        if not processed_feature_names: raise ValueError("Processed feature names not found in metadata.")
        data_paths = metadata.get("gcs_paths", {})
        x_train_path = data_paths.get("X_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_train_path = data_paths.get("y_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_test_path = data_paths.get("X_test_processed", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_test_path = data_paths.get("y_test", "").replace(f"gs://{GCS_BUCKET}/", "")
        if not all([x_train_path, y_train_path, x_test_path, y_test_path]):
            raise ValueError("One or more required data paths are missing in metadata.")
    except Exception as e:
        logging.error(f"Failed to load or parse metadata from {METADATA_URI}: {e}")
        return

    # --- 3. Load Processed Data ---
    X_train_df = load_data_from_gcs(GCS_BUCKET, x_train_path, processed_feature_names)
    y_train_df = load_data_from_gcs(GCS_BUCKET, y_train_path)
    X_test_df = load_data_from_gcs(GCS_BUCKET, x_test_path, processed_feature_names)
    y_test_df = load_data_from_gcs(GCS_BUCKET, y_test_path)

    # Convert to numpy arrays and then to PyTorch tensors
    X_train_np = X_train_df.values.astype(np.float32)
    y_train_np = y_train_df.iloc[:, 0].values.astype(np.int64) # Use int64 for targets
    X_test_np = X_test_df.values.astype(np.float32)
    y_test_np = y_test_df.iloc[:, 0].values.astype(np.int64)

    X_train_tensor = torch.from_numpy(X_train_np)
    y_train_tensor = torch.from_numpy(y_train_np)
    X_test_tensor = torch.from_numpy(X_test_np)
    y_test_tensor = torch.from_numpy(y_test_np)

    # --- 4. Create DataLoaders ---
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)

    # --- 5. Initialize Model, Criterion, Optimizer ---
    input_dim = X_train_tensor.shape[1]
    hidden_dims = [int(d) for d in args.mlp_hidden_dims.split(',')] # e.g., "128,64" -> [128, 64]
    model = SimpleMLP(input_dim, hidden_dims, output_dim=1, dropout_rate=args.mlp_dropout).to(device)
    logging.info(f"MLP Model Architecture:\n{model}")

    criterion = nn.BCEWithLogitsLoss() # Handles sigmoid internally
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate) # AdamW often preferred

    # --- 6. Training Loop ---
    logging.info("--- Starting Training ---")
    total_train_time = 0
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        avg_loss, duration = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        epoch_losses.append(avg_loss)
        total_train_time += duration
        # Optional: Add validation loop here if needed
    logging.info(f"--- Training Finished --- Total Duration: {total_train_time:.2f}s")

    # --- 7. Evaluate on Test Set ---
    logging.info("Evaluating model on test set...")
    test_loss, test_metrics, test_cm = evaluate_model(model, test_loader, criterion, device)

    # Retrieve probabilities needed for plotting ROC/PR curves
    all_probs_list = []
    all_targets_list = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs_logits = model(inputs)
            outputs_probs = torch.sigmoid(outputs_logits).squeeze()
            all_probs_list.append(outputs_probs.cpu().numpy())
            all_targets_list.append(targets.cpu().numpy()) # Targets are already on CPU from dataloader if pin_memory=False or after first access if True
    y_pred_proba_np = np.concatenate(all_probs_list)
    y_true_np = np.concatenate(all_targets_list)


    # --- 8. Generate and Save Plots ---
    plot_evaluation_charts(y_true_np, y_pred_proba_np, test_cm, 'MLP', GCS_BUCKET, GCS_OUTPUT_PREFIX)

    # --- 9. Perform SHAP Analysis ---
    shap_values = None
    shap_feature_importance = None
    if args.run_shap:
        shap_values, shap_feature_importance = perform_shap_analysis(
            model, X_train_tensor, X_test_tensor, processed_feature_names, device,
            GCS_BUCKET, GCS_OUTPUT_PREFIX, sample_size=args.shap_sample_size
        )
    else:
        logging.info("SHAP analysis skipped as per --run-shap flag.")

    # --- 10. Save Model ---
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/mlp_model_state_dict.pt"
    save_model_to_gcs(model.state_dict(), GCS_BUCKET, model_blob_name)

    # --- 11. Save Logs and Metrics ---
    logging.info("Saving final logs and metrics...")
    log_summary = {
        "model_type": "MLP",
        "script_args": vars(args),
        "device_used": str(device),
        "metadata_source": METADATA_URI,
        "model_architecture": str(model),
        "training_duration_seconds": total_train_time,
        "training_avg_loss_per_epoch": epoch_losses,
        "test_set_evaluation": {
             "loss": test_loss,
             "metrics": test_metrics
        },
        "shap_analysis_run": args.run_shap,
        "shap_top_features": shap_feature_importance,
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }

    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/mlp_training_log.json"
    try:
        log_string = json.dumps(log_summary, indent=4, default=lambda x: str(x)) # Handle potential non-serializable items
        gcs_utils.upload_string_to_gcs(GCS_BUCKET, log_string, log_blob_name, content_type='application/json')
        logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")
    except Exception as e:
        logging.error(f"Failed to save log summary: {e}")

    logging.info("--- MLP Training Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an MLP model using PyTorch on GPU with data from GCS.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    parser.add_argument("--metadata-uri", type=str, required=True, help="GCS URI of the preprocessing_metadata.json file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training and evaluation.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of worker processes for DataLoader.")
    # MLP Specific Hyperparameters
    parser.add_argument("--mlp-hidden-dims", type=str, default="256,128", help="Comma-separated list of hidden layer dimensions (e.g., '256,128').")
    parser.add_argument("--mlp-dropout", type=float, default=0.3, help="Dropout rate in MLP layers.")
    # SHAP arguments
    parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis.")
    parser.add_argument("--shap-sample-size", type=int, default=200, help="Number of samples for SHAP background and explanation.")

    args = parser.parse_args()
    main(args)