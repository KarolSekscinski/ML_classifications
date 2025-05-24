# mlp_finbench_cf2.py
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
import shap
import time
from io import BytesIO
import json
import copy  # For saving best model state

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
from ML_classifications.ML.src.gcs import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')

MODEL_NAME = "FinBench_MLP"  # Used for naming outputs


# --- Helper Functions (Reused and adapted) ---
def load_data_from_gcs(gcs_bucket, gcs_path_in_metadata, feature_names_list=None):
    if not gcs_path_in_metadata or not gcs_path_in_metadata.startswith(f"gs://{gcs_bucket}/"):
        logging.error(f"Invalid GCS path from metadata: {gcs_path_in_metadata} for bucket {gcs_bucket}")
        raise ValueError(f"Invalid GCS path: {gcs_path_in_metadata}")
    blob_name = gcs_path_in_metadata.replace(f"gs://{gcs_bucket}/", "")
    logging.info(f"Loading data from: gs://{gcs_bucket}/{blob_name}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, blob_name)
        if data_bytes is None: raise FileNotFoundError(f"GCS file not downloaded: gs://{gcs_bucket}/{blob_name}")
        df = pd.read_csv(BytesIO(data_bytes))
        if feature_names_list is not None:
            if len(feature_names_list) == df.shape[1]:
                df.columns = feature_names_list
            else:
                logging.error(
                    f"Feature name count ({len(feature_names_list)}) != data columns ({df.shape[1]}) for {blob_name}. Using existing names.")
        logging.info(f"Data loaded from gs://{gcs_bucket}/{blob_name}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from gs://{gcs_bucket}/{blob_name}: {e}"); raise


def save_plot_to_gcs(fig, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving plot to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        buf = BytesIO();
        fig.savefig(buf, format='png', bbox_inches='tight');
        buf.seek(0)
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, buf.read(), gcs_blob_name, content_type='image/png')
        logging.info(f"Plot saved to gs://{gcs_bucket}/{gcs_blob_name}");
        plt.close(fig)
    except Exception as e:
        logging.error(f"ERROR saving plot to GCS ({gcs_blob_name}): {e}"); plt.close(fig)


def save_model_to_gcs(model_state_dict, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving model state_dict to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        with BytesIO() as buf:
            torch.save(model_state_dict, buf); buf.seek(0)
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, buf.read(), gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model state_dict saved to gs://{gcs_bucket}/{gcs_blob_name}")
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

    def forward(self, x): return self.network(x)


# --- Training & Evaluation Functions ---
def train_epoch(model, loader, criterion, optimizer, device, epoch_num):
    model.train();
    total_loss = 0.0;
    start_time = time.time()
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
        optimizer.zero_grad();
        outputs = model(inputs);
        loss = criterion(outputs, targets)
        loss.backward();
        optimizer.step();
        total_loss += loss.item()
        if (i + 1) % 200 == 0: logging.info(f'Epoch {epoch_num}, Batch {i + 1}/{len(loader)}, Loss: {loss.item():.4f}')
    avg_loss = total_loss / len(loader)
    duration = time.time() - start_time
    logging.info(
        f'Epoch {epoch_num} Train Summary: Avg Loss: {avg_loss:.4f}, Duration: {duration:.2f}s')
    return avg_loss, duration


def evaluate_model_torch(model, loader, criterion, device, dataset_name="Validation"):
    model.eval();
    total_loss = 0.0;
    all_preds_np = [];
    all_targets_np = [];
    all_probs_np = []
    metrics_collection = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="binary"), 'precision': torchmetrics.Precision(task="binary"),
        'recall': torchmetrics.Recall(task="binary"), 'f1': torchmetrics.F1Score(task="binary"),
        'roc_auc': torchmetrics.AUROC(task="binary"), 'pr_auc': torchmetrics.AveragePrecision(task="binary")
    }).to(device)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets_dev = inputs.to(device), targets.to(device)
            targets_float = targets_dev.float().unsqueeze(1)
            targets_int = targets_dev.int()
            outputs_logits = model(inputs);
            loss = criterion(outputs_logits, targets_float);
            total_loss += loss.item()
            outputs_probs = torch.sigmoid(outputs_logits).squeeze()  # Probabilities for P(class 1)
            metrics_collection.update(outputs_probs, targets_int)
            all_probs_np.append(outputs_probs.cpu().numpy())
            all_preds_np.append((outputs_probs > 0.5).int().cpu().numpy())
            all_targets_np.append(targets_int.cpu().numpy())
    avg_loss = total_loss / len(loader);
    final_metrics_tensors = metrics_collection.compute()
    final_metrics = {k: v.item() for k, v in final_metrics_tensors.items()}
    all_preds_np = np.concatenate(all_preds_np);
    all_targets_np = np.concatenate(all_targets_np)
    all_probs_np = np.concatenate(all_probs_np)
    cm = confusion_matrix(all_targets_np, all_preds_np)
    final_metrics["confusion_matrix"] = cm.tolist()
    logging.info(f"--- {dataset_name} Evaluation ---")
    logging.info(f"Avg Loss: {avg_loss:.4f}")
    for name, value in final_metrics.items():
        if name != "confusion_matrix": logging.info(f"{name.replace('_', ' ').title()}: {value:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    return avg_loss, final_metrics, cm, all_targets_np, all_probs_np


def plot_evaluation_charts_torch(y_true_np, y_pred_proba_np, cm, plot_suffix, model_display_name, gcs_bucket,
                                 gcs_output_prefix):
    # (Identical to svm_finbench_cf2.py's plotting, taking numpy arrays)
    try:  # CM
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5));
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_display_name} CM ({plot_suffix})');
        ax_cm.set_xlabel('Predicted');
        ax_cm.set_ylabel('True')
        save_plot_to_gcs(fig_cm, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_cm_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed CM plot for {plot_suffix}: {e}")
    if y_pred_proba_np is None: logging.warning(f"No probabilities for {plot_suffix}. Skipping ROC/PR."); return
    try:  # ROC
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6));
        RocCurveDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_roc, name=model_display_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance');
        ax_roc.set_title(f'{model_display_name} ROC ({plot_suffix})');
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_roc_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed ROC plot for {plot_suffix}: {e}")
    try:  # PR
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6));
        PrecisionRecallDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_pr, name=model_display_name)
        ax_pr.set_title(f'{model_display_name} PR Curve ({plot_suffix})')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_pr_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed PR plot for {plot_suffix}: {e}")


def perform_shap_analysis(model, X_train_tensor, X_explain_tensor, feature_names_list, device, gcs_bucket,
                          gcs_output_prefix, sample_size=100):
    # (Adapted from previous mlp_pipeline.py, using X_explain_tensor for the data to explain)
    logging.info(f"--- SHAP Analysis ({MODEL_NAME} - DeepExplainer) ---")
    logging.info(
        f"Background sample size: {min(sample_size, X_train_tensor.shape[0])}, Explain sample size: {min(sample_size, X_explain_tensor.shape[0])}")
    try:
        model.eval()
        indices_bg = np.random.choice(X_train_tensor.shape[0], min(sample_size, X_train_tensor.shape[0]), replace=False)
        X_train_summary_tensor = X_train_tensor[indices_bg].to(device)
        indices_explain = np.random.choice(X_explain_tensor.shape[0], min(sample_size, X_explain_tensor.shape[0]),
                                           replace=False)
        X_explain_sample_tensor = X_explain_tensor[indices_explain].to(device)

        start_shap_time = time.time();
        logging.info("Initializing SHAP DeepExplainer...")
        explainer = shap.DeepExplainer(model, X_train_summary_tensor)
        logging.info(f"Calculating SHAP values for {X_explain_sample_tensor.shape[0]} samples...")
        shap_values_output = explainer.shap_values(X_explain_sample_tensor)
        end_shap_time = time.time();
        logging.info(f"SHAP values calculated in {end_shap_time - start_shap_time:.2f}s.")

        shap_values_focused = None;  # SHAP values for the positive class/single output
        if isinstance(shap_values_output, list):
            if len(shap_values_output) > 0:
                shap_values_focused = shap_values_output[0]  # For single logit output model
            else:
                raise ValueError("SHAP DeepExplainer returned empty list.")
        else:
            shap_values_focused = shap_values_output  # Assume it's a single tensor/array

        if torch.is_tensor(shap_values_focused):
            shap_values_np = shap_values_focused.cpu().numpy()
        elif isinstance(shap_values_focused, np.ndarray):
            shap_values_np = shap_values_focused
        else:
            raise TypeError(f"Unexpected SHAP values type: {type(shap_values_focused)}")

        X_explain_sample_np = X_explain_sample_tensor.cpu().numpy()
        X_explain_sample_df = pd.DataFrame(X_explain_sample_np, columns=feature_names_list)

        fig_shap, _ = plt.subplots();
        shap.summary_plot(shap_values_np, X_explain_sample_df, plot_type="dot", show=False)
        plt.title(f"SHAP Summary Plot ({MODEL_NAME})");
        try:
            plt.tight_layout()
        except:
            logging.warning("Could not apply tight_layout() to SHAP plot.")
        save_plot_to_gcs(fig_shap, gcs_bucket, f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_shap_summary.png")

        mean_abs_shap = np.mean(np.abs(shap_values_np), axis=0)
        if len(feature_names_list) == len(mean_abs_shap):
            feature_importance_df = pd.DataFrame({'feature': feature_names_list, 'mean_abs_shap': mean_abs_shap})
            feature_importance_df = feature_importance_df.sort_values('mean_abs_shap', ascending=False)
            logging.info(f"Top 10 features by Mean Abs SHAP:\n{feature_importance_df.head(10).to_string()}")
        else:
            feature_importance_df = pd.DataFrame()
        return shap_values_np, feature_importance_df.to_dict('records') if not feature_importance_df.empty else None
    except Exception as e:
        logging.error(f"Failed SHAP analysis: {e}");
        import traceback;
        logging.error(traceback.format_exc())
        return None, None


# --- Main Execution ---
def main(args):
    logging.info(f"--- Starting {MODEL_NAME} Training Pipeline for FinBench cf2 ---")
    GCS_BUCKET = args.gcs_bucket;
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/');
    METADATA_URI = args.metadata_uri
    if torch.cuda.is_available():
        device = torch.device("cuda"); logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu"); logging.info("Using CPU.")

    # 1. Load Metadata
    logging.info(f"Loading OHE metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata_str = gcs_utils.download_blob_as_string(GCS_BUCKET, metadata_blob_name)
        if metadata_str is None: raise FileNotFoundError("Metadata download failed.")
        metadata = json.loads(metadata_str);
        logging.info("OHE Metadata loaded.")
        processed_feature_names = metadata.get("processed_feature_names_list")
        if not processed_feature_names: raise ValueError("'processed_feature_names_list' missing in metadata.")
        data_paths = metadata.get("gcs_paths", {})
        paths = {k: data_paths.get(k) for k in
                 ["X_train_resampled", "y_train_resampled", "X_val_processed", "y_val", "X_test_processed", "y_test"]}
        if not all(paths.values()): raise ValueError("One or more data paths missing in OHE metadata.")
    except Exception as e:
        logging.error(f"Failed loading/parsing OHE metadata: {e}"); return

    # 2. Load Data & Create Tensors/DataLoaders
    try:
        logging.info("Loading OHE data & creating Tensors/DataLoaders...")
        X_train_df = load_data_from_gcs(GCS_BUCKET, paths["X_train_resampled"], processed_feature_names)
        y_train_df = load_data_from_gcs(GCS_BUCKET, paths["y_train_resampled"])
        X_val_df = load_data_from_gcs(GCS_BUCKET, paths["X_val_processed"], processed_feature_names)
        y_val_df = load_data_from_gcs(GCS_BUCKET, paths["y_val"])
        X_test_df = load_data_from_gcs(GCS_BUCKET, paths["X_test_processed"], processed_feature_names)
        y_test_df = load_data_from_gcs(GCS_BUCKET, paths["y_test"])

        X_train_tensor = torch.from_numpy(X_train_df.values.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train_df.iloc[:, 0].values.astype(np.int64))
        X_val_tensor = torch.from_numpy(X_val_df.values.astype(np.float32))
        y_val_tensor = torch.from_numpy(y_val_df.iloc[:, 0].values.astype(np.int64))
        X_test_tensor = torch.from_numpy(X_test_df.values.astype(np.float32))
        y_test_tensor = torch.from_numpy(y_test_df.iloc[:, 0].values.astype(np.int64))

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor);
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor);
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        pin_mem = device.type == 'cuda'
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=pin_mem)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=pin_mem)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=pin_mem)
        logging.info(
            f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        logging.error(f"Failed data loading or Tensor/DataLoader creation: {e}"); return

    # 3. Initialize Model, Criterion, Optimizer
    input_dim = X_train_tensor.shape[1]
    hidden_dims_list = [int(d.strip()) for d in args.mlp_hidden_dims.split(',')]
    model = SimpleMLP(input_dim, hidden_dims_list, output_dim=1, dropout_rate=args.mlp_dropout).to(device)
    logging.info(f"{MODEL_NAME} Architecture:\n{model}")
    criterion = nn.BCEWithLogitsLoss();
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 4. Training Loop with Early Stopping
    logging.info("--- Starting Training ---");
    total_train_time = 0.0
    best_val_loss = float('inf');
    epochs_no_improve = 0;
    best_model_state = None;
    best_epoch = 0
    all_epoch_train_losses = [];
    all_epoch_val_losses = [];
    all_epoch_val_aucpr = []

    for epoch in range(1, args.epochs + 1):
        logging.info(f"--- Epoch {epoch}/{args.epochs} ---")
        epoch_train_loss, duration = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        total_train_time += duration;
        all_epoch_train_losses.append(epoch_train_loss)

        epoch_val_loss, val_metrics_epoch, _, _, _ = evaluate_model_torch(model, val_loader, criterion, device,
                                                                          dataset_name="Validation (Epoch End)")
        all_epoch_val_losses.append(epoch_val_loss)
        all_epoch_val_aucpr.append(val_metrics_epoch.get('pr_auc', 0.0))  # Store AUC-PR for logging

        if epoch_val_loss < best_val_loss:
            logging.info(
                f"Validation loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}. Saving model state.")
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())  # Use deepcopy
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(
                f"Validation loss did not improve ({epoch_val_loss:.4f} vs best {best_val_loss:.4f}). Patience: {epochs_no_improve}/{args.early_stopping_patience}")

        if epochs_no_improve >= args.early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logging.info(
        f"--- Training Finished --- Total Duration: {total_train_time:.2f}s. Best epoch: {best_epoch} with val_loss: {best_val_loss:.4f}")
    if best_model_state:
        model.load_state_dict(best_model_state)  # Load best model
    else:
        logging.warning(
            "No best model state saved (e.g. training stopped early or no improvement). Using last model state.")

    # 5. Evaluate on Test Set (with best model)
    logging.info("Evaluating best model on Test set...")
    _, test_metrics, test_cm, test_y_true_np, test_y_pred_proba_np = evaluate_model_torch(model, test_loader, criterion,
                                                                                          device,
                                                                                          dataset_name="Test Set")
    plot_evaluation_charts_torch(test_y_true_np, test_y_pred_proba_np, test_cm, "Test", MODEL_NAME, GCS_BUCKET,
                                 GCS_OUTPUT_PREFIX)

    # Retrieve validation metrics for the best epoch
    best_val_metrics_dict = {}
    if best_epoch > 0 and (best_epoch - 1) < len(all_epoch_val_losses):  # ensure best_epoch index is valid
        # Rerun evaluation on val set with best model to get full metrics dict if not stored per epoch
        _, best_val_metrics_dict, _, _, _ = evaluate_model_torch(model, val_loader, criterion, device,
                                                                 dataset_name="Best Validation Epoch")

    # 6. Perform SHAP Analysis (on Test Set, using Training Set for background)
    shap_feature_importance = None
    if args.run_shap:
        _, shap_feature_importance = perform_shap_analysis(
            model, X_train_tensor, X_test_tensor, processed_feature_names, device,
            GCS_BUCKET, GCS_OUTPUT_PREFIX, sample_size=args.shap_sample_size
        )
    else:
        logging.info("SHAP analysis skipped.")

    # 7. Save Model
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/{MODEL_NAME.lower()}_best_model.pt"
    if best_model_state:
        save_model_to_gcs(best_model_state, GCS_BUCKET, model_blob_name)
    else:
        save_model_to_gcs(model.state_dict(), GCS_BUCKET, model_blob_name)  # Save last state if no best

    # 8. Save Logs and Metrics
    logging.info("Saving final logs and metrics...")
    log_summary = {
        "model_type": MODEL_NAME, "script_args": vars(args), "device_used": str(device),
        "metadata_source": METADATA_URI,  # "model_architecture": str(model), # Can be very long
        "training_total_duration_seconds": total_train_time,
        "epoch_training_losses": all_epoch_train_losses,
        "epoch_validation_losses": all_epoch_val_losses,
        "epoch_validation_aucpr": all_epoch_val_aucpr,
        "best_epoch_for_early_stopping": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_epoch_validation_metrics": best_val_metrics_dict,  # Metrics from evaluating best model on val set
        "test_set_evaluation_with_best_model": test_metrics,
        "shap_analysis_run": args.run_shap, "shap_top_features": shap_feature_importance,
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }
    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/{MODEL_NAME.lower()}_training_log.json"
    try:
        log_string = json.dumps(log_summary, indent=4, default=lambda x: str(x) if isinstance(x, torch.device) else x)
        gcs_utils.upload_string_to_gcs(GCS_BUCKET, log_string, log_blob_name, content_type='application/json')
        logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")
    except Exception as e:
        logging.error(f"Failed to save log summary: {e}")

    logging.info(f"--- {MODEL_NAME} Training Pipeline for FinBench cf2 Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Train and evaluate an {MODEL_NAME} model for FinBench cf2 using OHE data from GCS.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    parser.add_argument("--metadata-uri", type=str, required=True,
                        help="GCS URI of the OHE preprocessing_metadata.json file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Max number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Patience for early stopping based on validation loss.")
    # MLP Specific Hyperparameters
    parser.add_argument("--mlp-hidden-dims", type=str, default="256,128,64",
                        help="Comma-separated hidden layer dimensions.")
    parser.add_argument("--mlp-dropout", type=float, default=0.3, help="Dropout rate in MLP layers.")
    # SHAP arguments
    parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis.")
    parser.add_argument("--shap-sample-size", type=int, default=200,
                        help="Samples for SHAP background and explanation.")
    args = parser.parse_args()
    main(args)