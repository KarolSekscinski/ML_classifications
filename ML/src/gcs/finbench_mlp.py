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
# import shap # SHAP Removed
import time
from io import BytesIO
import json
import copy
import optuna  # Added for hyperparameter tuning

import gcs_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')
MODEL_NAME = "FinBench_MLP_Tuned"
N_OPTUNA_TRIALS = 20  # Number of Optuna trials, hardcoded as per instruction


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
        logging.error(f"Error loading data from gs://{gcs_bucket}/{blob_name}: {e}");
        raise


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
        logging.error(f"ERROR saving plot to GCS ({gcs_blob_name}): {e}");
        plt.close(fig)


def save_model_to_gcs(model_state_dict, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving model state_dict to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        with BytesIO() as buf:
            torch.save(model_state_dict, buf);
            buf.seek(0)
            # Corrected: Use buf.getvalue() or buf.read()
            model_bytes = buf.read()  # or buf.getvalue()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model state_dict saved to gs://{gcs_bucket}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving model state_dict to GCS ({gcs_blob_name}): {e}")


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


def train_epoch(model, loader, criterion, optimizer, device, epoch_num, is_trial=False):
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
        if not is_trial and (i + 1) % 200 == 0:
            logging.info(f'Epoch {epoch_num}, Batch {i + 1}/{len(loader)}, Loss: {loss.item():.4f}')
    avg_loss = total_loss / len(loader)
    duration = time.time() - start_time
    if not is_trial:
        logging.info(
            f'Epoch {epoch_num} Train Summary: Avg Loss: {avg_loss:.4f}, Duration: {duration:.2f}s')
    return avg_loss, duration


def evaluate_model_torch(model, loader, criterion, device, dataset_name="Validation", is_trial=False):
    model.eval();
    total_loss = 0.0;
    all_probs_np = []
    all_targets_np = []
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
            outputs_probs = torch.sigmoid(outputs_logits).squeeze()
            metrics_collection.update(outputs_probs, targets_int)
            if not is_trial:  # Only collect these for final eval to save memory during trials
                all_probs_np.append(outputs_probs.cpu().numpy())
                all_targets_np.append(targets_int.cpu().numpy())

    avg_loss = total_loss / len(loader);
    final_metrics_tensors = metrics_collection.compute()
    final_metrics = {k: v.item() for k, v in final_metrics_tensors.items()}

    cm = None
    if not is_trial:  # Compute CM only for final evaluations
        all_probs_np = np.concatenate(all_probs_np) if len(all_probs_np) > 0 else np.array([])
        all_targets_np = np.concatenate(all_targets_np) if len(all_targets_np) > 0 else np.array([])
        all_preds_np = (all_probs_np > 0.5).astype(int) if len(all_probs_np) > 0 else np.array([])
        if len(all_targets_np) > 0 and len(all_preds_np) > 0:
            cm = confusion_matrix(all_targets_np, all_preds_np)
            final_metrics["confusion_matrix"] = cm.tolist()
        else:
            final_metrics["confusion_matrix"] = [[0, 0], [0, 0]]

        logging.info(f"--- {dataset_name} Evaluation ---")
        logging.info(f"Avg Loss: {avg_loss:.4f}")
        for name, value in final_metrics.items():
            if name != "confusion_matrix": logging.info(f"{name.replace('_', ' ').title()}: {value:.4f}")
        if cm is not None: logging.info(f"Confusion Matrix:\n{cm}")

    # For Optuna, we primarily need the loss and a key metric like pr_auc
    return avg_loss, final_metrics, cm, (all_targets_np if not is_trial else None), (
        all_probs_np if not is_trial else None)


def plot_evaluation_charts_torch(y_true_np, y_pred_proba_np, cm, plot_suffix, model_display_name, gcs_bucket,
                                 gcs_output_prefix):
    if y_true_np is None or y_pred_proba_np is None or cm is None:
        logging.warning(f"Skipping plotting for {plot_suffix} due to missing data (likely during Optuna trial).")
        return
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5));
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_display_name} CM ({plot_suffix})');
        ax_cm.set_xlabel('Predicted');
        ax_cm.set_ylabel('True')
        save_plot_to_gcs(fig_cm, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_cm_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed CM plot for {plot_suffix}: {e}")

    if y_pred_proba_np.size == 0: logging.warning(f"No probabilities for {plot_suffix}. Skipping ROC/PR."); return
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6));
        RocCurveDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_roc, name=model_display_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance');
        ax_roc.set_title(f'{model_display_name} ROC ({plot_suffix})');
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_roc_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed ROC plot for {plot_suffix}: {e}")
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6));
        PrecisionRecallDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_pr, name=model_display_name)
        ax_pr.set_title(f'{model_display_name} PR Curve ({plot_suffix})')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_pr_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed PR plot for {plot_suffix}: {e}")


# SHAP function perform_shap_analysis removed

def main(args):
    logging.info(f"--- Starting {MODEL_NAME} Training Pipeline for FinBench cf2 ---")
    GCS_BUCKET = args.gcs_bucket;
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/');
    METADATA_URI = args.metadata_uri
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

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
        logging.error(f"Failed loading/parsing OHE metadata: {e}");
        return

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
        # Batch size will be tuned by Optuna, so initial loader here is for context, will be overridden in trial
    except Exception as e:
        logging.error(f"Failed data loading or Tensor/DataLoader creation: {e}");
        return

    input_dim = X_train_tensor.shape[1]

    # --- Optuna Objective Function ---
    def objective(trial):
        # Hyperparameters to tune
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_dims_list = []
        for i in range(n_layers):
            out_features = trial.suggest_int(f"n_units_l{i}", 32, 512, log=True)
            hidden_dims_list.append(out_features)

        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        trial_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=pin_mem)
        trial_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=pin_mem)

        model = SimpleMLP(input_dim, hidden_dims_list, output_dim=1, dropout_rate=dropout_rate).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_metric_trial = -float('inf')  # Optuna maximizes, so higher is better (e.g. PR AUC)
        # If minimizing loss, use float('inf') and return loss
        epochs_no_improve_trial = 0

        # Use args.epochs for max epochs in a trial
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, trial_train_loader, criterion, optimizer, device, epoch, is_trial=True)
            val_loss_trial, val_metrics_trial, _, _, _ = evaluate_model_torch(model, trial_val_loader, criterion,
                                                                              device, dataset_name="Validation (Trial)",
                                                                              is_trial=True)

            current_val_metric = val_metrics_trial.get('pr_auc', 0.0)  # Optimize for PR AUC

            if current_val_metric > best_val_metric_trial:
                best_val_metric_trial = current_val_metric
                epochs_no_improve_trial = 0
            else:
                epochs_no_improve_trial += 1

            if epochs_no_improve_trial >= args.early_stopping_patience:
                logging.info(f"Trial {trial.number} early stopping at epoch {epoch}.")
                break

            trial.report(current_val_metric, epoch)  # Report intermediate values to Optuna
            if trial.should_prune():  # Pruning
                raise optuna.exceptions.TrialPruned()

        return best_val_metric_trial  # Optuna maximizes this

    # --- Run Optuna Study ---
    logging.info(f"--- Starting Optuna Hyperparameter Search ({N_OPTUNA_TRIALS} trials) ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best_hyperparameters = study.best_params
    best_trial_metric_value = study.best_value
    logging.info(f"Best trial PR AUC: {best_trial_metric_value:.4f}")
    logging.info(f"Best hyperparameters: {best_hyperparameters}")

    # --- Final Model Training with Best Hyperparameters ---
    logging.info("--- Training Final Model with Best Hyperparameters ---")
    final_hidden_dims = [best_hyperparameters[f"n_units_l{i}"] for i in range(best_hyperparameters["n_layers"])]
    final_model = SimpleMLP(input_dim, final_hidden_dims, output_dim=1,
                            dropout_rate=best_hyperparameters["dropout_rate"]).to(device)

    final_train_loader = DataLoader(train_dataset, batch_size=best_hyperparameters["batch_size"], shuffle=True,
                                    num_workers=args.num_workers, pin_memory=pin_mem)
    final_val_loader = DataLoader(val_dataset, batch_size=best_hyperparameters["batch_size"], shuffle=False,
                                  num_workers=args.num_workers, pin_memory=pin_mem)
    final_test_loader = DataLoader(test_dataset, batch_size=best_hyperparameters["batch_size"], shuffle=False,
                                   num_workers=args.num_workers, pin_memory=pin_mem)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=best_hyperparameters["lr"],
                            weight_decay=best_hyperparameters["weight_decay"])

    total_train_time = 0.0
    best_val_loss_final = float('inf')
    epochs_no_improve_final = 0
    best_model_state_final = None
    best_epoch_final = 0
    all_epoch_train_losses, all_epoch_val_losses, all_epoch_val_aucpr = [], [], []

    for epoch in range(1, args.epochs + 1):  # Use args.epochs for final training too
        logging.info(f"--- Final Training Epoch {epoch}/{args.epochs} ---")
        epoch_train_loss, duration = train_epoch(final_model, final_train_loader, criterion, optimizer, device, epoch)
        total_train_time += duration
        all_epoch_train_losses.append(epoch_train_loss)

        epoch_val_loss, val_metrics_epoch, _, _, _ = evaluate_model_torch(final_model, final_val_loader, criterion,
                                                                          device,
                                                                          dataset_name="Validation (Final Model)")
        all_epoch_val_losses.append(epoch_val_loss)
        all_epoch_val_aucpr.append(val_metrics_epoch.get('pr_auc', 0.0))

        if epoch_val_loss < best_val_loss_final:
            best_val_loss_final = epoch_val_loss
            best_model_state_final = copy.deepcopy(final_model.state_dict())
            best_epoch_final = epoch
            epochs_no_improve_final = 0
            logging.info(f"Final model val loss improved to {best_val_loss_final:.4f}.")
        else:
            epochs_no_improve_final += 1

        if epochs_no_improve_final >= args.early_stopping_patience:
            logging.info(f"Final model early stopping at epoch {epoch}.")
            break

    logging.info(
        f"--- Final Training Finished --- Total Duration: {total_train_time:.2f}s. Best epoch: {best_epoch_final} with val_loss: {best_val_loss_final:.4f}")
    if best_model_state_final:
        final_model.load_state_dict(best_model_state_final)
    else:
        logging.warning("No best model state saved for final model. Using last state.")

    # Evaluate on Test Set
    logging.info("Evaluating best final model on Test set...")
    _, test_metrics, test_cm, test_y_true_np, test_y_pred_proba_np = evaluate_model_torch(final_model,
                                                                                          final_test_loader, criterion,
                                                                                          device,
                                                                                          dataset_name="Test Set")
    plot_evaluation_charts_torch(test_y_true_np, test_y_pred_proba_np, test_cm, "Test_Final", MODEL_NAME, GCS_BUCKET,
                                 GCS_OUTPUT_PREFIX)

    best_val_metrics_dict_final = {}
    if best_model_state_final:
        _, best_val_metrics_dict_final, _, _, _ = evaluate_model_torch(final_model, final_val_loader, criterion, device,
                                                                       dataset_name="Best Validation Epoch (Final Model)")

    # Save Model
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/{MODEL_NAME.lower()}_best_model.pt"
    save_model_to_gcs(best_model_state_final if best_model_state_final else final_model.state_dict(), GCS_BUCKET,
                      model_blob_name)

    # Save Logs and Metrics
    logging.info("Saving final logs and metrics...")
    log_summary = {
        "model_type": MODEL_NAME, "script_args": vars(args), "device_used": str(device),
        "metadata_source": METADATA_URI,
        "optuna_n_trials": N_OPTUNA_TRIALS,
        "optuna_best_trial_metric_value_pr_auc": best_trial_metric_value,  # Or the metric optimized
        "best_hyperparameters": best_hyperparameters,
        "final_model_training_total_duration_seconds": total_train_time,
        "final_model_epoch_training_losses": all_epoch_train_losses,
        "final_model_epoch_validation_losses": all_epoch_val_losses,
        "final_model_epoch_validation_aucpr": all_epoch_val_aucpr,
        "final_model_best_epoch_for_early_stopping": best_epoch_final,
        "final_model_best_validation_loss": best_val_loss_final,
        "final_model_best_epoch_validation_metrics": best_val_metrics_dict_final,
        "final_model_test_set_evaluation": test_metrics,
        # SHAP Removed "shap_analysis_run": False, "shap_top_features": None,
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
    parser = argparse.ArgumentParser(description=f"Train and evaluate an {MODEL_NAME} model with Optuna.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    parser.add_argument("--metadata-uri", type=str, required=True,
                        help="GCS URI of the OHE preprocessing_metadata.json file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")

    # These args are now defaults or used by Optuna trials if not tuned over
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max number of training epochs per trial and for final model.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Default batch size (might be overridden by Optuna).")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Default learning rate (might be overridden by Optuna).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--mlp-hidden-dims", type=str, default="256,128,64",
                        help="Default MLP hidden layer dimensions (might be overridden).")
    parser.add_argument("--mlp-dropout", type=float, default=0.3, help="Default dropout rate (might be overridden).")

    # SHAP arguments removed
    # parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis.")
    # parser.add_argument("--shap-sample-size", type=int, default=200, help="Samples for SHAP background and explanation.")
    args = parser.parse_args()
    main(args)