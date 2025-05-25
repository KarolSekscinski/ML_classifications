import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchmetrics
import rtdl
import matplotlib.pyplot as plt
import seaborn as sns
# import shap # SHAP Removed
import time
from io import BytesIO
import json
import copy
import traceback
import optuna  # Added for hyperparameter tuning
from sklearn.metrics import confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay

import gcs_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')
MODEL_NAME = "FinBench_FTTransformer_Tuned"
N_OPTUNA_TRIALS = 15  # Number of Optuna trials for FT-Transformer (can be computationally intensive)


def load_data_from_gcs(gcs_bucket, gcs_path_in_metadata, expected_columns_list=None):
    if gcs_path_in_metadata is None:
        if expected_columns_list is not None and not expected_columns_list:
            logging.info("No data path provided and no columns expected. Returning empty DataFrame.")
            return pd.DataFrame()
        else:
            logging.error(f"GCS path is None but columns were expected: {expected_columns_list}")
            raise ValueError(f"GCS path is None but columns were expected for features: {expected_columns_list}")

    if not gcs_path_in_metadata.startswith(f"gs://{gcs_bucket}/"):
        logging.error(f"Invalid GCS path format: {gcs_path_in_metadata} for bucket {gcs_bucket}")
        raise ValueError(f"Invalid GCS path format: {gcs_path_in_metadata}")

    blob_name = gcs_path_in_metadata.replace(f"gs://{gcs_bucket}/", "")
    logging.info(f"Loading data from: gs://{gcs_bucket}/{blob_name}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, blob_name)
        if data_bytes is None: raise FileNotFoundError(f"GCS file download failed: gs://{gcs_bucket}/{blob_name}")
        df = pd.read_csv(BytesIO(data_bytes))

        logging.info(f"Data loaded from gs://{gcs_bucket}/{blob_name}. Shape: {df.shape}")
        if expected_columns_list is not None and not df.empty:
            if list(df.columns) != expected_columns_list and len(df.columns) == len(expected_columns_list):
                logging.warning(
                    f"Column name mismatch for {blob_name} but counts match. Expected: {expected_columns_list}, Got: {list(df.columns)}. Assuming order is correct.")
                df.columns = expected_columns_list
            elif len(df.columns) != len(expected_columns_list) and expected_columns_list:
                logging.warning(
                    f"Column count mismatch for {blob_name}. Loaded: {len(df.columns)}, Expected: {len(expected_columns_list)}.")
        elif expected_columns_list and df.empty and len(expected_columns_list) > 0:
            logging.warning(f"Loaded empty DataFrame from {blob_name} but expected columns: {expected_columns_list}")
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


def save_model_to_gcs(model_state_dict_or_model, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving model/state_dict to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    buf = None
    try:
        buf = BytesIO()
        torch.save(model_state_dict_or_model, buf)
        model_bytes = buf.getvalue()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model/state_dict successfully saved to gs://{gcs_bucket}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving model/state_dict to GCS ({gcs_blob_name}): {e}")
        logging.error(traceback.format_exc())
    finally:
        if buf is not None:
            buf.close()


class TabularDataset(Dataset):
    def __init__(self, X_num_np, X_cat_np, Y_np):
        self.X_num = torch.tensor(X_num_np, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat_np, dtype=torch.int64)
        self.Y = torch.tensor(Y_np, dtype=torch.int64)
        self.has_num = X_num_np.shape[1] > 0 if X_num_np.ndim == 2 else False
        self.has_cat = X_cat_np.shape[1] > 0 if X_cat_np.ndim == 2 else False
        n_samples_y = Y_np.shape[0]
        if self.has_num and X_num_np.shape[0] != n_samples_y: raise ValueError(
            f"Mismatch samples: X_num ({X_num_np.shape[0]}) vs Y ({n_samples_y})")
        if self.has_cat and X_cat_np.shape[0] != n_samples_y: raise ValueError(
            f"Mismatch samples: X_cat ({X_cat_np.shape[0]}) vs Y ({n_samples_y})")
        num_samples_x_num = X_num_np.shape[0] if self.has_num else n_samples_y
        num_samples_x_cat = X_cat_np.shape[0] if self.has_cat else n_samples_y
        if not (num_samples_x_num == num_samples_x_cat == n_samples_y):
            if not (num_samples_x_num == 0 and num_samples_x_cat == 0 and n_samples_y == 0):
                raise ValueError(
                    f"Sample count mismatch. Y: {n_samples_y}, X_num: {num_samples_x_num if self.has_num else 'N/A'}, X_cat: {num_samples_x_cat if self.has_cat else 'N/A'}")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x_num_item = self.X_num[idx] if self.has_num else torch.empty(0, dtype=torch.float32)
        x_cat_item = self.X_cat[idx] if self.has_cat else torch.empty(0, dtype=torch.int64)
        return x_num_item, x_cat_item, self.Y[idx]


def train_epoch_ft(model, loader, criterion, optimizer, device, epoch_num, is_trial=False):
    model.train();
    total_loss = 0.0;
    start_time = time.time()
    for i, (x_num, x_cat, targets) in enumerate(loader):
        x_num_dev, x_cat_dev, targets_dev = x_num.to(device), x_cat.to(device), targets.to(device)
        targets_float = targets_dev.float().unsqueeze(1)
        optimizer.zero_grad()
        current_x_num = x_num_dev if x_num_dev.numel() > 0 and x_num_dev.shape[-1] > 0 else None
        current_x_cat = x_cat_dev if x_cat_dev.numel() > 0 and x_cat_dev.shape[-1] > 0 else None
        outputs = model(current_x_num, current_x_cat)
        loss = criterion(outputs, targets_float);
        loss.backward();
        optimizer.step()
        total_loss += loss.item()
        if not is_trial and (i + 1) % 200 == 0: logging.info(
            f'Epoch {epoch_num}, Batch {i + 1}/{len(loader)}, Loss: {loss.item():.4f}')
    avg_loss = total_loss / len(loader);
    epoch_duration = time.time() - start_time
    if not is_trial: logging.info(
        f'Epoch {epoch_num} Train Summary: Avg Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s')
    return avg_loss, epoch_duration


def evaluate_model_ft(model, loader, criterion, device, dataset_name="Validation",
                      actual_cat_cardinalities_for_debug=None, cat_feat_names_for_debug=None, is_trial=False):
    model.eval();
    total_loss = 0.0;
    all_preds_np, all_targets_np, all_probs_np = [], [], []
    metrics_collection = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="binary"), 'precision': torchmetrics.Precision(task="binary"),
        'recall': torchmetrics.Recall(task="binary"), 'f1': torchmetrics.F1Score(task="binary"),
        'roc_auc': torchmetrics.AUROC(task="binary"), 'pr_auc': torchmetrics.AveragePrecision(task="binary")
    }).to(device)

    with torch.no_grad():
        for batch_idx, (x_num, x_cat, targets) in enumerate(loader):
            x_num_dev, x_cat_dev, targets_dev = x_num.to(device), x_cat.to(device), targets.to(device)
            targets_float = targets_dev.float().unsqueeze(1);
            targets_int = targets_dev.int()

            if x_cat_dev.numel() > 0 and x_cat_dev.shape[-1] > 0 and actual_cat_cardinalities_for_debug is not None:
                x_cat_dev_cleaned = x_cat_dev.clone()
                for i in range(x_cat_dev_cleaned.shape[1]):
                    cardinality = actual_cat_cardinalities_for_debug[i]
                    max_valid_index = cardinality - 1
                    if max_valid_index < 0: max_valid_index = 0  # Handle cardinality 0 or 1 case

                    negative_mask = x_cat_dev_cleaned[:, i] < 0
                    if negative_mask.any():
                        x_cat_dev_cleaned[:, i][negative_mask] = 0
                    too_large_mask = x_cat_dev_cleaned[:, i] > max_valid_index
                    if too_large_mask.any():
                        x_cat_dev_cleaned[:, i][too_large_mask] = max_valid_index
                current_x_cat = x_cat_dev_cleaned
            else:
                current_x_cat = x_cat_dev if x_cat_dev.numel() > 0 and x_cat_dev.shape[-1] > 0 else None

            current_x_num = x_num_dev if x_num_dev.numel() > 0 and x_num_dev.shape[-1] > 0 else None
            outputs_logits = model(current_x_num, current_x_cat)
            loss = criterion(outputs_logits, targets_float);
            total_loss += loss.item()
            outputs_probs = torch.sigmoid(outputs_logits).squeeze()
            metrics_collection.update(outputs_probs, targets_int)
            if not is_trial:
                all_probs_np.append(outputs_probs.cpu().numpy());
                all_preds_np.append((outputs_probs > 0.5).int().cpu().numpy())
                all_targets_np.append(targets_int.cpu().numpy())

    avg_loss = total_loss / len(loader);
    final_metrics_tensors = metrics_collection.compute()
    final_metrics = {k: v.item() for k, v in final_metrics_tensors.items()}

    cm = None
    if not is_trial:
        all_preds_np = np.concatenate(all_preds_np) if len(all_preds_np) > 0 else np.array([])
        all_targets_np = np.concatenate(all_targets_np) if len(all_targets_np) > 0 else np.array([])
        all_probs_np = np.concatenate(all_probs_np) if len(all_probs_np) > 0 else np.array([])
        if all_targets_np.size > 0 and all_preds_np.size > 0:
            cm = confusion_matrix(all_targets_np, all_preds_np)
        else:
            cm = np.zeros((2, 2), dtype=int)
        final_metrics["confusion_matrix"] = cm.tolist()

        logging.info(f"--- {dataset_name} Evaluation ---");
        logging.info(f"Avg Loss: {avg_loss:.4f}")
        for name, metric_val in final_metrics.items():  # Renamed value to metric_val
            if name != "confusion_matrix": logging.info(f"{name.replace('_', ' ').title()}: {metric_val:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")

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


# SHAP function perform_shap_analysis_ft_kernel removed

def main(args):
    logging.info(f"--- Starting {MODEL_NAME} Training Pipeline for FinBench cf2 ---")
    GCS_BUCKET = args.gcs_bucket;
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/');
    METADATA_URI = args.metadata_uri
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Loading FT metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata_str = gcs_utils.download_blob_as_string(GCS_BUCKET, metadata_blob_name)
        if metadata_str is None: raise FileNotFoundError("Metadata download failed.")
        metadata = json.loads(metadata_str);
        logging.info("FT Metadata loaded.")
        num_feat_names = metadata.get("original_numerical_features", [])
        cat_feat_names = metadata.get("original_categorical_features", [])
        actual_cat_cardinalities = metadata.get("cat_feature_cardinalities_from_encoder")
        if cat_feat_names and actual_cat_cardinalities is None: raise ValueError(
            "'cat_feature_cardinalities_from_encoder' missing.")
        if cat_feat_names and len(cat_feat_names) != len(actual_cat_cardinalities): raise ValueError(
            "Mismatch cat_feat_names and cardinalities count.")
        data_paths = metadata.get("gcs_paths", {});
        paths = {
            "X_train_num": data_paths.get("X_train_num_scaled"), "X_train_cat": data_paths.get("X_train_cat_indices"),
            "y_train": data_paths.get("y_train_resampled"), "X_val_num": data_paths.get("X_val_num_scaled"),
            "X_val_cat": data_paths.get("X_val_cat_indices"), "y_val": data_paths.get("y_val"),
            "X_test_num": data_paths.get("X_test_num_scaled"), "X_test_cat": data_paths.get("X_test_cat_indices"),
            "y_test": data_paths.get("y_test")
        }
        required_y_paths = ["y_train", "y_val", "y_test"]
        if num_feat_names: required_y_paths.extend(["X_train_num", "X_val_num", "X_test_num"])
        if cat_feat_names: required_y_paths.extend(["X_train_cat", "X_val_cat", "X_test_cat"])
        for p_name in required_y_paths:
            if paths.get(p_name) is None and not ((p_name.endswith("_num") and not num_feat_names) or (
                    p_name.endswith("_cat") and not cat_feat_names)):
                raise ValueError(f"Path for {p_name} is None/missing in metadata.")
            elif paths.get(p_name) == "":
                raise ValueError(f"Required data path for '{p_name}' is an empty string.")
    except Exception as e:
        logging.error(f"Failed loading/parsing FT metadata: {e}\n{traceback.format_exc()}"); return

    try:
        logging.info("Loading FT data & creating Tensors/DataLoaders...")
        X_train_num_df = load_data_from_gcs(GCS_BUCKET, paths["X_train_num"],
                                            num_feat_names) if num_feat_names else pd.DataFrame()
        X_val_num_df = load_data_from_gcs(GCS_BUCKET, paths["X_val_num"],
                                          num_feat_names) if num_feat_names else pd.DataFrame()
        X_test_num_df = load_data_from_gcs(GCS_BUCKET, paths["X_test_num"],
                                           num_feat_names) if num_feat_names else pd.DataFrame()
        X_train_cat_df = load_data_from_gcs(GCS_BUCKET, paths["X_train_cat"],
                                            cat_feat_names) if cat_feat_names else pd.DataFrame()
        X_val_cat_df = load_data_from_gcs(GCS_BUCKET, paths["X_val_cat"],
                                          cat_feat_names) if cat_feat_names else pd.DataFrame()
        X_test_cat_df = load_data_from_gcs(GCS_BUCKET, paths["X_test_cat"],
                                           cat_feat_names) if cat_feat_names else pd.DataFrame()
        y_train_df = load_data_from_gcs(GCS_BUCKET, paths["y_train"]);
        y_val_df = load_data_from_gcs(GCS_BUCKET, paths["y_val"]);
        y_test_df = load_data_from_gcs(GCS_BUCKET, paths["y_test"])
        if y_train_df is None or y_val_df is None or y_test_df is None: logging.error(
            "Target data failed to load."); return

        X_train_num_np = X_train_num_df.values.astype(np.float32) if not X_train_num_df.empty else np.empty(
            (len(y_train_df), 0), dtype=np.float32)
        X_train_cat_np = X_train_cat_df.values.astype(np.int64) if not X_train_cat_df.empty else np.empty(
            (len(y_train_df), 0), dtype=np.int64)
        y_train_np = y_train_df.iloc[:, 0].values.astype(np.int64)
        X_val_num_np = X_val_num_df.values.astype(np.float32) if not X_val_num_df.empty else np.empty(
            (len(y_val_df), 0), dtype=np.float32)
        X_val_cat_np = X_val_cat_df.values.astype(np.int64) if not X_val_cat_df.empty else np.empty((len(y_val_df), 0),
                                                                                                    dtype=np.int64)
        y_val_np = y_val_df.iloc[:, 0].values.astype(np.int64)
        X_test_num_np = X_test_num_df.values.astype(np.float32) if not X_test_num_df.empty else np.empty(
            (len(y_test_df), 0), dtype=np.float32)
        X_test_cat_np = X_test_cat_df.values.astype(np.int64) if not X_test_cat_df.empty else np.empty(
            (len(y_test_df), 0), dtype=np.int64)
        y_test_np = y_test_df.iloc[:, 0].values.astype(np.int64)

        train_dataset = TabularDataset(X_train_num_np, X_train_cat_np, y_train_np)
        val_dataset = TabularDataset(X_val_num_np, X_val_cat_np, y_val_np)
        test_dataset = TabularDataset(X_test_num_np, X_test_cat_np, y_test_np)
        pin_mem = device.type == 'cuda'
    except Exception as e:
        logging.error(f"Failed FT data loading or Tensor/DataLoader creation: {e}\n{traceback.format_exc()}"); return

    n_actual_num_features = X_train_num_np.shape[1]
    current_cat_cardinalities = actual_cat_cardinalities if cat_feat_names else []
    if not current_cat_cardinalities and X_train_cat_np.shape[1] > 0:  # Should not happen if metadata is correct
        logging.warning("Categorical features present but no cardinalities found in metadata. This is problematic.")
        # Fallback: try to infer, but this is risky
        # current_cat_cardinalities = [int(X_train_cat_np[:, i].max() + 1) for i in range(X_train_cat_np.shape[1])]
        # logging.warning(f"Inferred cardinalities (use with caution): {current_cat_cardinalities}")

    # --- Optuna Objective Function ---
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay_val = trial.suggest_float("weight_decay", 1e-6, 1e-3,
                                               log=True)  # Renamed from weight_decay to avoid conflict
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        ft_d_token_val = trial.suggest_categorical("ft_d_token",
                                                   [64, 128, 192, 256])  # Ensure multiple of n_heads (default 8)
        ft_n_blocks_val = trial.suggest_int("ft_n_blocks", 1, 4)  # Renamed from ft_n_blocks
        ft_attention_dropout_val = trial.suggest_float("ft_attention_dropout", 0.0, 0.3)
        ft_ffn_dropout_val = trial.suggest_float("ft_ffn_dropout", 0.0, 0.3)
        ft_residual_dropout_val = trial.suggest_float("ft_residual_dropout", 0.0, 0.2)
        # ft_ffn_factor_val = trial.suggest_float("ft_ffn_factor", 1.0, 2.0) # Renamed from ft_ffn_factor

        # Ensure d_token is multiple of default n_heads=8
        if ft_d_token_val % 8 != 0:
            ft_d_token_val = (ft_d_token_val // 8) * 8
            if ft_d_token_val == 0 and 192 // 8 * 8 > 0:
                ft_d_token_val = 192 // 8 * 8  # Ensure it's not 0, use default if calc leads to 0
            elif ft_d_token_val == 0:
                ft_d_token_val = 8  # Smallest possible multiple

        calculated_ffn_d_hidden = int(
            ft_d_token_val * (4 / 3))  # Using default ffn_factor for simplicity or trial.suggest for it

        trial_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=pin_mem)
        trial_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=pin_mem)

        try:
            model = rtdl.FTTransformer.make_baseline(
                n_num_features=n_actual_num_features,
                cat_cardinalities=current_cat_cardinalities if current_cat_cardinalities else None,
                d_token=ft_d_token_val, n_blocks=ft_n_blocks_val,
                attention_dropout=ft_attention_dropout_val, ffn_d_hidden=calculated_ffn_d_hidden,
                ffn_dropout=ft_ffn_dropout_val, residual_dropout=ft_residual_dropout_val, d_out=1
            ).to(device)
        except Exception as model_init_e:
            logging.error(f"Trial {trial.number} failed model initialization: {model_init_e}")
            raise optuna.exceptions.TrialPruned()  # Prune if model cannot be initialized

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_val)

        best_val_metric_trial = -float('inf')
        epochs_no_improve_trial = 0

        for epoch in range(1, args.epochs + 1):  # Using args.epochs for max epochs in a trial
            train_epoch_ft(model, trial_train_loader, criterion, optimizer, device, epoch, is_trial=True)
            val_loss_trial, val_metrics_trial, _, _, _ = evaluate_model_ft(
                model, trial_val_loader, criterion, device, "Validation (Trial)",
                actual_cat_cardinalities, cat_feat_names, is_trial=True
            )
            current_val_metric = val_metrics_trial.get('pr_auc', 0.0)  # Optimize for PR AUC

            if current_val_metric > best_val_metric_trial:
                best_val_metric_trial = current_val_metric
                epochs_no_improve_trial = 0
            else:
                epochs_no_improve_trial += 1

            if epochs_no_improve_trial >= args.early_stopping_patience:
                logging.info(f"Trial {trial.number} early stopping at epoch {epoch}.")
                break
            trial.report(current_val_metric, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return best_val_metric_trial

    logging.info(f"--- Starting Optuna Hyperparameter Search ({N_OPTUNA_TRIALS} trials) ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best_hyperparameters = study.best_params
    best_trial_metric_value = study.best_value
    logging.info(f"Best trial PR AUC: {best_trial_metric_value:.4f}")
    logging.info(f"Best hyperparameters: {best_hyperparameters}")

    # --- Final Model Training ---
    logging.info("--- Training Final Model with Best Hyperparameters ---")
    final_d_token = best_hyperparameters["ft_d_token"]
    if final_d_token % 8 != 0:  # Re-ensure multiple of 8 for safety
        final_d_token = (final_d_token // 8) * 8
        if final_d_token == 0 and 192 // 8 * 8 > 0:
            final_d_token = 192 // 8 * 8
        elif final_d_token == 0:
            final_d_token = 8

    final_calculated_ffn_d_hidden = int(final_d_token * (4 / 3))  # Using default ffn_factor

    final_model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_actual_num_features,
        cat_cardinalities=current_cat_cardinalities if current_cat_cardinalities else None,
        d_token=final_d_token, n_blocks=best_hyperparameters["ft_n_blocks"],
        attention_dropout=best_hyperparameters["ft_attention_dropout"], ffn_d_hidden=final_calculated_ffn_d_hidden,
        ffn_dropout=best_hyperparameters["ft_ffn_dropout"],
        residual_dropout=best_hyperparameters["ft_residual_dropout"], d_out=1
    ).to(device)

    final_train_loader = DataLoader(train_dataset, batch_size=best_hyperparameters["batch_size"], shuffle=True,
                                    num_workers=args.num_workers, pin_memory=pin_mem)
    final_val_loader = DataLoader(val_dataset, batch_size=best_hyperparameters["batch_size"], shuffle=False,
                                  num_workers=args.num_workers, pin_memory=pin_mem)
    final_test_loader = DataLoader(test_dataset, batch_size=best_hyperparameters["batch_size"], shuffle=False,
                                   num_workers=args.num_workers, pin_memory=pin_mem)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=best_hyperparameters["lr"],
                            weight_decay=best_hyperparameters["weight_decay"])

    total_train_time = 0.0;
    best_val_loss_final = float('inf');
    epochs_no_improve_final = 0;
    best_model_state_final = None;
    best_epoch_final = 0
    all_epoch_train_losses, all_epoch_val_losses, all_epoch_val_aucpr = [], [], []

    for epoch in range(1, args.epochs + 1):
        logging.info(f"--- Final Training Epoch {epoch}/{args.epochs} ---")
        epoch_train_loss, duration = train_epoch_ft(final_model, final_train_loader, criterion, optimizer, device,
                                                    epoch)
        total_train_time += duration;
        all_epoch_train_losses.append(epoch_train_loss)
        epoch_val_loss, val_metrics_epoch, _, _, _ = evaluate_model_ft(
            final_model, final_val_loader, criterion, device, "Validation (Final Model)", actual_cat_cardinalities,
            cat_feat_names
        )
        all_epoch_val_losses.append(epoch_val_loss);
        all_epoch_val_aucpr.append(val_metrics_epoch.get('pr_auc', 0.0))
        if epoch_val_loss < best_val_loss_final:
            best_val_loss_final = epoch_val_loss;
            best_model_state_final = copy.deepcopy(final_model.state_dict());
            best_epoch_final = epoch;
            epochs_no_improve_final = 0
            logging.info(f"Final model val loss improved to {best_val_loss_final:.4f}.")
        else:
            epochs_no_improve_final += 1
        if epochs_no_improve_final >= args.early_stopping_patience: logging.info(
            f"Final model early stopping at epoch {epoch}."); break

    logging.info(
        f"--- Final Training Finished --- Duration: {total_train_time:.2f}s. Best epoch: {best_epoch_final}, val_loss: {best_val_loss_final:.4f}")
    if best_model_state_final:
        final_model.load_state_dict(best_model_state_final)
    else:
        logging.warning("No best model state for final model.")

    logging.info("Evaluating best final model on Test set...")
    _, test_metrics, test_cm, test_y_true_np, test_y_pred_proba_np = evaluate_model_ft(
        final_model, final_test_loader, criterion, device, "Test Set", actual_cat_cardinalities, cat_feat_names
    )
    plot_evaluation_charts_torch(test_y_true_np, test_y_pred_proba_np, test_cm, "Test_Final", MODEL_NAME, GCS_BUCKET,
                                 GCS_OUTPUT_PREFIX)

    best_val_metrics_dict_final = {}
    if best_model_state_final:
        _, best_val_metrics_dict_final, _, _, _ = evaluate_model_ft(final_model, final_val_loader, criterion, device,
                                                                    "Best Validation Epoch (Final Model)",
                                                                    actual_cat_cardinalities, cat_feat_names)

    # SHAP analysis removed

    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/{MODEL_NAME.lower()}_best_model.pt"
    save_model_to_gcs(best_model_state_final if best_model_state_final else final_model.state_dict(), GCS_BUCKET,
                      model_blob_name)

    logging.info("Saving final logs and metrics...")
    log_summary = {
        "model_type": MODEL_NAME, "script_args": vars(args), "device_used": str(device),
        "metadata_source": METADATA_URI, "n_numerical_features": n_actual_num_features,
        "categorical_cardinalities": current_cat_cardinalities,
        "optuna_n_trials": N_OPTUNA_TRIALS,
        "optuna_best_trial_metric_value_pr_auc": best_trial_metric_value,
        "best_hyperparameters": best_hyperparameters,
        "final_model_training_total_duration_seconds": total_train_time,
        "final_model_epoch_training_losses": all_epoch_train_losses,
        "final_model_epoch_validation_losses": all_epoch_val_losses,
        "final_model_epoch_validation_aucpr": all_epoch_val_aucpr,
        "final_model_best_epoch_for_early_stopping": best_epoch_final,
        "final_model_best_validation_loss": best_val_loss_final,
        "final_model_best_epoch_validation_metrics": best_val_metrics_dict_final,
        "final_model_test_set_evaluation": test_metrics,
        # SHAP Removed "shap_analysis_run": False, "shap_top_numerical_features": None,
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
                        help="GCS URI of the FT preprocessing_ft_metadata.json file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")

    parser.add_argument("--epochs", type=int, default=30,
                        help="Max number of training epochs per trial and for final model.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Default batch size (might be overridden by Optuna).")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Default learning rate (might be overridden).")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Default AdamW weight decay (might be overridden).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping.")

    parser.add_argument("--ft-d-token", type=int, default=192, help="Default token dimension (might be overridden).")
    parser.add_argument("--ft-n-blocks", type=int, default=3,
                        help="Default number of transformer blocks (might be overridden).")
    parser.add_argument("--ft-attention-dropout", type=float, default=0.2,
                        help="Default attention dropout (might be overridden).")
    parser.add_argument("--ft-ffn-dropout", type=float, default=0.1, help="Default FFN dropout (might be overridden).")
    parser.add_argument("--ft-residual-dropout", type=float, default=0.0,
                        help="Default residual dropout (might be overridden).")
    parser.add_argument("--ft-ffn-factor", type=float, default=4 / 3,
                        help="Default multiplier for FFN hidden dim relative to d_token (currently not tuned, uses 4/3).")

    # SHAP arguments removed
    # parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis.")
    # parser.add_argument("--shap-background-sample-size", type=int, default=50, help="Background samples for KernelSHAP masker.")
    # parser.add_argument("--shap-explain-sample-size", type=int, default=10, help="Instances to explain with KernelSHAP.")
    args = parser.parse_args()
    main(args)