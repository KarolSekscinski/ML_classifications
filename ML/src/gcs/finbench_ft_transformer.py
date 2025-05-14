# ft_transformer_finbench_cf2.py
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
import shap
import time
from io import BytesIO, StringIO
import json
import os
import copy

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')

MODEL_NAME = "FinBench_FTTransformer"  # Used for naming outputs


# --- Helper Functions (Reused and adapted) ---
def load_data_from_gcs(gcs_bucket, gcs_path_in_metadata, expected_columns=None):
    if not gcs_path_in_metadata or not gcs_path_in_metadata.startswith(f"gs://{gcs_bucket}/"):
        # Handle cases where a feature type might be absent (e.g. no numerical features)
        if gcs_path_in_metadata is None and expected_columns is not None and not expected_columns:  # No columns expected
            logging.info(
                f"No data path provided and no columns expected (e.g. no numerical features). Returning empty DataFrame.")
            return pd.DataFrame()
        logging.error(f"Invalid GCS path from metadata: {gcs_path_in_metadata} for bucket {gcs_bucket}")
        raise ValueError(f"Invalid GCS path: {gcs_path_in_metadata}")

    blob_name = gcs_path_in_metadata.replace(f"gs://{gcs_bucket}/", "")
    logging.info(f"Loading data from: gs://{gcs_bucket}/{blob_name}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, blob_name)
        if data_bytes is None: raise FileNotFoundError(f"GCS file download failed: gs://{gcs_bucket}/{blob_name}")
        df = pd.read_csv(BytesIO(data_bytes))
        # No feature_names_list to assign here, as columns should be correct from preprocessing
        logging.info(f"Data loaded from gs://{gcs_bucket}/{blob_name}. Shape: {df.shape}")
        if expected_columns is not None and df.shape[1] > 0:  # Check if df has columns
            if list(df.columns) != expected_columns:
                logging.warning(
                    f"Column mismatch for {blob_name}. Expected: {expected_columns}, Got: {list(df.columns)}. This might be okay if order differs but names are present for selection later.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from gs://{gcs_bucket}/{blob_name}: {e}"); raise


def save_plot_to_gcs(fig, gcs_bucket, gcs_blob_name):
    # (Identical to previous versions)
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
    # (Identical to previous versions)
    logging.info(f"Saving model state_dict to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        with BytesIO() as buf:
            torch.save(model_state_dict, buf); buf.seek(0)
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, buf.read(), gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model state_dict saved to gs://{gcs_bucket}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving model state_dict to GCS ({gcs_blob_name}): {e}")


# --- PyTorch Dataset Definition for FT-Transformer ---
class TabularDataset(Dataset):
    def __init__(self, X_num_np, X_cat_np, Y_np):
        self.X_num = torch.tensor(X_num_np, dtype=torch.float32)
        # Ensure X_cat is int64 for embedding layers
        self.X_cat = torch.tensor(X_cat_np, dtype=torch.int64)
        self.Y = torch.tensor(Y_np, dtype=torch.int64)  # Targets for metrics
        # Check if any feature set is empty
        self.has_num = X_num_np.shape[1] > 0
        self.has_cat = X_cat_np.shape[1] > 0

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # Return empty tensors if features are not present, model's forward needs to handle this
        x_num_item = self.X_num[idx] if self.has_num else torch.empty(0, dtype=torch.float32)
        x_cat_item = self.X_cat[idx] if self.has_cat else torch.empty(0, dtype=torch.int64)
        return x_num_item, x_cat_item, self.Y[idx]


# --- Training & Evaluation Functions (Adapted for FT-Transformer input) ---
def train_epoch_ft(model, loader, criterion, optimizer, device, epoch_num):
    model.train();
    total_loss = 0.0;
    start_time = time.time()
    for i, (x_num, x_cat, targets) in enumerate(loader):
        x_num, x_cat, targets_dev = x_num.to(device), x_cat.to(device), targets.to(device)
        targets_float = targets_dev.float().unsqueeze(1)
        optimizer.zero_grad()
        # Pass None if feature type is absent; model's forward in rtdl handles this
        current_x_num = x_num if x_num.shape[-1] > 0 else None
        current_x_cat = x_cat if x_cat.shape[-1] > 0 else None
        outputs = model(current_x_num, current_x_cat)
        loss = criterion(outputs, targets_float);
        loss.backward();
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 200 == 0: logging.info(f'Epoch {epoch_num}, Batch {i + 1}/{len(loader)}, Loss: {loss.item():.4f}')
    avg_loss = total_loss / len(loader)
    epoch_duration = time.time() - start_time  # Calculate duration
    logging.info(f'Epoch {epoch_num} Train Summary: Avg Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s')
    return avg_loss, epoch_duration


def evaluate_model_ft(model, loader, criterion, device, dataset_name="Validation"):
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
        for x_num, x_cat, targets in loader:
            x_num, x_cat, targets_dev = x_num.to(device), x_cat.to(device), targets.to(device)
            targets_float = targets_dev.float().unsqueeze(1);
            targets_int = targets_dev.int()
            current_x_num = x_num if x_num.shape[-1] > 0 else None
            current_x_cat = x_cat if x_cat.shape[-1] > 0 else None
            outputs_logits = model(current_x_num, current_x_cat)
            loss = criterion(outputs_logits, targets_float);
            total_loss += loss.item()
            outputs_probs = torch.sigmoid(outputs_logits).squeeze()
            metrics_collection.update(outputs_probs, targets_int)
            all_probs_np.append(outputs_probs.cpu().numpy());
            all_preds_np.append((outputs_probs > 0.5).int().cpu().numpy())
            all_targets_np.append(targets_int.cpu().numpy())
    avg_loss = total_loss / len(loader);
    final_metrics_tensors = metrics_collection.compute()
    final_metrics = {k: v.item() for k, v in final_metrics_tensors.items()}
    all_preds_np = np.concatenate(all_preds_np);
    all_targets_np = np.concatenate(all_targets_np);
    all_probs_np = np.concatenate(all_probs_np)
    cm = confusion_matrix(all_targets_np, all_preds_np)
    final_metrics["confusion_matrix"] = cm.tolist()
    logging.info(f"--- {dataset_name} Evaluation ---");
    logging.info(f"Avg Loss: {avg_loss:.4f}")
    for name, value in final_metrics.items():
        if name != "confusion_matrix": logging.info(f"{name.replace('_', ' ').title()}: {value:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    return avg_loss, final_metrics, cm, all_targets_np, all_probs_np


# plot_evaluation_charts_torch (Identical to mlp_finbench_cf2.py)
def plot_evaluation_charts_torch(y_true_np, y_pred_proba_np, cm, plot_suffix, model_display_name, gcs_bucket,
                                 gcs_output_prefix):
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


# SHAP using KernelExplainer for FT-Transformer's numerical features
def perform_shap_analysis_ft_kernel(model, train_loader, test_loader_num_only,
                                    # test_loader_num_only provides only X_num_np for explanation
                                    numerical_feature_names, fixed_cat_background_tensor,
                                    # Fixed categorical tensor from training data
                                    device, gcs_bucket, gcs_output_prefix,
                                    background_sample_size=50, explain_sample_size=10):
    logging.warning(f"--- SHAP Analysis ({MODEL_NAME} - KernelExplainer on Numerical Features) ---")
    logging.warning(
        f"KernelExplainer is VERY SLOW. Background: {background_sample_size}, Explain: {explain_sample_size} samples.")
    try:
        model.eval()
        # 1. Prepare Background Data for KernelExplainer (Numerical Part)
        bg_num_list = []
        count = 0
        for x_num, _, _ in train_loader:  # Iterate to collect numerical training samples
            batch_size = x_num.shape[0];
            needed = background_sample_size - count;
            take = min(needed, batch_size)
            if needed <= 0: break
            bg_num_list.append(x_num[:take].cpu().numpy());
            count += take  # Store as numpy
        if not bg_num_list: raise ValueError("Could not get numerical background samples for SHAP masker.")
        background_num_masker_np = np.concatenate(bg_num_list)
        # Summarize if too large
        if background_num_masker_np.shape[0] > 2 * background_sample_size:  # Heuristic
            background_masker_data = shap.kmeans(background_num_masker_np, min(50, background_num_masker_np.shape[0]))
            logging.info(f"Summarized numerical background data for masker to shape {background_masker_data.shape}")
        else:
            background_masker_data = background_num_masker_np

        # 2. Prepare Data to Explain (Numerical Part from test_loader_num_only)
        explain_num_list = []
        count = 0
        for x_num_batch_np in test_loader_num_only:  # test_loader_num_only yields batches of X_num_np directly
            current_batch_size = x_num_batch_np.shape[0];
            needed = explain_sample_size - count;
            take = min(needed, current_batch_size)
            if needed <= 0: break
            explain_num_list.append(x_num_batch_np[:take]);
            count += take
        if not explain_num_list: raise ValueError("Could not get numerical samples to explain for SHAP.")
        test_sample_num_np = np.concatenate(explain_num_list)
        logging.info(f"Data to explain (numerical part) shape: {test_sample_num_np.shape}")

        # 3. Define Prediction Function for KernelExplainer
        # It takes a batch of numerical features (numpy) and uses a fixed categorical tensor.
        # Use a single representative fixed categorical vector for all predictions to simplify.
        if fixed_cat_background_tensor is not None and fixed_cat_background_tensor.shape[0] > 0:
            single_fixed_cat_vector = fixed_cat_background_tensor[0:1, :].to(device)  # Shape [1, n_cat_features]
        else:  # No categorical features
            single_fixed_cat_vector = None

        def predict_fn_for_kernel(x_num_np_batch):
            x_num_torch = torch.tensor(x_num_np_batch, dtype=torch.float32).to(device)
            current_batch_size = x_num_torch.shape[0]

            cat_input_for_model = None
            if single_fixed_cat_vector is not None:
                cat_input_for_model = single_fixed_cat_vector.repeat(current_batch_size, 1)

            num_input_for_model = x_num_torch if x_num_torch.shape[1] > 0 else None

            with torch.no_grad():
                logits = model(num_input_for_model, cat_input_for_model)
                probs = torch.sigmoid(logits)
            # KernelExplainer usually expects probabilities for each class if multi-output
            # For binary, returning P(class 1) and P(class 0) is common.
            # Or if it expects single output for shap_values then just P(class 1)
            probs_for_shap = torch.cat((1 - probs, probs), dim=1)  # [P(class 0), P(class 1)]
            return probs_for_shap.cpu().numpy()

        # 4. Initialize and Run KernelExplainer
        start_shap_time = time.time();
        logging.info("Initializing SHAP KernelExplainer...")
        explainer = shap.KernelExplainer(predict_fn_for_kernel, background_masker_data)
        logging.info(f"Calculating SHAP values for numerical features on {test_sample_num_np.shape[0]} samples...")
        # nsamples needs to be carefully chosen for KernelExplainer for performance.
        # 'auto' can be very large. For N features, 2*N+2048 is a rule of thumb for full coverage.
        # For fewer features, a smaller number might be okay. Start small.
        num_shap_samples = min(2 * test_sample_num_np.shape[1] + 512, 200)  # Heuristic for nsamples
        if test_sample_num_np.shape[1] == 0:  # No numerical features
            logging.warning("No numerical features to explain with SHAP. Skipping SHAP plot.")
            return None, None

        shap_values_output = explainer.shap_values(test_sample_num_np, nsamples=num_shap_samples)
        end_shap_time = time.time();
        logging.info(f"SHAP values (numerical) calculated in {end_shap_time - start_shap_time:.2f}s.")

        # KernelExplainer with predict_fn returning 2 columns (for binary) will give a list of 2 SHAP arrays
        shap_values_num_np = shap_values_output[1] if isinstance(shap_values_output, list) and len(
            shap_values_output) == 2 else shap_values_output

        # 5. Plot and Log
        fig_shap, _ = plt.subplots();
        shap.summary_plot(shap_values_num_np, test_sample_num_np, feature_names=numerical_feature_names,
                          plot_type="dot", show=False)
        plt.title(f"SHAP Summary Plot ({MODEL_NAME} - Numerical Features - Kernel)");
        try:
            plt.tight_layout()
        except:
            logging.warning("Could not apply tight_layout() to SHAP plot.")
        save_plot_to_gcs(fig_shap, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_shap_summary_numerical_kernel.png")

        mean_abs_shap_num = np.mean(np.abs(shap_values_num_np), axis=0)
        if len(numerical_feature_names) == len(mean_abs_shap_num):
            feature_importance_df = pd.DataFrame(
                {'feature': numerical_feature_names, 'mean_abs_shap': mean_abs_shap_num})
            feature_importance_df = feature_importance_df.sort_values('mean_abs_shap', ascending=False)
            logging.info(
                f"Top Numerical features by Mean Abs SHAP (Kernel):\n{feature_importance_df.head(10).to_string()}")
        else:
            feature_importance_df = pd.DataFrame()
        logging.warning(
            "SHAP for categorical features with KernelExplainer on FT-Transformer is complex and not implemented here.")
        return shap_values_num_np, feature_importance_df.to_dict('records') if not feature_importance_df.empty else None
    except Exception as e:
        logging.error(f"Failed KernelExplainer SHAP analysis: {e}");
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

    # 1. Load FT-Specific Metadata
    logging.info(f"Loading FT metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata_str = gcs_utils.download_blob_as_string(GCS_BUCKET, metadata_blob_name)
        if metadata_str is None: raise FileNotFoundError("Metadata download failed.")
        metadata = json.loads(metadata_str);
        logging.info("FT Metadata loaded.")

        num_feat_names = metadata.get("original_numerical_features", [])
        cat_feat_names = metadata.get("original_categorical_features", [])
        cat_cardinalities = metadata.get("cat_feature_cardinalities_from_encoder")  # Use encoder's cardinalities
        if cat_feat_names and not cat_cardinalities: raise ValueError(
            "'cat_feature_cardinalities_from_encoder' missing in metadata when categorical features exist.")
        if cat_feat_names and len(cat_feat_names) != len(cat_cardinalities): raise ValueError(
            "Mismatch between cat feature names and cardinalities count.")

        data_paths = metadata.get("gcs_paths", {})
        paths = {
            "X_train_num": data_paths.get("X_train_num_scaled"), "X_train_cat": data_paths.get("X_train_cat_indices"),
            "y_train": data_paths.get("y_train_resampled"),
            "X_val_num": data_paths.get("X_val_num_scaled"), "X_val_cat": data_paths.get("X_val_cat_indices"),
            "y_val": data_paths.get("y_val"),
            "X_test_num": data_paths.get("X_test_num_scaled"), "X_test_cat": data_paths.get("X_test_cat_indices"),
            "y_test": data_paths.get("y_test")
        }
        # Check paths based on feature presence
        required_y_paths = ["y_train", "y_val", "y_test"]
        if num_feat_names: required_y_paths.extend(["X_train_num", "X_val_num", "X_test_num"])
        if cat_feat_names: required_y_paths.extend(["X_train_cat", "X_val_cat", "X_test_cat"])
        if not all(paths.get(p) for p in required_y_paths):
            missing_p = [p for p in required_y_paths if not paths.get(p)]
            raise ValueError(f"Required data paths missing in FT metadata: {missing_p}")

    except Exception as e:
        logging.error(f"Failed loading/parsing FT metadata: {e}"); return

    # 2. Load Data & Create Tensors/DataLoaders
    try:
        logging.info("Loading FT data & creating Tensors/DataLoaders...")
        # Load numerical data if present
        X_train_num_df = load_data_from_gcs(GCS_BUCKET, paths["X_train_num"],
                                            num_feat_names) if num_feat_names else pd.DataFrame()
        X_val_num_df = load_data_from_gcs(GCS_BUCKET, paths["X_val_num"],
                                          num_feat_names) if num_feat_names else pd.DataFrame()
        X_test_num_df = load_data_from_gcs(GCS_BUCKET, paths["X_test_num"],
                                           num_feat_names) if num_feat_names else pd.DataFrame()
        # Load categorical data if present
        X_train_cat_df = load_data_from_gcs(GCS_BUCKET, paths["X_train_cat"],
                                            cat_feat_names) if cat_feat_names else pd.DataFrame()
        X_val_cat_df = load_data_from_gcs(GCS_BUCKET, paths["X_val_cat"],
                                          cat_feat_names) if cat_feat_names else pd.DataFrame()
        X_test_cat_df = load_data_from_gcs(GCS_BUCKET, paths["X_test_cat"],
                                           cat_feat_names) if cat_feat_names else pd.DataFrame()
        # Load target variables
        y_train_df = load_data_from_gcs(GCS_BUCKET, paths["y_train"]);
        y_val_df = load_data_from_gcs(GCS_BUCKET, paths["y_val"]);
        y_test_df = load_data_from_gcs(GCS_BUCKET, paths["y_test"])

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
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=pin_mem)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=pin_mem)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=pin_mem)
        logging.info(
            f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    except Exception as e:
        logging.error(f"Failed FT data loading or Tensor/DataLoader creation: {e}"); import traceback; logging.error(
            traceback.format_exc()); return

    # 3. Initialize Model, Criterion, Optimizer
    n_actual_num_features = X_train_num_np.shape[1]
    # Use cat_cardinalities from metadata (derived from OrdinalEncoder in preprocess_ft)
    actual_cat_cardinalities = cat_cardinalities if cat_feat_names else []

    # Ensure arguments for FTTransformer.make_baseline are correct based on your rtdl version
    # This instantiation uses the API from rtdl > 0.0.13 (e.g., 0.4.1) as discussed
    calculated_ffn_d_hidden = int(args.ft_d_token * args.ft_ffn_factor)
    logging.info(
        f"Initializing FTTransformer using make_baseline. n_num_features={n_actual_num_features}, cat_cardinalities={actual_cat_cardinalities}, d_token={args.ft_d_token}, n_blocks={args.ft_n_blocks}, ffn_d_hidden={calculated_ffn_d_hidden}")
    try:
        model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_actual_num_features,
            cat_cardinalities=actual_cat_cardinalities if actual_cat_cardinalities else None,
            d_token=args.ft_d_token, n_blocks=args.ft_n_blocks,
            attention_dropout=args.ft_attention_dropout, ffn_d_hidden=calculated_ffn_d_hidden,
            ffn_dropout=args.ft_ffn_dropout, residual_dropout=args.ft_residual_dropout,
            d_out=1
        ).to(device)
        logging.info(f"{MODEL_NAME} Architecture (rtdl.FTTransformer.make_baseline used).")
    except Exception as e:
        logging.error(f"Failed to initialize FTTransformer: {e}. Check rtdl version and parameters.");
        import traceback;
        logging.error(traceback.format_exc());
        return

    criterion = nn.BCEWithLogitsLoss();
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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
        epoch_train_loss, duration = train_epoch_ft(model, train_loader, criterion, optimizer, device, epoch)
        total_train_time += duration;
        all_epoch_train_losses.append(epoch_train_loss)
        epoch_val_loss, val_metrics_epoch, _, _, _ = evaluate_model_ft(model, val_loader, criterion, device,
                                                                       dataset_name="Validation (Epoch End)")
        all_epoch_val_losses.append(epoch_val_loss);
        all_epoch_val_aucpr.append(val_metrics_epoch.get('pr_auc', 0.0))
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss;
            best_model_state = copy.deepcopy(model.state_dict());
            best_epoch = epoch;
            epochs_no_improve = 0
            logging.info(f"Val loss improved to {best_val_loss:.4f}. Saving model state.")
        else:
            epochs_no_improve += 1
            logging.info(
                f"Val loss ({epoch_val_loss:.4f}) did not improve from best ({best_val_loss:.4f}). Patience: {epochs_no_improve}/{args.early_stopping_patience}")
        if epochs_no_improve >= args.early_stopping_patience: logging.info(f"Early stopping at epoch {epoch}."); break

    logging.info(
        f"--- Training Finished --- Total: {total_train_time:.2f}s. Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        logging.warning("No best model state saved. Using last model state.")

    # 5. Evaluate on Test Set
    logging.info("Evaluating best model on Test set...")
    _, test_metrics, test_cm, test_y_true_np, test_y_pred_proba_np = evaluate_model_ft(model, test_loader, criterion,
                                                                                       device, dataset_name="Test Set")
    plot_evaluation_charts_torch(test_y_true_np, test_y_pred_proba_np, test_cm, "Test", MODEL_NAME, GCS_BUCKET,
                                 GCS_OUTPUT_PREFIX)

    best_val_metrics_dict = {}
    if best_epoch > 0:  # If early stopping did save a best model
        _, best_val_metrics_dict, _, _, _ = evaluate_model_ft(model, val_loader, criterion, device,
                                                              dataset_name="Best Validation Epoch")

    # 6. Perform SHAP Analysis (KernelExplainer for numerical, if present)
    shap_feature_importance = None
    if args.run_shap and num_feat_names:  # Only run SHAP if there are numerical features
        # Create a simple dataloader for X_test_num_np for SHAP KernelExplainer to iterate through easily
        # This loader just yields batches of the numpy array
        class SimpleNumPyLoader:
            def __init__(self, data_np, batch_size):
                self.data_np, self.batch_size = data_np, batch_size; self.n_samples = data_np.shape[0]

            def __iter__(self):
                self.current_idx = 0; return self

            def __next__(self):
                if self.current_idx < self.n_samples:
                    end_idx = min(self.current_idx + self.batch_size, self.n_samples)
                    batch = self.data_np[self.current_idx:end_idx];
                    self.current_idx = end_idx;
                    return batch
                else:
                    raise StopIteration

            def __len__(self):
                return (self.n_samples + self.batch_size - 1) // self.batch_size

        shap_test_loader_num_only = SimpleNumPyLoader(X_test_num_np, args.batch_size)  # Use test numerical data
        # Prepare fixed categorical background tensor (from training data)
        fixed_cat_bg_tensor = torch.tensor(X_train_cat_np, dtype=torch.int64) if X_train_cat_np.shape[1] > 0 else None

        _, shap_feature_importance = perform_shap_analysis_ft_kernel(
            model, train_loader, shap_test_loader_num_only,
            num_feat_names, fixed_cat_bg_tensor, device,
            GCS_BUCKET, GCS_OUTPUT_PREFIX,
            sample_size=args.shap_background_sample_size,
            explain_sample_size=args.shap_explain_sample_size
        )
    elif args.run_shap and not num_feat_names:
        logging.warning("SHAP analysis skipped: No numerical features to analyze with the current SHAP setup.")
    else:
        logging.info("SHAP analysis skipped by user.")

    # 7. Save Model
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/{MODEL_NAME.lower()}_best_model.pt"
    if best_model_state:
        save_model_to_gcs(best_model_state, GCS_BUCKET, model_blob_name)
    else:
        save_model_to_gcs(model.state_dict(), GCS_BUCKET, model_blob_name)

    # 8. Save Logs and Metrics
    logging.info("Saving final logs and metrics...")
    log_summary = {
        "model_type": MODEL_NAME, "script_args": vars(args), "device_used": str(device),
        "metadata_source": METADATA_URI, "n_numerical_features": n_actual_num_features,
        "categorical_cardinalities": actual_cat_cardinalities,
        "training_total_duration_seconds": total_train_time,
        "epoch_training_losses": all_epoch_train_losses, "epoch_validation_losses": all_epoch_val_losses,
        "epoch_validation_aucpr": all_epoch_val_aucpr, "best_epoch_for_early_stopping": best_epoch,
        "best_validation_loss": best_val_loss, "best_epoch_validation_metrics": best_val_metrics_dict,
        "test_set_evaluation_with_best_model": test_metrics,
        "shap_analysis_run": args.run_shap and bool(num_feat_names),
        "shap_top_numerical_features": shap_feature_importance,
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
        description=f"Train and evaluate an {MODEL_NAME} model for FinBench cf2 using FT data from GCS.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    parser.add_argument("--metadata-uri", type=str, required=True,
                        help="GCS URI of the FT preprocessing_ft_metadata.json file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Max number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Patience for early stopping.")
    # FT-Transformer Specific (match make_baseline and rtdl defaults where appropriate)
    parser.add_argument("--ft-d-token", type=int, default=192,
                        help="Token dimension. Must be multiple of n_heads (default 8 for make_baseline).")
    parser.add_argument("--ft-n-blocks", type=int, default=3, help="Number of transformer blocks.")
    # --ft-n-heads is not directly used by make_baseline (uses default 8), but d_token must be compatible.
    # parser.add_argument("--ft-n-heads", type=int, default=8, help="Number of attention heads (info: make_baseline uses its own default).")
    parser.add_argument("--ft-attention-dropout", type=float, default=0.2,
                        help="Attention dropout rate.")  # Matched to rtdl example for n_blocks=3
    parser.add_argument("--ft-ffn-dropout", type=float, default=0.1, help="FFN dropout rate.")  # Matched
    parser.add_argument("--ft-residual-dropout", type=float, default=0.0, help="Residual dropout rate.")
    parser.add_argument("--ft-ffn-factor", type=float, default=4 / 3,
                        help="Multiplier for FFN hidden dim relative to d_token (e.g., 4/3 for ReGLU).")
    # SHAP arguments
    parser.add_argument("--run-shap", action='store_true',
                        help="Run SHAP analysis (KernelExplainer, very slow, numerical only).")
    parser.add_argument("--shap-background-sample-size", type=int, default=50,
                        help="Background samples for KernelSHAP masker.")
    parser.add_argument("--shap-explain-sample-size", type=int, default=10,
                        help="Instances to explain with KernelSHAP.")

    args = parser.parse_args()
    main(args)