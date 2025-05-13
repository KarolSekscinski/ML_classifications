# ft_transformer_pipeline.py (Updated May 12, 2025)
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchmetrics
# Ensure rtdl is installed: pip install rtdl
import rtdl # Revisiting Tabular Deep Learning library
from sklearn.metrics import confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay # Import specific display functions
import matplotlib.pyplot as plt
import seaborn as sns
import shap # Attempt SHAP, may be difficult
import time
from io import BytesIO, StringIO
import json
import os

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
    # (Identical to previous versions)
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

def save_model_to_gcs(model_state_dict, gcs_bucket, gcs_blob_name):
    """Saves a PyTorch model state dictionary to GCS."""
    # (Identical to previous versions)
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

# --- PyTorch Dataset Definition for FT-Transformer ---
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, Y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.int64) # Categorical indices MUST be long integers
        self.Y = torch.tensor(Y, dtype=torch.int64)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.Y[idx]

# --- Training & Evaluation Functions (Adapted for FT-Transformer input) ---
# (train_epoch_ft, evaluate_model_ft functions are identical to the previous version)
def train_epoch_ft(model, loader, criterion, optimizer, device, epoch_num):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for i, (x_num, x_cat, targets) in enumerate(loader):
        x_num, x_cat, targets = x_num.to(device), x_cat.to(device), targets.to(device)
        targets_float = targets.float().unsqueeze(1) # Target must be float for BCEWithLogitsLoss
        optimizer.zero_grad()
        outputs = model(x_num, x_cat)
        loss = criterion(outputs, targets_float)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 100 == 0: logging.info(f'Epoch {epoch_num}, Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}')
    avg_loss = total_loss / len(loader)
    epoch_duration = time.time() - start_time
    logging.info(f'Epoch {epoch_num} Training Summary: Average Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s')
    return avg_loss, epoch_duration

def evaluate_model_ft(model, loader, criterion, device):
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
            targets_int = targets.int() # Keep int version for metrics
            outputs_logits = model(x_num, x_cat)
            loss = criterion(outputs_logits, targets_float)
            total_loss += loss.item()
            outputs_probs = torch.sigmoid(outputs_logits).squeeze()
            metrics_collection.update(outputs_probs, targets_int) # Use probs for AUC/AP, preds derived internally
            # Store predictions and targets for CM
            outputs_preds = (outputs_probs > 0.5).int()
            all_preds.append(outputs_preds.cpu().numpy())
            all_targets.append(targets_int.cpu().numpy())
    avg_loss = total_loss / len(loader)
    final_metrics = metrics_collection.compute()
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    cm = confusion_matrix(all_targets, all_preds)
    metrics_dict = {k: v.item() for k, v in final_metrics.items()}
    metrics_dict["confusion_matrix"] = cm.tolist()
    logging.info("--- Model Evaluation (Test Set) ---")
    logging.info(f"Average Loss: {avg_loss:.4f}")
    for name, value in metrics_dict.items():
        if name != "confusion_matrix": logging.info(f"{name.replace('_', ' ').title()}: {value:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    return avg_loss, metrics_dict, cm

# --- Plotting Function ---
# (plot_evaluation_charts function is identical to the previous version)
def plot_evaluation_charts(y_true_np, y_pred_proba_np, cm, model_name, gcs_bucket, gcs_output_prefix):
    """Generates and saves Confusion Matrix, ROC, and PR curve plots."""
    # 1. Confusion Matrix Plot
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_name} Confusion Matrix')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        save_plot_to_gcs(fig_cm, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_confusion_matrix.png")
    except Exception as e: logging.error(f"Failed to generate or save Confusion Matrix plot: {e}")
    # 2. ROC Curve Plot
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_roc, name=model_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
        ax_roc.set_title(f'{model_name} Receiver Operating Characteristic (ROC) Curve')
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_roc_curve.png")
    except Exception as e: logging.error(f"Failed to generate or save ROC Curve plot: {e}")
    # 3. Precision-Recall Curve Plot
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(y_true_np, y_pred_proba_np, ax=ax_pr, name=model_name)
        ax_pr.set_title(f'{model_name} Precision-Recall (PR) Curve')
        save_plot_to_gcs(fig_pr, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_pr_curve.png")
    except Exception as e: logging.error(f"Failed to generate or save PR Curve plot: {e}")

# --- SHAP Analysis Function ---
# (perform_shap_analysis_ft using simplified numerical-only approach is identical to the previous version)
def perform_shap_analysis_ft(model, train_loader, test_loader, feature_names_num, feature_names_cat, device, gcs_bucket, gcs_output_prefix, sample_size=100):
    """Attempts SHAP analysis for FT-Transformer using DeepExplainer (simplified numerical only)."""
    logging.warning("--- SHAP Analysis (FT-Transformer - DeepExplainer - Simplified) ---")
    logging.warning("SHAP for Transformers is complex. Explaining numerical features only.")
    logging.warning(f"Using background/explanation sample size: approx {sample_size}")
    try:
        model.eval()
        # Get background samples (Num + Cat needed for wrapper)
        bg_num_list, bg_cat_list = [], []
        count = 0
        for x_num, x_cat, _ in train_loader:
             batch_size = x_num.shape[0]; needed = sample_size - count; take = min(needed, batch_size)
             if needed <= 0: break
             bg_num_list.append(x_num[:take]); bg_cat_list.append(x_cat[:take]); count += take
        if not bg_num_list: raise ValueError("Could not get background samples for SHAP.")
        background_num = torch.cat(bg_num_list).to(device); background_cat = torch.cat(bg_cat_list).to(device)
        # Get test samples to explain
        test_num_list, test_cat_list = [], []; count = 0
        for x_num, x_cat, _ in test_loader:
            batch_size = x_num.shape[0]; needed = sample_size - count; take = min(needed, batch_size)
            if needed <= 0: break
            test_num_list.append(x_num[:take]); test_cat_list.append(x_cat[:take]); count += take
        if not test_num_list: raise ValueError("Could not get test samples for SHAP.")
        test_sample_num = torch.cat(test_num_list).to(device)
        # Wrapper class definition (simplified)
        class ModelWrapperNumOnly(nn.Module):
            def __init__(self, ft_model, fixed_cat_input):
                super().__init__(); self.ft_model = ft_model
                # Use the first background sample's cat features for all explanations in this simplified approach
                self.fixed_cat_input_single = fixed_cat_input[0:1, :].to(device) # Shape [1, n_cat]
            def forward(self, x_num):
                fixed_cat_batch = self.fixed_cat_input_single.repeat(x_num.shape[0], 1) # Repeat for batch size
                return self.ft_model(x_num, fixed_cat_batch)
        # --- End Wrapper ---
        wrapped_model = ModelWrapperNumOnly(model, background_cat)
        start_shap_time = time.time(); logging.info("Initializing SHAP DeepExplainer...")
        explainer = shap.DeepExplainer(wrapped_model, background_num)
        logging.info(f"Calculating SHAP values for numerical features on {test_sample_num.shape[0]} test samples...")
        shap_values_num_tensor = explainer.shap_values(test_sample_num)
        end_shap_time = time.time(); logging.info(f"SHAP values (numerical) calculated in {end_shap_time - start_shap_time:.2f}s.")
        # Process results (convert to numpy, plot, get importance)
        shap_values_num_np = shap_values_num_tensor.cpu().numpy(); test_sample_num_np = test_sample_num.cpu().numpy()
        test_sample_num_df = pd.DataFrame(test_sample_num_np, columns=feature_names_num)
        fig_shap_num, ax_shap_num = plt.subplots(); shap.summary_plot(shap_values_num_np, test_sample_num_df, plot_type="dot", show=False)
        plt.title("SHAP Summary Plot (FT-Transformer - Numerical Features Only)")
        try: plt.tight_layout()
        except: logging.warning("Could not apply tight_layout() to SHAP plot.")
        save_plot_to_gcs(fig_shap_num, gcs_bucket, f"{gcs_output_prefix}/plots/ft_transformer_shap_summary_numerical.png")
        mean_abs_shap_num = np.mean(np.abs(shap_values_num_np), axis=0)
        feature_importance_num = pd.DataFrame({'feature': feature_names_num, 'mean_abs_shap': mean_abs_shap_num})
        feature_importance_num = feature_importance_num.sort_values('mean_abs_shap', ascending=False)
        logging.info("Top 10 Numerical features by Mean Absolute SHAP value:\n" + feature_importance_num.head(10).to_string())
        logging.warning("SHAP values for categorical features were not calculated.")
        return shap_values_num_np, feature_importance_num.to_dict('records')
    except Exception as e:
        logging.error(f"Failed during simplified SHAP analysis: {e}"); import traceback; logging.error(traceback.format_exc())
        return None, None

# --- Main Execution ---

def main(args):
    logging.info("--- Starting FT-Transformer Training Pipeline ---")
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

    # --- 2. Load FT-Transformer Specific Metadata --- ## UPDATED SECTION ##
    logging.info(f"Loading FT-Transformer specific metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata_str = gcs_utils.download_blob_as_string(GCS_BUCKET, metadata_blob_name)
        if metadata_str is None: raise FileNotFoundError("Could not download metadata file.")
        metadata = json.loads(metadata_str)
        logging.info("Metadata loaded successfully.")

        # Extract paths for FT-Transformer specific files
        data_paths = metadata.get("gcs_paths", {})
        x_train_num_path = data_paths.get("X_train_num_scaled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_train_cat_path = data_paths.get("X_train_cat_indices", "").replace(f"gs://{GCS_BUCKET}/", "") # Path to indices
        y_train_path = data_paths.get("y_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_test_num_path = data_paths.get("X_test_num_scaled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_test_cat_path = data_paths.get("X_test_cat_indices", "").replace(f"gs://{GCS_BUCKET}/", "") # Path to indices
        y_test_path = data_paths.get("y_test", "").replace(f"gs://{GCS_BUCKET}/", "")

        # Check if all required paths were found
        required_paths = [x_train_num_path, x_train_cat_path, y_train_path, x_test_num_path, x_test_cat_path, y_test_path]
        if not all(required_paths):
            missing = [name for name, path in zip(["x_train_num", "x_train_cat", "y_train", "x_test_num", "x_test_cat", "y_test"], required_paths) if not path]
            raise ValueError(f"Missing required data paths in metadata: {missing}. Ensure '{METADATA_URI}' is from preprocess_ft.py.")

        # Extract other necessary metadata
        cat_cardinalities = metadata.get("cat_feature_cardinalities")
        if not cat_cardinalities: raise ValueError("Categorical cardinalities ('cat_feature_cardinalities') not found in metadata.")
        numerical_features_used = metadata.get("numerical_features_used", [])
        categorical_features_used = metadata.get("categorical_features_used", [])
        if not numerical_features_used: logging.warning("Numerical feature names not found in metadata.")
        if not categorical_features_used: logging.warning("Categorical feature names not found in metadata.")

    except Exception as e:
        logging.error(f"Failed to load or parse metadata from {METADATA_URI}: {e}")
        return

    # --- 3. Load Processed Data (Numerical and Categorical Indices) --- ## UPDATED SECTION ##
    try:
        logging.info(f"Loading numerical training data from: {x_train_num_path}")
        X_train_num_df = load_data_from_gcs(GCS_BUCKET, x_train_num_path)
        logging.info(f"Loading categorical index training data from: {x_train_cat_path}")
        X_train_cat_df = load_data_from_gcs(GCS_BUCKET, x_train_cat_path)
        logging.info(f"Loading training target data from: {y_train_path}")
        y_train_df = load_data_from_gcs(GCS_BUCKET, y_train_path)

        logging.info(f"Loading numerical test data from: {x_test_num_path}")
        X_test_num_df = load_data_from_gcs(GCS_BUCKET, x_test_num_path)
        logging.info(f"Loading categorical index test data from: {x_test_cat_path}")
        X_test_cat_df = load_data_from_gcs(GCS_BUCKET, x_test_cat_path)
        logging.info(f"Loading test target data from: {y_test_path}")
        y_test_df = load_data_from_gcs(GCS_BUCKET, y_test_path)
    except Exception as e:
        logging.error(f"Failed to load data from GCS paths specified in metadata: {e}")
        return

    # Convert to numpy arrays
    X_train_num_np = X_train_num_df.values.astype(np.float32)
    X_train_cat_np = X_train_cat_df.values.astype(np.int64) # Indices MUST be integer type
    y_train_np = y_train_df.iloc[:, 0].values.astype(np.int64)
    X_test_num_np = X_test_num_df.values.astype(np.float32)
    X_test_cat_np = X_test_cat_df.values.astype(np.int64) # Indices MUST be integer type
    y_test_np = y_test_df.iloc[:, 0].values.astype(np.int64)

    # --- 4. Create DataLoaders ---
    train_dataset = TabularDataset(X_train_num_np, X_train_cat_np, y_train_np)
    test_dataset = TabularDataset(X_test_num_np, X_test_cat_np, y_test_np)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)

    # --- 5. Initialize Model, Criterion, Optimizer ---
    n_num_features = X_train_num_np.shape[1] # Get actual count from loaded data
    # Ensure cardinalities match the number of categorical features loaded
    if len(cat_cardinalities) != X_train_cat_np.shape[1]:
        logging.error(f"Mismatch between number of cardinalities ({len(cat_cardinalities)}) and number of loaded categorical features ({X_train_cat_np.shape[1]}). Check metadata and preprocessing.")
        return

    # In ft_transformer_pipeline.py, around line 331:

    model = rtdl.FTTransformer(
        n_cont_features=n_num_features,  # RENAMED from n_num_features
        cat_cardinalities=cat_cardinalities,
        d_token=args.ft_d_token,
        n_blocks=args.ft_n_blocks,
        attention_n_heads=args.ft_n_heads,
        attention_dropout=args.ft_attention_dropout,
        ffn_d_hidden_factor=args.ft_ffn_factor,  # CHANGED: Pass the factor directly
        ffn_dropout=args.ft_ffn_dropout,
        residual_dropout=args.ft_residual_dropout,
        # The older rtdl versions have defaults for 'activation', 'prenormalization', 'initialization'
        # which are often 'reglu', True, and 'kaiming' respectively.
        # We will rely on those defaults unless you want to add them as command-line arguments.
        d_out=1
    ).to(device)
    logging.info(f"FT-Transformer Model Architecture:\n{model}")
    logging.info(f"Using {n_num_features} numerical features and {len(cat_cardinalities)} categorical features.")
    logging.info(f"Categorical cardinalities: {cat_cardinalities}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # --- 6. Training Loop ---
    logging.info("--- Starting Training ---")
    total_train_time = 0; epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        avg_loss, duration = train_epoch_ft(model, train_loader, criterion, optimizer, device, epoch)
        epoch_losses.append(avg_loss); total_train_time += duration
    logging.info(f"--- Training Finished --- Total Duration: {total_train_time:.2f}s")

    # --- 7. Evaluate on Test Set ---
    logging.info("Evaluating model on test set...")
    test_loss, test_metrics, test_cm = evaluate_model_ft(model, test_loader, criterion, device)

    # Retrieve probabilities needed for plotting
    all_probs_list = []; all_targets_list = []
    model.eval()
    with torch.no_grad():
        for x_num, x_cat, targets in test_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            outputs_logits = model(x_num, x_cat)
            outputs_probs = torch.sigmoid(outputs_logits).squeeze()
            all_probs_list.append(outputs_probs.cpu().numpy())
            all_targets_list.append(targets.cpu().numpy()) # Targets are already on CPU/numpy
    y_pred_proba_np = np.concatenate(all_probs_list)
    y_true_np = np.concatenate(all_targets_list)

    # --- 8. Generate and Save Plots ---
    plot_evaluation_charts(y_true_np, y_pred_proba_np, test_cm, 'FT-Transformer', GCS_BUCKET, GCS_OUTPUT_PREFIX)

    # --- 9. Perform SHAP Analysis (Simplified Numerical Only) ---
    shap_values = None; shap_feature_importance = None
    if args.run_shap:
        # Pass feature names extracted from metadata
        shap_values, shap_feature_importance = perform_shap_analysis_ft(
            model, train_loader, test_loader, numerical_features_used, categorical_features_used, device,
            GCS_BUCKET, GCS_OUTPUT_PREFIX, sample_size=args.shap_sample_size
        )
    else:
        logging.info("SHAP analysis skipped as per --run-shap flag.")

    # --- 10. Save Model ---
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/ft_transformer_model_state_dict.pt"
    save_model_to_gcs(model.state_dict(), GCS_BUCKET, model_blob_name)

    # --- 11. Save Logs and Metrics ---
    logging.info("Saving final logs and metrics...")
    log_summary = {
        "model_type": "FT-Transformer", "script_args": vars(args), "device_used": str(device),
        "metadata_source": METADATA_URI, #"model_architecture": str(model), # Can be too verbose
        "n_numerical_features": n_num_features, "categorical_cardinalities": cat_cardinalities,
        "training_duration_seconds": total_train_time, "training_avg_loss_per_epoch": epoch_losses,
        "test_set_evaluation": {"loss": test_loss, "metrics": test_metrics},
        "shap_analysis_run": args.run_shap, "shap_top_numerical_features": shap_feature_importance,
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }
    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/ft_transformer_training_log.json"
    try:
        log_string = json.dumps(log_summary, indent=4, default=str)
        gcs_utils.upload_string_to_gcs(GCS_BUCKET, log_string, log_blob_name, content_type='application/json')
        logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")
    except Exception as e:
        logging.error(f"Failed to save log summary: {e}")

    logging.info("--- FT-Transformer Training Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an FT-Transformer model using PyTorch on GPU.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    # Updated help text for metadata URI
    parser.add_argument("--metadata-uri", type=str, required=True, help="GCS URI of the preprocessing_ft_metadata.json file (generated by preprocess_ft.py).")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    # FT-Transformer Specific Hyperparameters
    parser.add_argument("--ft-d-token", type=int, default=192, help="Dimension of embeddings/transformer layers.")
    parser.add_argument("--ft-n-blocks", type=int, default=3, help="Number of transformer blocks.")
    parser.add_argument("--ft-n-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--ft-attention-dropout", type=float, default=0.2, help="Dropout rate for attention.")
    parser.add_argument("--ft-ffn-dropout", type=float, default=0.1, help="Dropout rate for feed-forward layers.")
    parser.add_argument("--ft-residual-dropout", type=float, default=0.0, help="Dropout for residual connections.")
    parser.add_argument("--ft-ffn-factor", type=float, default=4/3, help="Multiplier for FFN hidden dim.")
    # SHAP arguments
    parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis (experimental simplified version).")
    parser.add_argument("--shap-sample-size", type=int, default=200, help="Approx number of samples for SHAP.")

    args = parser.parse_args()
    main(args)