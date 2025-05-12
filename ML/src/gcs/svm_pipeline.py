# svm_pipeline.py
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import joblib # Preferred for scikit-learn models with numpy arrays
import json
import os

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid') # Use a seaborn style for plots

# --- Helper Functions ---

def load_data_from_gcs(gcs_bucket, gcs_path, feature_names=None):
    """Loads CSV data from GCS into a pandas DataFrame."""
    logging.info(f"Loading data from: gs://{gcs_bucket}/{gcs_path}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, gcs_path)
        if data_bytes is None:
            raise FileNotFoundError(f"Could not download GCS file: gs://{gcs_bucket}/{gcs_path}")
        df = pd.read_csv(BytesIO(data_bytes))
        # If feature_names are provided (for X data), assign them
        if feature_names is not None and not df.columns.equals(pd.Index(feature_names)):
             if len(feature_names) == len(df.columns):
                 logging.warning(f"Assigning provided feature names to loaded data from {gcs_path}.")
                 df.columns = feature_names
             else:
                 logging.error(f"Feature name count ({len(feature_names)}) does not match column count ({len(df.columns)}) in {gcs_path}. Cannot assign names.")
                 # Depending on strictness, you might raise an error here
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
        logging.info(f"Plot successfully saved to gs://{gcs_bucket}/{gcs_blob_name}")
        plt.close(fig) # Close the figure to free memory
    except Exception as e:
        logging.error(f"ERROR saving plot to GCS ({gcs_blob_name}): {e}")
        plt.close(fig) # Ensure figure is closed even on error

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculates and logs standard classification metrics."""
    logging.info("--- Model Evaluation ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist() # Convert numpy array to list for JSON serialization
    }

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall (Sensitivity): {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"AUC-ROC: {roc_auc:.4f}")
    logging.info(f"AUC-PR: {pr_auc:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")

    return metrics, cm # Return metrics dict and cm object for plotting

def plot_evaluation_charts(model, X_test, y_test, cm, gcs_bucket, gcs_output_prefix):
    """Generates and saves Confusion Matrix, ROC, and PR curve plots."""
    # 1. Confusion Matrix Plot
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title('Confusion Matrix')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        save_plot_to_gcs(fig_cm, gcs_bucket, f"{gcs_output_prefix}/plots/svm_confusion_matrix.png")
    except Exception as e:
        logging.error(f"Failed to generate or save Confusion Matrix plot: {e}")

    # 2. ROC Curve Plot
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc, name='SVM')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)') # Add chance line
        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket, f"{gcs_output_prefix}/plots/svm_roc_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save ROC Curve plot: {e}")

    # 3. Precision-Recall Curve Plot
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr, name='SVM')
        ax_pr.set_title('Precision-Recall (PR) Curve')
        save_plot_to_gcs(fig_pr, gcs_bucket, f"{gcs_output_prefix}/plots/svm_pr_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save PR Curve plot: {e}")

def perform_shap_analysis(model, X_train, X_test, feature_names, gcs_bucket, gcs_output_prefix, sample_size=100):
    """Performs SHAP analysis using KernelExplainer and saves summary plot."""
    logging.warning("--- SHAP Analysis (SVM - KernelExplainer) ---")
    logging.warning(f"KernelExplainer can be VERY SLOW for SVM.")
    logging.warning(f"Using a background sample of size {min(sample_size, X_train.shape[0])} from training data.")
    logging.warning(f"Explaining a sample of size {min(sample_size, X_test.shape[0])} from test data.")

    try:
        # Sample background data (important for KernelExplainer performance)
        X_train_summary = shap.sample(X_train, min(sample_size, X_train.shape[0]), random_state=42)
        if isinstance(X_train_summary, pd.DataFrame):
             X_train_summary_np = X_train_summary.values
        else: # Already a numpy array
             X_train_summary_np = X_train_summary

        # Sample data to explain (if test set is large)
        X_test_sample = shap.sample(X_test, min(sample_size, X_test.shape[0]), random_state=43)
        if isinstance(X_test_sample, pd.DataFrame):
             X_test_sample_df = X_test_sample # Keep as df for column names if needed
        else: # Convert numpy array back to DataFrame with feature names
             X_test_sample_df = pd.DataFrame(X_test_sample, columns=feature_names)

        # KernelExplainer requires a function that outputs model probabilities
        # Ensure model.predict_proba exists (requires probability=True in SVC)
        if not hasattr(model, "predict_proba"):
             logging.error("SVC model was not trained with probability=True. Cannot compute SHAP values with KernelExplainer predict_proba.")
             return None, None

        start_shap_time = time.time()
        logging.info("Initializing SHAP KernelExplainer...")
        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary_np)

        logging.info(f"Calculating SHAP values for {X_test_sample_df.shape[0]} test samples...")
        # Ensure we pass a DataFrame with column names if X_test_sample was originally numpy
        shap_values = explainer.shap_values(X_test_sample_df)
        end_shap_time = time.time()
        logging.info(f"SHAP values calculated in {end_shap_time - start_shap_time:.2f} seconds.")

        # For binary classification, shap_values returns a list [shap_values_class_0, shap_values_class_1]
        # We usually plot for the positive class (class 1)
        shap_values_pos_class = shap_values[1]

        # Generate and save summary plot
        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values_pos_class, X_test_sample_df, plot_type="dot", show=False)
        plt.title("SHAP Summary Plot (SVM - Positive Class)")
        save_plot_to_gcs(fig_shap, gcs_bucket, f"{gcs_output_prefix}/plots/svm_shap_summary.png")

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values_pos_class), axis=0)
        feature_importance = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
        feature_importance = feature_importance.sort_values('mean_abs_shap', ascending=False)
        logging.info("Top 10 features by Mean Absolute SHAP value:\n" + feature_importance.head(10).to_string())

        return shap_values, feature_importance.to_dict('records')

    except Exception as e:
        logging.error(f"Failed during SHAP analysis: {e}")
        return None, None

def save_model_to_gcs(model, gcs_bucket, gcs_blob_name):
    """Saves a model object to GCS using joblib."""
    logging.info(f"Saving model object to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        with BytesIO() as buf:
            joblib.dump(model, buf)
            buf.seek(0)
            model_bytes = buf.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model successfully saved to gs://{gcs_bucket}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving model to GCS ({gcs_blob_name}): {e}")

# --- Main Execution ---

def main(args):
    """Main training and evaluation pipeline function."""
    logging.info("--- Starting SVM Training Pipeline ---")
    GCS_BUCKET = args.gcs_bucket
    # Ensure prefixes don't have leading/trailing slashes for consistency
    GCS_INPUT_PREFIX = args.gcs_input_prefix.strip('/')
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')
    METADATA_URI = args.metadata_uri

    # --- 1. Load Metadata ---
    logging.info(f"Loading metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata_str = gcs_utils.download_blob_as_string(GCS_BUCKET, metadata_blob_name)
        if metadata_str is None:
             raise FileNotFoundError("Could not download metadata file.")
        metadata = json.loads(metadata_str)
        logging.info("Metadata loaded successfully.")
        # Extract necessary info
        processed_feature_names = metadata.get("processed_feature_names")
        if not processed_feature_names:
             logging.error("Processed feature names not found in metadata. Cannot proceed with named features.")
             return
        data_paths = metadata.get("gcs_paths", {})
        x_train_path = data_paths.get("X_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_train_path = data_paths.get("y_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_test_path = data_paths.get("X_test_processed", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_test_path = data_paths.get("y_test", "").replace(f"gs://{GCS_BUCKET}/", "")

        if not all([x_train_path, y_train_path, x_test_path, y_test_path]):
             logging.error("One or more required data paths are missing in metadata.")
             return

    except Exception as e:
        logging.error(f"Failed to load or parse metadata from {METADATA_URI}: {e}")
        return

    # --- 2. Load Processed Data ---
    X_train = load_data_from_gcs(GCS_BUCKET, x_train_path, processed_feature_names)
    y_train_df = load_data_from_gcs(GCS_BUCKET, y_train_path)
    X_test = load_data_from_gcs(GCS_BUCKET, x_test_path, processed_feature_names)
    y_test_df = load_data_from_gcs(GCS_BUCKET, y_test_path)

    # Convert target Series/DataFrames to 1D numpy arrays
    y_train = y_train_df.iloc[:, 0].values # Get first column as numpy array
    y_test = y_test_df.iloc[:, 0].values   # Get first column as numpy array

    # --- 3. Train SVM Model ---
    logging.info("--- Training SVM Model ---")
    # Using probability=True is required for predict_proba (needed for AUC, SHAP) but slows down training.
    # Consider class_weight='balanced' if data wasn't resampled (but we used SMOTE).
    svm_model = SVC(
        C=args.svm_c,
        kernel=args.svm_kernel,
        gamma=args.svm_gamma, # Often 'scale' or 'auto' for RBF
        probability=True, # Crucial for AUC/SHAP
        random_state=42,
        verbose=True # Show scikit-learn progress
    )

    start_train_time = time.time()
    svm_model.fit(X_train, y_train)
    end_train_time = time.time()
    training_duration = end_train_time - start_train_time
    logging.info(f"SVM training completed in {training_duration:.2f} seconds.")

    # --- 4. Predict on Test Set ---
    logging.info("Predicting on test set...")
    y_pred = svm_model.predict(X_test)
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1] # Probability of positive class

    # --- 5. Evaluate Model ---
    metrics, conf_matrix = evaluate_model(y_test, y_pred, y_pred_proba)

    # --- 6. Generate and Save Plots ---
    logging.info("Generating evaluation plots...")
    # Check if X_test is pandas DF or numpy array - plotting functions might need numpy
    X_test_eval = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    plot_evaluation_charts(svm_model, X_test_eval, y_test, conf_matrix, GCS_BUCKET, GCS_OUTPUT_PREFIX)

    # --- 7. Perform SHAP Analysis ---
    shap_values = None
    shap_feature_importance = None
    if args.run_shap:
        # Pass DataFrame X_train and X_test to SHAP if possible for feature names
        shap_values, shap_feature_importance = perform_shap_analysis(
            svm_model, X_train, X_test, processed_feature_names,
            GCS_BUCKET, GCS_OUTPUT_PREFIX, sample_size=args.shap_sample_size
        )
    else:
        logging.info("SHAP analysis skipped as per --run-shap flag.")

    # --- 8. Save Model ---
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/svm_model.joblib"
    save_model_to_gcs(svm_model, GCS_BUCKET, model_blob_name)

    # --- 9. Save Logs and Metrics ---
    logging.info("Saving logs and metrics...")
    log_summary = {
        "model_type": "SVM",
        "script_args": vars(args),
        "metadata_source": METADATA_URI,
        "training_duration_seconds": training_duration,
        "evaluation_metrics": metrics,
        "shap_analysis_run": args.run_shap,
        "shap_top_features": shap_feature_importance, # Will be None if SHAP wasn't run
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }

    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/svm_training_log.json"
    try:
        log_string = json.dumps(log_summary, indent=4)
        gcs_utils.upload_string_to_gcs(GCS_BUCKET, log_string, log_blob_name, content_type='application/json')
        logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")
    except Exception as e:
        logging.error(f"Failed to save log summary: {e}")

    logging.info("--- SVM Training Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an SVM model using data from GCS.")
    parser.add_argument("--gcs-bucket", type=str, required=True,
                        help="GCS bucket name for input metadata/data and output artifacts.")
    parser.add_argument("--metadata-uri", type=str, required=True,
                        help="GCS URI (gs://bucket/path/to/preprocessing_metadata.json) of the metadata file.")
    # Input prefix is derived from metadata, output prefix needs specifying
    parser.add_argument("--gcs-output-prefix", type=str, required=True,
                        help="GCS prefix (folder path) within the bucket to save outputs (logs, plots, model) (e.g., 'fraud_detection/svm_run_1').")
    # Optional SVM Hyperparameters
    parser.add_argument("--svm-c", type=float, default=1.0, help="SVM regularization parameter C.")
    parser.add_argument("--svm-kernel", type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'], help="SVM kernel type.")
    parser.add_argument("--svm-gamma", type=str, default='scale', help="SVM kernel coefficient gamma ('scale', 'auto', or float).")
    # SHAP arguments
    parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis (can be very slow for SVM).")
    parser.add_argument("--shap-sample-size", type=int, default=100, help="Number of samples for SHAP background and explanation.")

    # The input prefix isn't strictly needed if we use the metadata file,
    # but keeping it might be useful if someone wants to override paths.
    # Let's rely solely on metadata URI for input paths for simplicity now.
    # parser.add_argument("--gcs-input-prefix", type=str, required=True,
    #                     help="GCS prefix where the 'processed_data' folder resides (e.g., 'fraud_detection/processed').")

    args = parser.parse_args()

    # Create local directories for output structure if they don't exist (optional, mainly for local debugging)
    # For GCS uploads, the path is handled by the blob name.
    # Path(f"{args.gcs_output_prefix}/plots").mkdir(parents=True, exist_ok=True)
    # Path(f"{args.gcs_output_prefix}/logs").mkdir(parents=True, exist_ok=True)
    # Path(f"{args.gcs_output_prefix}/model").mkdir(parents=True, exist_ok=True)

    main(args)