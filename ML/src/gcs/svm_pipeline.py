# svm_pipeline.py (Modified for LinearSVC)
import argparse
import logging
import pandas as pd
import numpy as np
# *** CHANGED IMPORTS ***
from sklearn.svm import LinearSVC # Use LinearSVC
from sklearn.calibration import CalibratedClassifierCV # To get probabilities
# *** END CHANGED IMPORTS ***
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay # Ensure these are imported
)
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import joblib
import json
import os

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')

# --- Helper Functions (No changes needed) ---
# load_data_from_gcs, save_plot_to_gcs remain the same
def load_data_from_gcs(gcs_bucket, gcs_path, feature_names=None):
    """Loads CSV data from GCS into a pandas DataFrame."""
    logging.info(f"Loading data from: gs://{gcs_bucket}/{gcs_path}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, gcs_path)
        if data_bytes is None: raise FileNotFoundError(f"Could not download GCS file: gs://{gcs_bucket}/{gcs_path}")
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

# evaluate_model remains the same as it requires y_pred_proba, which CalibratedClassifierCV provides
def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculates and logs standard classification metrics."""
    logging.info("--- Model Evaluation ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Ensure y_pred_proba exists and is valid before calculating AUC scores
    roc_auc = -1.0 # Default value if probabilities are not available
    pr_auc = -1.0
    if y_pred_proba is not None:
         try:
             roc_auc = roc_auc_score(y_true, y_pred_proba)
             pr_auc = average_precision_score(y_true, y_pred_proba)
         except ValueError as e:
             logging.warning(f"Could not calculate AUC scores: {e}")
             roc_auc = -1.0
             pr_auc = -1.0
    else:
         logging.warning("Probability scores (y_pred_proba) not provided. AUC scores cannot be calculated.")

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "confusion_matrix": cm.tolist()
    }
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall (Sensitivity): {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    if y_pred_proba is not None:
         logging.info(f"AUC-ROC: {roc_auc:.4f}")
         logging.info(f"AUC-PR: {pr_auc:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    return metrics, cm

# plot_evaluation_charts remains compatible as CalibratedClassifierCV follows sklearn API
def plot_evaluation_charts(calibrated_model, X_test, y_test, cm, model_name, gcs_bucket, gcs_output_prefix):
    """Generates and saves Confusion Matrix, ROC, and PR curve plots for the calibrated model."""
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

    # Ensure the model has predict_proba (CalibratedClassifierCV does)
    if not hasattr(calibrated_model, "predict_proba"):
         logging.error(f"Model {model_name} does not have predict_proba method. Cannot plot ROC/PR curves.")
         return

    # 2. ROC Curve Plot
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_estimator(calibrated_model, X_test, y_test, ax=ax_roc, name=model_name) # Use calibrated model
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
        ax_roc.set_title(f'{model_name} Receiver Operating Characteristic (ROC) Curve')
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_roc_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save ROC Curve plot: {e}")

    # 3. Precision-Recall Curve Plot
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_estimator(calibrated_model, X_test, y_test, ax=ax_pr, name=model_name) # Use calibrated model
        ax_pr.set_title(f'{model_name} Precision-Recall (PR) Curve')
        save_plot_to_gcs(fig_pr, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_pr_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save PR Curve plot: {e}")


# perform_shap_analysis uses KernelExplainer which requires predict_proba, compatible with CalibratedClassifierCV
# Note: It will still be slow, but potentially faster than KernelExplainer on SVC(kernel='rbf')
def perform_shap_analysis(calibrated_model, X_train, X_test, feature_names, gcs_bucket, gcs_output_prefix,
                          sample_size=100):
    logging.warning("--- SHAP Analysis (LinearSVC + Calibrated - KernelExplainer) ---")
    logging.warning(
        f"KernelExplainer can be slow. Using background sample size {min(sample_size, X_train.shape[0])} and explaining {min(sample_size, X_test.shape[0])} samples.")

    try:
        if not isinstance(feature_names, list) or not all(isinstance(fn, str) for fn in feature_names):
            logging.error(
                f"feature_names is not a list of strings. Type: {type(feature_names)}. Attempting conversion if numpy array.")
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()
                if not all(isinstance(fn, str) for fn in feature_names):
                    logging.error("Conversion of feature_names to list of strings failed.")
                    return None, None
            else:
                logging.error("feature_names is not a list or numpy array of strings.")
                return None, None

        if isinstance(X_train, pd.DataFrame):
            X_train_np = X_train.values
        elif isinstance(X_train, np.ndarray):
            X_train_np = X_train
        else:
            logging.error(f"X_train is of unexpected type: {type(X_train)}");
            return None, None

        if isinstance(X_test, pd.DataFrame):
            X_test_np = X_test.values
        elif isinstance(X_test, np.ndarray):
            X_test_np = X_test
        else:
            logging.error(f"X_test is of unexpected type: {type(X_test)}");
            return None, None

        if X_train_np.shape[1] != len(feature_names):
            logging.error(
                f"Feature count mismatch: X_train_np ({X_train_np.shape[1]}) vs feature_names ({len(feature_names)}).");
            return None, None
        if X_test_np.shape[1] != len(feature_names):
            logging.error(
                f"Feature count mismatch: X_test_np ({X_test_np.shape[1]}) vs feature_names ({len(feature_names)}).");
            return None, None

        X_train_summary_np = shap.sample(X_train_np, min(sample_size, X_train_np.shape[0]), random_state=42)
        X_test_sample_np = shap.sample(X_test_np, min(sample_size, X_test_np.shape[0]), random_state=43)

        if not hasattr(calibrated_model, "predict_proba"):
            logging.error("Calibrated model lacks predict_proba method for SHAP.");
            return None, None

        start_shap_time = time.time()
        logging.info("Initializing SHAP KernelExplainer...")
        explainer = shap.KernelExplainer(calibrated_model.predict_proba, X_train_summary_np)

        logging.info(f"Calculating SHAP values for {X_test_sample_np.shape[0]} samples...")
        shap_values_output = explainer.shap_values(X_test_sample_np)
        end_shap_time = time.time()
        logging.info(f"SHAP values calculated in {end_shap_time - start_shap_time:.2f} seconds.")

        shap_values_pos_class = None
        if isinstance(shap_values_output, list) and len(shap_values_output) == 2:
            # Expected for binary classification with predict_proba (returns P(class 0), P(class 1))
            # SHAP returns one array of shap_values for each output of predict_proba
            shap_values_pos_class = shap_values_output[1]  # SHAP values for P(class 1)
            logging.info("SHAP output is a list of 2 arrays; using index 1 for positive class SHAP values.")
        elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 2:
            # This might happen if predict_proba was effectively for a single class or SHAP simplified.
            # This is less typical for KernelExplainer with a sklearn binary classifier's predict_proba.
            logging.warning("SHAP output is a single 2D NumPy array. Assuming these are for the positive class.")
            shap_values_pos_class = shap_values_output
        else:
            logging.error(f"Unexpected SHAP values output format. Type: {type(shap_values_output)}")
            if isinstance(shap_values_output, list):
                logging.error(f"List length: {len(shap_values_output)}")
            elif isinstance(shap_values_output, np.ndarray):
                logging.error(f"Array ndim: {shap_values_output.ndim}")
            return None, None

        if shap_values_pos_class is None or shap_values_pos_class.ndim != 2:
            logging.error(
                f"shap_values_pos_class is not a 2D array as expected. Shape: {getattr(shap_values_pos_class, 'shape', 'N/A')}")
            return None, None

        # Final sanity checks before plotting
        if shap_values_pos_class.shape[0] != X_test_sample_np.shape[0]:
            logging.error(
                f"Sample count mismatch: SHAP values ({shap_values_pos_class.shape[0]}) vs Data ({X_test_sample_np.shape[0]})")
            return None, None
        if shap_values_pos_class.shape[1] != X_test_sample_np.shape[1]:
            logging.error(
                f"Feature count mismatch: SHAP values ({shap_values_pos_class.shape[1]}) vs Data ({X_test_sample_np.shape[1]})")
            return None, None
        if X_test_sample_np.shape[1] != len(feature_names):
            logging.error(
                f"Feature count mismatch: Data ({X_test_sample_np.shape[1]}) vs feature_names ({len(feature_names)})")
            return None, None

        logging.info(
            f"Plotting: shap_values_pos_class.shape={shap_values_pos_class.shape}, X_test_sample_np.shape={X_test_sample_np.shape}, len(feature_names)={len(feature_names)}")

        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values_pos_class, X_test_sample_np, feature_names=feature_names, plot_type="dot",
                          show=False)
        plt.title("SHAP Summary Plot (LinearSVC + Calibrated - Positive Class)")
        try:
            plt.tight_layout()
        except Exception:
            logging.warning("Could not apply tight_layout() to SHAP plot.")
        save_plot_to_gcs(fig_shap, gcs_bucket, f"{gcs_output_prefix}/plots/linearsvc_shap_summary.png")

        mean_abs_shap = np.mean(np.abs(shap_values_pos_class), axis=0)
        if len(feature_names) == len(mean_abs_shap):
            feature_importance = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
            feature_importance = feature_importance.sort_values('mean_abs_shap', ascending=False)
            logging.info("Top 10 features by Mean Absolute SHAP value:\n" + feature_importance.head(10).to_string())
        else:
            logging.error("Mismatch for feature importance table. mean_abs_shap length: %s, feature_names length: %s",
                          len(mean_abs_shap), len(feature_names))
            feature_importance = pd.DataFrame()

        return shap_values_pos_class, feature_importance.to_dict('records') if not feature_importance.empty else None

    except Exception as e:
        logging.error(f"Failed during SHAP analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

# save_model_to_gcs remains the same
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
    # *** CHANGED LOG MESSAGE ***
    logging.info("--- Starting LinearSVC (Calibrated) Training Pipeline ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')
    METADATA_URI = args.metadata_uri

    # --- 1. Load Metadata (No change) ---
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

    # --- 2. Load Processed Data (No change) ---
    logging.info("Loading processed data...")
    X_train = load_data_from_gcs(GCS_BUCKET, x_train_path, processed_feature_names)
    y_train_df = load_data_from_gcs(GCS_BUCKET, y_train_path)
    X_test = load_data_from_gcs(GCS_BUCKET, x_test_path, processed_feature_names)
    y_test_df = load_data_from_gcs(GCS_BUCKET, y_test_path)
    y_train = y_train_df.iloc[:, 0].values
    y_test = y_test_df.iloc[:, 0].values
    logging.info(f"Data loading complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


    # --- 3. Train LinearSVC Model with Calibration --- ## *** MODIFIED SECTION *** ##
    logging.info("--- Training LinearSVC Model with Calibration ---")
    # Increased max_iter for potentially large dataset, dual=False recommended when n_samples > n_features
    # Add class_weight='balanced' if input data might still be imbalanced despite SMOTE
    base_svm = LinearSVC(
        C=args.svm_c,
        loss='squared_hinge', # Generally preferred over hinge
        penalty='l2',
        dual="auto", # Changed from False to auto, let sklearn decide based on n_samples/n_features
        random_state=42,
        max_iter=args.svm_max_iter, # Use argument for max_iter
        verbose=1, # Show LinearSVC progress
        # class_weight='balanced' # Optional: Add if needed
    )

    # Wrap the base LinearSVC model with CalibratedClassifierCV
    # cv=3 for faster training than cv=5. 'sigmoid' is Platt scaling.
    calibrated_svm_model = CalibratedClassifierCV(
        base_svm,
        cv=3,
        method='sigmoid', # Can also try 'isotonic', might be better but needs more data
        n_jobs=-1 # Use all available CPU cores for cross-validation if possible
    )
    logging.info(f"Using LinearSVC wrapped with CalibratedClassifierCV(cv=3, method='sigmoid').")

    start_train_time = time.time()
    # Fit the calibrated model. This handles the internal cross-validation and fitting.
    calibrated_svm_model.fit(X_train, y_train)
    end_train_time = time.time()
    training_duration = end_train_time - start_train_time
    logging.info(f"LinearSVC + Calibration training completed in {training_duration:.2f} seconds.")
    # --- End Modified Section ---

    # --- 4. Predict on Test Set ---
    logging.info("Predicting on test set...")
    y_pred = calibrated_svm_model.predict(X_test) # Use calibrated model
    # Use predict_proba from the calibrated model
    y_pred_proba = None
    if hasattr(calibrated_svm_model, "predict_proba"):
         y_pred_proba = calibrated_svm_model.predict_proba(X_test)[:, 1] # Probability of positive class
    else:
         logging.error("Calibrated model unexpectedly lacks predict_proba method.")


    # --- 5. Evaluate Model ---
    # Pass y_pred_proba obtained from calibrated model
    metrics, conf_matrix = evaluate_model(y_test, y_pred, y_pred_proba)

    # --- 6. Generate and Save Plots ---
    logging.info("Generating evaluation plots...")
    X_test_eval = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    # Pass the calibrated model and a descriptive name
    plot_evaluation_charts(calibrated_svm_model, X_test_eval, y_test, conf_matrix, "LinearSVC (Calibrated)", GCS_BUCKET, GCS_OUTPUT_PREFIX)

    # --- 7. Perform SHAP Analysis ---
    shap_values = None
    shap_feature_importance = None
    if args.run_shap:
        # Pass the calibrated model for SHAP analysis
        shap_values, shap_feature_importance = perform_shap_analysis(
            calibrated_svm_model, X_train, X_test, processed_feature_names,
            GCS_BUCKET, GCS_OUTPUT_PREFIX, sample_size=args.shap_sample_size
        )
    else:
        logging.info("SHAP analysis skipped as per --run-shap flag.")

    # --- 8. Save Model ---
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/linearsvc_calibrated_model.joblib" # Updated filename
    save_model_to_gcs(calibrated_svm_model, GCS_BUCKET, model_blob_name) # Save the calibrated model

    # --- 9. Save Logs and Metrics ---
    logging.info("Saving logs and metrics...")
    # Create a dictionary of arguments passed, excluding kernel/gamma
    script_args = vars(args)
    script_args.pop('svm_kernel', None) # Remove kernel if it exists
    script_args.pop('svm_gamma', None) # Remove gamma if it exists

    log_summary = {
        "model_type": "LinearSVC (Calibrated)", # Updated model type
        "script_args": script_args,
        "metadata_source": METADATA_URI,
        "training_duration_seconds": training_duration,
        "evaluation_metrics": metrics,
        "shap_analysis_run": args.run_shap,
        "shap_top_features": shap_feature_importance,
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }

    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/linearsvc_training_log.json" # Updated filename
    try:
        # Use default=str for robustness if any non-serializable items exist
        log_string = json.dumps(log_summary, indent=4, default=str)
        gcs_utils.upload_string_to_gcs(GCS_BUCKET, log_string, log_blob_name, content_type='application/json')
        logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")
    except Exception as e:
        logging.error(f"Failed to save log summary: {e}")

    logging.info("--- LinearSVC (Calibrated) Training Pipeline Finished ---")


if __name__ == "__main__":
    # *** MODIFIED ARGPARSE ***
    parser = argparse.ArgumentParser(description="Train and evaluate a Calibrated LinearSVC model using data from GCS.")
    parser.add_argument("--gcs-bucket", type=str, required=True,
                        help="GCS bucket name for input metadata/data and output artifacts.")
    parser.add_argument("--metadata-uri", type=str, required=True,
                        help="GCS URI (gs://bucket/path/to/preprocessing_metadata.json) of the metadata file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True,
                        help="GCS prefix (folder path) within the bucket to save outputs (logs, plots, model) (e.g., 'fraud_detection/linearsvc_run_1').")
    # LinearSVC Hyperparameters
    parser.add_argument("--svm-c", type=float, default=1.0, help="LinearSVC regularization parameter C.")
    parser.add_argument("--svm-max-iter", type=int, default=2000, help="Maximum iterations for LinearSVC.") # Increased default
    # SHAP arguments
    parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis using KernelExplainer (can be slow).")
    parser.add_argument("--shap-sample-size", type=int, default=100, help="Number of samples for SHAP background and explanation.")

    args = parser.parse_args()
    main(args)