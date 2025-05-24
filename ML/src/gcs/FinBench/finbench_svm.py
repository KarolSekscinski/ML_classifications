# svm_finbench_cf2.py
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.svm import SVC  # Using kernelized SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay  # Ensure these are imported
)
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import joblib
import json

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
from ML_classifications.ML.src.gcs import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')  # Use a seaborn style for plots

MODEL_NAME = "FinBench_SVC"  # Used for naming outputs


# --- Helper Functions (Reused and adapted) ---

def load_data_from_gcs(gcs_bucket, gcs_path_in_metadata, feature_names_list=None):
    """
    Loads CSV data from GCS path specified in metadata into a pandas DataFrame.
    gcs_path_in_metadata should be the full gs://bucket/path/to/file.csv from metadata.
    """
    if not gcs_path_in_metadata or not gcs_path_in_metadata.startswith(f"gs://{gcs_bucket}/"):
        logging.error(f"Invalid GCS path from metadata: {gcs_path_in_metadata} for bucket {gcs_bucket}")
        raise ValueError(f"Invalid GCS path: {gcs_path_in_metadata}")

    blob_name = gcs_path_in_metadata.replace(f"gs://{gcs_bucket}/", "")
    logging.info(f"Loading data from: gs://{gcs_bucket}/{blob_name}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, blob_name)
        if data_bytes is None:
            raise FileNotFoundError(f"Could not download GCS file: gs://{gcs_bucket}/{blob_name}")
        df = pd.read_csv(BytesIO(data_bytes))

        # If feature_names_list are provided (for X data), assign them if counts match
        if feature_names_list is not None:
            if len(feature_names_list) == df.shape[1]:
                df.columns = feature_names_list
                logging.info(f"Assigned provided feature names. Columns: {list(df.columns[:5])}...")  # Log first few
            else:
                logging.error(
                    f"Feature name count ({len(feature_names_list)}) does not match column count ({df.shape[1]}) in data from {blob_name}. Using existing/generic column names.")

        logging.info(f"Data loaded successfully from gs://{gcs_bucket}/{blob_name}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from gs://{gcs_bucket}/{blob_name}: {e}")
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


def evaluate_model(y_true, y_pred, y_pred_proba, dataset_name="Test Set"):
    logging.info(f"--- Model Evaluation on {dataset_name} ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = -1.0
    pr_auc = -1.0
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
        except ValueError as e:
            logging.warning(f"Could not calculate AUC scores on {dataset_name}: {e}")
    else:
        logging.warning(f"Probability scores not available for {dataset_name}. AUCs not calculated.")

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


def plot_evaluation_charts(model, X_data, y_data, cm, plot_suffix, model_display_name, gcs_bucket, gcs_output_prefix):
    # 1. Confusion Matrix Plot
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_display_name} Confusion Matrix ({plot_suffix})')
        ax_cm.set_xlabel('Predicted Label');
        ax_cm.set_ylabel('True Label')
        save_plot_to_gcs(fig_cm, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_confusion_matrix_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed to generate/save Confusion Matrix plot for {plot_suffix}: {e}")

    if not hasattr(model, "predict_proba"):
        logging.warning(f"Model does not have predict_proba. Skipping ROC/PR plots for {plot_suffix}.")
        return
    # 2. ROC Curve Plot
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_estimator(model, X_data, y_data, ax=ax_roc, name=model_display_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
        ax_roc.set_title(f'{model_display_name} ROC Curve ({plot_suffix})');
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_roc_curve_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed to generate/save ROC Curve plot for {plot_suffix}: {e}")
    # 3. Precision-Recall Curve Plot
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_estimator(model, X_data, y_data, ax=ax_pr, name=model_display_name)
        ax_pr.set_title(f'{model_display_name} PR Curve ({plot_suffix})')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_pr_curve_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed to generate/save PR Curve plot for {plot_suffix}: {e}")


def perform_shap_analysis(model_with_proba, X_train_data, X_explain_data, feature_names_list, gcs_bucket,
                          gcs_output_prefix, sample_size=100):
    logging.warning(f"--- SHAP Analysis ({MODEL_NAME} - KernelExplainer) ---")
    logging.warning(
        f"KernelExplainer can be very slow. Using background sample size {min(sample_size, X_train_data.shape[0])} and explaining {min(sample_size, X_explain_data.shape[0])} samples.")
    try:
        # Ensure data is NumPy for SHAP KernelExplainer
        X_train_np = X_train_data.values if isinstance(X_train_data, pd.DataFrame) else X_train_data
        X_explain_np = X_explain_data.values if isinstance(X_explain_data, pd.DataFrame) else X_explain_data

        # Validate feature_names_list
        if not (isinstance(feature_names_list, list) and all(isinstance(fn, str) for fn in feature_names_list)):
            logging.error("feature_names_list is not a list of strings. SHAP plots might be incorrect.")
            # Attempt to use columns from DataFrame if possible, otherwise generic names
            if isinstance(X_explain_data, pd.DataFrame):
                feature_names_list = X_explain_data.columns.tolist()
            else:
                feature_names_list = [f"feature_{i}" for i in range(X_explain_np.shape[1])]

        if X_train_np.shape[1] != len(feature_names_list) or X_explain_np.shape[1] != len(feature_names_list):
            logging.error(
                f"Feature count mismatch. X_train_np: {X_train_np.shape[1]}, X_explain_np: {X_explain_np.shape[1]}, feature_names: {len(feature_names_list)}")
            return None, None

        X_train_summary_np = shap.sample(X_train_np, min(sample_size, X_train_np.shape[0]), random_state=42)
        X_explain_sample_np = shap.sample(X_explain_np, min(sample_size, X_explain_np.shape[0]), random_state=43)

        if not hasattr(model_with_proba, "predict_proba"):
            logging.error("Model for SHAP does not have predict_proba. Cannot compute SHAP values.")
            return None, None

        start_shap_time = time.time()
        logging.info("Initializing SHAP KernelExplainer...")
        explainer = shap.KernelExplainer(model_with_proba.predict_proba, X_train_summary_np)
        logging.info(f"Calculating SHAP values for {X_explain_sample_np.shape[0]} samples...")
        shap_values_output = explainer.shap_values(X_explain_sample_np)  # Explain on NumPy data
        end_shap_time = time.time()
        logging.info(f"SHAP values calculated in {end_shap_time - start_shap_time:.2f} seconds.")

        shap_values_pos_class = None
        if isinstance(shap_values_output, list) and len(shap_values_output) == 2:
            shap_values_pos_class = shap_values_output[1]
        elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 2:
            shap_values_pos_class = shap_values_output  # If KernelExplainer returns single array for positive class
        else:
            logging.error(f"Unexpected SHAP values output structure. Type: {type(shap_values_output)}")
            return None, None

        if shap_values_pos_class.shape[1] != len(feature_names_list):
            logging.error(
                f"SHAP values feature count ({shap_values_pos_class.shape[1]}) != feature_names count ({len(feature_names_list)}). Plotting will fail.")
            return None, None

        fig_shap, ax_shap = plt.subplots();
        shap.summary_plot(shap_values_pos_class, X_explain_sample_np, feature_names=feature_names_list, plot_type="dot",
                          show=False)
        plt.title(f"SHAP Summary Plot ({MODEL_NAME} - Positive Class)");
        try:
            plt.tight_layout()
        except:
            logging.warning("Could not apply tight_layout() to SHAP plot.")
        save_plot_to_gcs(fig_shap, gcs_bucket, f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_shap_summary.png")

        mean_abs_shap = np.mean(np.abs(shap_values_pos_class), axis=0)
        feature_importance_df = pd.DataFrame({'feature': feature_names_list, 'mean_abs_shap': mean_abs_shap})
        feature_importance_df = feature_importance_df.sort_values('mean_abs_shap', ascending=False)
        logging.info(f"Top 10 features by Mean Absolute SHAP value:\n{feature_importance_df.head(10).to_string()}")
        return shap_values_pos_class, feature_importance_df.to_dict('records')
    except Exception as e:
        logging.error(f"Failed during SHAP analysis: {e}");
        import traceback;
        logging.error(traceback.format_exc())
        return None, None


def save_model_to_gcs(model, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving model to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        with BytesIO() as buf:
            joblib.dump(model, buf); buf.seek(0); model_bytes = buf.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model successfully saved to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving model to GCS ({gcs_blob_name}): {e}")


# --- Main Execution ---
def main(args):
    logging.info(f"--- Starting {MODEL_NAME} Training Pipeline for FinBench cf2 ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')
    METADATA_URI = args.metadata_uri

    # 1. Load Metadata
    logging.info(f"Loading OHE metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata_str = gcs_utils.download_blob_as_string(GCS_BUCKET, metadata_blob_name)
        if metadata_str is None: raise FileNotFoundError("Could not download metadata file.")
        metadata = json.loads(metadata_str)
        logging.info("OHE Metadata loaded successfully.")

        processed_feature_names = metadata.get("processed_feature_names_list")
        if not processed_feature_names: raise ValueError(
            "Processed feature names list ('processed_feature_names_list') not found in metadata.")

        data_paths = metadata.get("gcs_paths", {})
        path_x_train = data_paths.get("X_train_resampled")
        path_y_train = data_paths.get("y_train_resampled")
        path_x_val = data_paths.get("X_val_processed")
        path_y_val = data_paths.get("y_val")
        path_x_test = data_paths.get("X_test_processed")
        path_y_test = data_paths.get("y_test")

        if not all([path_x_train, path_y_train, path_x_val, path_y_val, path_x_test, path_y_test]):
            raise ValueError("One or more required data paths are missing in OHE metadata.")
    except Exception as e:
        logging.error(f"Failed to load or parse OHE metadata from {METADATA_URI}: {e}");
        return

    # 2. Load Processed Data
    try:
        logging.info("Loading processed OHE data from GCS...")
        X_train = load_data_from_gcs(GCS_BUCKET, path_x_train, processed_feature_names)
        y_train_df = load_data_from_gcs(GCS_BUCKET, path_y_train)
        X_val = load_data_from_gcs(GCS_BUCKET, path_x_val, processed_feature_names)
        y_val_df = load_data_from_gcs(GCS_BUCKET, path_y_val)
        X_test = load_data_from_gcs(GCS_BUCKET, path_x_test, processed_feature_names)
        y_test_df = load_data_from_gcs(GCS_BUCKET, path_y_test)
    except Exception as e:
        logging.error(f"Failed during data loading from GCS: {e}");
        return

    y_train = y_train_df.iloc[:, 0].values
    y_val = y_val_df.iloc[:, 0].values
    y_test = y_test_df.iloc[:, 0].values
    logging.info(f"Data loading complete. X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # 3. Train SVC Model
    logging.info(f"--- Training {MODEL_NAME} Model ---")
    svm_model = SVC(
        C=args.svm_c, kernel=args.svm_kernel, gamma=args.svm_gamma,
        probability=True,  # Crucial for AUC and SHAP KernelExplainer
        random_state=42, verbose=args.svm_verbose
    )
    start_train_time = time.time()
    svm_model.fit(X_train, y_train)
    end_train_time = time.time()
    training_duration = end_train_time - start_train_time
    logging.info(f"{MODEL_NAME} training completed in {training_duration:.2f} seconds.")

    # 4. Predict & Evaluate on Validation Set
    logging.info("Predicting on Validation set...")
    y_val_pred = svm_model.predict(X_val)
    y_val_pred_proba = svm_model.predict_proba(X_val)[:, 1] if hasattr(svm_model, "predict_proba") else None
    val_metrics, val_cm = evaluate_model(y_val, y_val_pred, y_val_pred_proba, dataset_name="Validation Set")
    plot_evaluation_charts(svm_model, X_val, y_val, val_cm, "Validation", MODEL_NAME, GCS_BUCKET, GCS_OUTPUT_PREFIX)

    # 5. Predict & Evaluate on Test Set
    logging.info("Predicting on Test set...")
    y_test_pred = svm_model.predict(X_test)
    y_test_pred_proba = svm_model.predict_proba(X_test)[:, 1] if hasattr(svm_model, "predict_proba") else None
    test_metrics, test_cm = evaluate_model(y_test, y_test_pred, y_test_pred_proba, dataset_name="Test Set")
    plot_evaluation_charts(svm_model, X_test, y_test, test_cm, "Test", MODEL_NAME, GCS_BUCKET, GCS_OUTPUT_PREFIX)

    # 6. Perform SHAP Analysis (on Test Set, using Training Set for background)
    shap_feature_importance = None
    if args.run_shap:
        shap_values, shap_feature_importance = perform_shap_analysis(
            svm_model, X_train, X_test, processed_feature_names,  # Pass list of names
            GCS_BUCKET, GCS_OUTPUT_PREFIX, sample_size=args.shap_sample_size
        )
    else:
        logging.info("SHAP analysis skipped.")

    # 7. Save Model
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/{MODEL_NAME.lower()}_model.joblib"
    save_model_to_gcs(svm_model, GCS_BUCKET, model_blob_name)

    # 8. Save Logs and Metrics
    logging.info("Saving logs and metrics...")
    log_summary = {
        "model_type": MODEL_NAME, "script_args": vars(args), "metadata_source": METADATA_URI,
        "training_duration_seconds": training_duration,
        "validation_set_evaluation": val_metrics,
        "test_set_evaluation": test_metrics,
        "shap_analysis_run": args.run_shap, "shap_top_features": shap_feature_importance,
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }
    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/{MODEL_NAME.lower()}_training_log.json"
    try:
        log_string = json.dumps(log_summary, indent=4, default=str)
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
    parser.add_argument("--gcs-output-prefix", type=str, required=True,
                        help="GCS prefix for saving outputs (e.g., 'finbench_cf2/results/svc_run1').")
    # SVM Hyperparameters
    parser.add_argument("--svm-c", type=float, default=1.0, help="SVC regularization parameter C.")
    parser.add_argument("--svm-kernel", type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help="SVC kernel type.")
    parser.add_argument("--svm-gamma", type=str, default='scale',
                        help="SVC kernel coefficient gamma ('scale', 'auto', or float).")
    parser.add_argument("--svm-verbose", action='store_true', help="Enable verbose output for SVC training.")
    # SHAP arguments
    parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis (can be very slow).")
    parser.add_argument("--shap-sample-size", type=int, default=10,
                        help="Number of samples for SHAP background and explanation.")
    args = parser.parse_args()
    main(args)