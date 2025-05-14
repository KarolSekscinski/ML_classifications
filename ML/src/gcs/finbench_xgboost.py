# xgboost_finbench_cf2.py
import argparse
import logging
import pandas as pd
import numpy as np
import xgboost as xgb  # Using XGBoost
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay  # Ensure these are imported
)
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
# import joblib # XGBoost has its own save/load, but joblib could be an alternative
import json
import os

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')

MODEL_NAME = "FinBench_XGBoost"  # Used for naming outputs


# --- Helper Functions (Reused and adapted from svm_finbench_cf2.py) ---

def load_data_from_gcs(gcs_bucket, gcs_path_in_metadata, feature_names_list=None):
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
        if feature_names_list is not None:
            if len(feature_names_list) == df.shape[1]:
                df.columns = feature_names_list
                logging.info(f"Assigned provided feature names. Columns: {list(df.columns[:5])}...")
            else:
                logging.error(
                    f"Feature name count ({len(feature_names_list)}) != column count ({df.shape[1]}) in data from {blob_name}. Using existing/generic names.")
        logging.info(f"Data loaded successfully from gs://{gcs_bucket}/{blob_name}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from gs://{gcs_bucket}/{blob_name}: {e}")
        raise


def save_plot_to_gcs(fig, gcs_bucket, gcs_blob_name):
    # (Identical to svm_finbench_cf2.py)
    logging.info(f"Saving plot to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        buf = BytesIO();
        fig.savefig(buf, format='png', bbox_inches='tight');
        buf.seek(0)
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, buf.read(), gcs_blob_name, content_type='image/png')
        logging.info(f"Plot successfully saved to gs://{gcs_bucket}/{gcs_blob_name}");
        plt.close(fig)
    except Exception as e:
        logging.error(f"ERROR saving plot to GCS ({gcs_blob_name}): {e}");
        plt.close(fig)


def evaluate_model(y_true, y_pred, y_pred_proba, dataset_name="Test Set"):
    # (Identical to svm_finbench_cf2.py)
    logging.info(f"--- Model Evaluation on {dataset_name} ---")
    accuracy = accuracy_score(y_true, y_pred);
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0);
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = -1.0;
    pr_auc = -1.0
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba); pr_auc = average_precision_score(y_true, y_pred_proba)
        except ValueError as e:
            logging.warning(f"Could not calculate AUC scores on {dataset_name}: {e}")
    else:
        logging.warning(f"Probability scores not available for {dataset_name}. AUCs not calculated.")
    cm = confusion_matrix(y_true, y_pred)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "roc_auc": roc_auc,
               "pr_auc": pr_auc, "confusion_matrix": cm.tolist()}
    logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    if y_pred_proba is not None: logging.info(f"AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    return metrics, cm


def plot_evaluation_charts(model, X_data, y_data, cm, plot_suffix, model_display_name, gcs_bucket, gcs_output_prefix):
    # (Identical to svm_finbench_cf2.py, ensures model has predict_proba for plots)
    try:  # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5));
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_display_name} Confusion Matrix ({plot_suffix})');
        ax_cm.set_xlabel('Predicted');
        ax_cm.set_ylabel('True')
        save_plot_to_gcs(fig_cm, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_confusion_matrix_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed CM plot for {plot_suffix}: {e}")
    if not hasattr(model, "predict_proba"): logging.warning(
        f"No predict_proba for {model_display_name}. Skipping ROC/PR for {plot_suffix}."); return
    try:  # ROC Curve
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6));
        RocCurveDisplay.from_estimator(model, X_data, y_data, ax=ax_roc, name=model_display_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC=0.5)');
        ax_roc.set_title(f'{model_display_name} ROC ({plot_suffix})');
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_roc_curve_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed ROC plot for {plot_suffix}: {e}")
    try:  # PR Curve
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6));
        PrecisionRecallDisplay.from_estimator(model, X_data, y_data, ax=ax_pr, name=model_display_name)
        ax_pr.set_title(f'{model_display_name} PR Curve ({plot_suffix})')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_pr_curve_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed PR plot for {plot_suffix}: {e}")


def perform_shap_analysis_xgb(model, X_explain_data, feature_names_list, gcs_bucket, gcs_output_prefix):
    """Performs SHAP analysis for XGBoost using TreeExplainer."""
    logging.info(f"--- SHAP Analysis ({MODEL_NAME} - TreeExplainer) ---")
    try:
        # X_explain_data can be Pandas DataFrame or NumPy array for TreeExplainer
        # If it's numpy, feature_names_list will be used by summary_plot.
        # If it's DataFrame, summary_plot uses its column names.
        # Ensure feature_names_list is correctly passed if X_explain_data is numpy.

        X_explain_df_for_shap = None
        if isinstance(X_explain_data, pd.DataFrame):
            X_explain_df_for_shap = X_explain_data
            if list(X_explain_df_for_shap.columns) != feature_names_list:
                logging.warning(
                    "X_explain_data DataFrame columns mismatch feature_names_list. Using DataFrame columns for SHAP.")
                feature_names_list = X_explain_df_for_shap.columns.tolist()  # Prioritize DF columns if available
        elif isinstance(X_explain_data, np.ndarray):
            if X_explain_data.shape[1] == len(feature_names_list):
                X_explain_df_for_shap = pd.DataFrame(X_explain_data, columns=feature_names_list)  # For plotting ease
            else:
                logging.error(
                    f"NumPy X_explain_data feature count ({X_explain_data.shape[1]}) != feature_names_list count ({len(feature_names_list)}).")
                return None, None
        else:
            logging.error(f"X_explain_data is not a DataFrame or NumPy array (type: {type(X_explain_data)}).")
            return None, None

        start_shap_time = time.time()
        logging.info("Initializing SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)  # For XGBoost, TreeExplainer is efficient

        logging.info(f"Calculating SHAP values for {X_explain_df_for_shap.shape[0]} samples...")
        shap_values_output = explainer.shap_values(X_explain_df_for_shap)  # Explain on DataFrame
        end_shap_time = time.time()
        logging.info(f"SHAP values calculated in {end_shap_time - start_shap_time:.2f} seconds.")

        # For binary classification, shap_values might be for the positive class directly with XGB
        # or a list of two arrays if explainer.model.objective is 'binary:logistic' and output_margin=True
        # XGBClassifier's predict_proba usually leads to shap_values for the positive class by default
        # Or, if it returns two arrays, shap_values_output[1] is typically for the positive class.
        shap_values_for_plot = shap_values_output
        if isinstance(shap_values_output, list) and len(shap_values_output) == 2:
            logging.info("SHAP output is a list of 2 arrays; using index 1 for positive class SHAP values.")
            shap_values_for_plot = shap_values_output[1]
        elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim != 2:
            logging.error(f"Unexpected SHAP values array dimension: {shap_values_output.ndim}. Expected 2D.")
            return None, None

        fig_shap, ax_shap = plt.subplots();
        shap.summary_plot(shap_values_for_plot, X_explain_df_for_shap, plot_type="dot", show=False)
        plt.title(f"SHAP Summary Plot ({MODEL_NAME})");
        try:
            plt.tight_layout()
        except:
            logging.warning("Could not apply tight_layout() to SHAP plot.")
        save_plot_to_gcs(fig_shap, gcs_bucket, f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_shap_summary.png")

        mean_abs_shap = np.mean(np.abs(shap_values_for_plot), axis=0)
        if len(feature_names_list) == len(mean_abs_shap):
            feature_importance_df = pd.DataFrame({'feature': feature_names_list, 'mean_abs_shap': mean_abs_shap})
            feature_importance_df = feature_importance_df.sort_values('mean_abs_shap', ascending=False)
            logging.info(f"Top 10 features by Mean Absolute SHAP value:\n{feature_importance_df.head(10).to_string()}")
        else:
            logging.error(
                f"Mismatch for feature importance. mean_abs_shap: {len(mean_abs_shap)}, names: {len(feature_names_list)}")
            feature_importance_df = pd.DataFrame()

        return shap_values_output, feature_importance_df.to_dict('records') if not feature_importance_df.empty else None
    except Exception as e:
        logging.error(f"Failed during SHAP analysis: {e}");
        import traceback;
        logging.error(traceback.format_exc())
        return None, None


def save_model_to_gcs_xgb(model, gcs_bucket, gcs_blob_name):
    """Saves an XGBoost model to GCS using its native save_model method."""
    logging.info(f"Saving XGBoost model to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    temp_model_path = "temp_xgb_model.json"  # XGBoost can save as json, ubj (binary), or old binary format
    try:
        model.save_model(temp_model_path)  # Saves in JSON format by default if extension is .json
        with open(temp_model_path, 'rb') as f:
            model_bytes = f.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/json')
        logging.info(f"XGBoost model successfully saved to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
        os.remove(temp_model_path)
    except Exception as e:
        logging.error(f"ERROR saving XGBoost model to GCS ({gcs_blob_name}): {e}")
        if os.path.exists(temp_model_path): os.remove(temp_model_path)


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
        data_paths = metadata.get("gcs_paths", {});
        path_x_train = data_paths.get("X_train_resampled");
        path_y_train = data_paths.get("y_train_resampled")
        path_x_val = data_paths.get("X_val_processed");
        path_y_val = data_paths.get("y_val")
        path_x_test = data_paths.get("X_test_processed");
        path_y_test = data_paths.get("y_test")
        if not all([path_x_train, path_y_train, path_x_val, path_y_val, path_x_test, path_y_test]):
            raise ValueError("One or more required data paths missing in OHE metadata.")
    except Exception as e:
        logging.error(f"Failed to load/parse OHE metadata from {METADATA_URI}: {e}"); return

    # 2. Load Processed Data
    try:
        logging.info("Loading processed OHE data from GCS...")
        X_train = load_data_from_gcs(GCS_BUCKET, path_x_train, processed_feature_names)
        y_train_df = load_data_from_gcs(GCS_BUCKET, path_y_train);
        y_train = y_train_df.iloc[:, 0].values
        X_val = load_data_from_gcs(GCS_BUCKET, path_x_val, processed_feature_names)
        y_val_df = load_data_from_gcs(GCS_BUCKET, path_y_val);
        y_val = y_val_df.iloc[:, 0].values
        X_test = load_data_from_gcs(GCS_BUCKET, path_x_test, processed_feature_names)
        y_test_df = load_data_from_gcs(GCS_BUCKET, path_y_test);
        y_test = y_test_df.iloc[:, 0].values
    except Exception as e:
        logging.error(f"Failed data loading from GCS: {e}"); return
    logging.info(f"Data loading complete. X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # 3. Train XGBoost Model
    logging.info(f"--- Training {MODEL_NAME} Model ---")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric=args.xgb_eval_metric, use_label_encoder=False,
        # use_label_encoder deprecated
        n_estimators=args.xgb_n_estimators, learning_rate=args.xgb_learning_rate,
        max_depth=args.xgb_max_depth, subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree, gamma=args.xgb_gamma,
        random_state=42, n_jobs=-1  # Use all cores
    )
    start_train_time = time.time()
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],  # Use validation set for early stopping
                  early_stopping_rounds=args.xgb_early_stopping_rounds,
                  verbose=args.xgb_verbose)
    end_train_time = time.time()
    training_duration = end_train_time - start_train_time
    logging.info(f"{MODEL_NAME} training completed in {training_duration:.2f} seconds.")
    if hasattr(xgb_model, 'best_iteration') and xgb_model.best_iteration is not None:  # Check for early stopping
        logging.info(
            f"Best iteration: {xgb_model.best_iteration}, Best score ({args.xgb_eval_metric}): {xgb_model.best_score:.4f}")
    else:  # If early stopping wasn't triggered or not used
        logging.info("Early stopping not triggered or not used. Model trained for full n_estimators.")
        # Set best_iteration to n_estimators for consistent prediction if not available
        xgb_model.best_iteration = args.xgb_n_estimators - 1  # 0-indexed

    # 4. Predict & Evaluate on Validation Set (using best iteration)
    logging.info("Predicting on Validation set...")
    y_val_pred_proba = xgb_model.predict_proba(X_val, iteration_range=(0, xgb_model.best_iteration + 1))[:, 1]
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)
    val_metrics, val_cm = evaluate_model(y_val, y_val_pred, y_val_pred_proba, dataset_name="Validation Set")
    plot_evaluation_charts(xgb_model, X_val, y_val, val_cm, "Validation", MODEL_NAME, GCS_BUCKET, GCS_OUTPUT_PREFIX)

    # 5. Predict & Evaluate on Test Set (using best iteration)
    logging.info("Predicting on Test set...")
    y_test_pred_proba = xgb_model.predict_proba(X_test, iteration_range=(0, xgb_model.best_iteration + 1))[:, 1]
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    test_metrics, test_cm = evaluate_model(y_test, y_test_pred, y_test_pred_proba, dataset_name="Test Set")
    plot_evaluation_charts(xgb_model, X_test, y_test, test_cm, "Test", MODEL_NAME, GCS_BUCKET, GCS_OUTPUT_PREFIX)

    # 6. Perform SHAP Analysis (on Test Set)
    shap_feature_importance = None
    if args.run_shap:
        # Pass DataFrame X_test to SHAP for feature names, or X_test (numpy) + processed_feature_names
        shap_values, shap_feature_importance = perform_shap_analysis_xgb(
            xgb_model, X_test, processed_feature_names, GCS_BUCKET, GCS_OUTPUT_PREFIX
        )
    else:
        logging.info("SHAP analysis skipped.")

    # 7. Save Model
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/{MODEL_NAME.lower()}_model.json"  # Save as JSON
    save_model_to_gcs_xgb(xgb_model, GCS_BUCKET, model_blob_name)

    # 8. Save Logs and Metrics
    logging.info("Saving logs and metrics...")
    log_summary = {
        "model_type": MODEL_NAME, "script_args": vars(args), "metadata_source": METADATA_URI,
        "training_duration_seconds": training_duration,
        "best_iteration": xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else None,
        "best_eval_score": xgb_model.best_score if hasattr(xgb_model, 'best_score') else None,
        "validation_set_evaluation": val_metrics, "test_set_evaluation": test_metrics,
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
                        help="GCS prefix for saving outputs (e.g., 'finbench_cf2/results/xgb_run1').")
    # XGBoost Hyperparameters
    parser.add_argument("--xgb-n-estimators", type=int, default=500, help="Number of boosting rounds.")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--xgb-max-depth", type=int, default=5, help="Maximum depth of a tree.")
    parser.add_argument("--xgb-subsample", type=float, default=0.8, help="Subsample ratio of the training instance.")
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8,
                        help="Subsample ratio of columns when constructing each tree.")
    parser.add_argument("--xgb-gamma", type=float, default=0.1,
                        help="Minimum loss reduction required to make a further partition.")
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=20,
                        help="Activates early stopping. Validation error needs to decrease at least every <N> round(s) to continue training.")
    parser.add_argument("--xgb-eval-metric", type=str, default="aucpr",
                        help="Evaluation metric for validation data (e.g., 'logloss', 'auc', 'aucpr').")
    parser.add_argument("--xgb-verbose", type=int, default=100,
                        help="Verbosity of printing messages during training (e.g., 0 for silent, 1 for progress, N for every N iterations). Use True/False or int.")

    # SHAP arguments
    parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis.")
    # SHAP sample size is less critical for TreeExplainer with XGBoost, but can be added if needed for consistency.
    args = parser.parse_args()
    main(args)