# xgboost_pipeline.py
import argparse
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
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
import json
import os

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')

# --- Helper Functions (Identical to SVM script - reuse or place in a shared utils file) ---

def load_data_from_gcs(gcs_bucket, gcs_path, feature_names=None):
    """Loads CSV data from GCS into a pandas DataFrame."""
    logging.info(f"Loading data from: gs://{gcs_bucket}/{gcs_path}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, gcs_path)
        if data_bytes is None:
            raise FileNotFoundError(f"Could not download GCS file: gs://{gcs_bucket}/{gcs_path}")
        df = pd.read_csv(BytesIO(data_bytes))
        # Assign feature names if provided
        if feature_names is not None and not df.columns.equals(pd.Index(feature_names)):
             if len(feature_names) == len(df.columns):
                 logging.warning(f"Assigning provided feature names to loaded data from {gcs_path}.")
                 df.columns = feature_names
             else:
                 logging.error(f"Feature name count ({len(feature_names)}) does not match column count ({len(df.columns)}) in {gcs_path}. Cannot assign names.")
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
        "confusion_matrix": cm.tolist()
    }

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall (Sensitivity): {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"AUC-ROC: {roc_auc:.4f}")
    logging.info(f"AUC-PR: {pr_auc:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")

    return metrics, cm

def plot_evaluation_charts(estimator, X_test, y_test, cm, model_name, gcs_bucket, gcs_output_prefix):
    """Generates and saves Confusion Matrix, ROC, and PR curve plots."""
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

    # 2. ROC Curve Plot
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        # Use from_estimator if it's compatible (sklearn interface) or manually plot for pure xgboost
        if hasattr(estimator, 'predict_proba'): # Check if it behaves like sklearn estimator
             RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax_roc, name=model_name)
        else: # Manual plot if needed (less likely with XGBClassifier)
             fpr, tpr, _ = roc_curve(y_test, estimator.predict(xgb.DMatrix(X_test))) # Example if using predict directly
             roc_auc = auc(fpr, tpr)
             ax_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
        ax_roc.set_title(f'{model_name} Receiver Operating Characteristic (ROC) Curve')
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_roc_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save ROC Curve plot: {e}")

    # 3. Precision-Recall Curve Plot
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        if hasattr(estimator, 'predict_proba'):
             PrecisionRecallDisplay.from_estimator(estimator, X_test, y_test, ax=ax_pr, name=model_name)
        else: # Manual plot if needed
            precision, recall, _ = precision_recall_curve(y_test, estimator.predict(xgb.DMatrix(X_test)))
            pr_auc = average_precision_score(y_test, estimator.predict(xgb.DMatrix(X_test)))
            ax_pr.plot(recall, precision, label=f'{model_name} (AP = {pr_auc:.2f})')

        ax_pr.set_title(f'{model_name} Precision-Recall (PR) Curve')
        save_plot_to_gcs(fig_pr, gcs_bucket, f"{gcs_output_prefix}/plots/{model_name.lower()}_pr_curve.png")
    except Exception as e:
        logging.error(f"Failed to generate or save PR Curve plot: {e}")


def perform_shap_analysis(model, X_test, feature_names, gcs_bucket, gcs_output_prefix):
    """Performs SHAP analysis using TreeExplainer and saves summary plot."""
    logging.info("--- SHAP Analysis (XGBoost - TreeExplainer) ---")
    try:
        start_shap_time = time.time()
        logging.info("Initializing SHAP TreeExplainer...")
        # Use the model directly with TreeExplainer
        explainer = shap.TreeExplainer(model)

        # Ensure X_test has feature names if it's not already a DataFrame
        if not isinstance(X_test, pd.DataFrame):
             X_test_df = pd.DataFrame(X_test, columns=feature_names)
        else:
             X_test_df = X_test

        logging.info(f"Calculating SHAP values for {X_test_df.shape[0]} test samples...")
        shap_values = explainer.shap_values(X_test_df) # For XGBoost, often directly gives values for the positive class output
        end_shap_time = time.time()
        logging.info(f"SHAP values calculated in {end_shap_time - start_shap_time:.2f} seconds.")

        # Check SHAP value structure (sometimes it's just one array, sometimes two for binary)
        # If explainer.model.objective starts with 'binary:', shap_values are usually for the positive class.
        shap_values_plot = shap_values

        # Generate and save summary plot
        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values_plot, X_test_df, plot_type="dot", show=False)
        plt.title("SHAP Summary Plot (XGBoost)")
        save_plot_to_gcs(fig_shap, gcs_bucket, f"{gcs_output_prefix}/plots/xgboost_shap_summary.png")

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values_plot), axis=0)
        feature_importance = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
        feature_importance = feature_importance.sort_values('mean_abs_shap', ascending=False)
        logging.info("Top 10 features by Mean Absolute SHAP value:\n" + feature_importance.head(10).to_string())

        return shap_values, feature_importance.to_dict('records')

    except Exception as e:
        logging.error(f"Failed during SHAP analysis: {e}")
        return None, None

def save_model_to_gcs(model, gcs_bucket, gcs_blob_name):
    """Saves an XGBoost model object to GCS."""
    logging.info(f"Saving model object to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    # XGBoost models have a native save method that's generally preferred
    temp_model_path = "temp_xgb_model.json" # Save locally first, then upload
    try:
        model.save_model(temp_model_path)
        with open(temp_model_path, 'rb') as f:
             model_bytes = f.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/json') # Save as json
        logging.info(f"Model successfully saved to gs://{gcs_bucket}/{gcs_blob_name}")
        os.remove(temp_model_path) # Clean up local temp file
    except Exception as e:
        logging.error(f"ERROR saving model to GCS ({gcs_blob_name}): {e}")
        if os.path.exists(temp_model_path):
             os.remove(temp_model_path) # Clean up even on error

# --- Main Execution ---

def main(args):
    """Main training and evaluation pipeline function."""
    logging.info("--- Starting XGBoost Training Pipeline ---")
    GCS_BUCKET = args.gcs_bucket
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
        processed_feature_names = metadata.get("processed_feature_names")
        if not processed_feature_names:
             logging.error("Processed feature names not found in metadata.")
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

    y_train = y_train_df.iloc[:, 0].values
    y_test = y_test_df.iloc[:, 0].values

    # --- 3. Train XGBoost Model ---
    logging.info("--- Training XGBoost Model ---")
    # Use LabelEncoder is deprecated. Ensure target is 0/1.
    # Using AUC-PR as eval_metric as it's often better for imbalanced data.
    # Early stopping helps prevent overfitting and speeds up training.
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr', # Area under PR curve
        use_label_encoder=False, # Deprecated
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_learning_rate,
        max_depth=args.xgb_max_depth,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        gamma=args.xgb_gamma, # Min loss reduction to make split
        random_state=42,
        n_jobs=-1 # Use all available CPU cores
    )

    start_train_time = time.time()
    # Use test set for early stopping evaluation. For rigor, a separate validation set is better.
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  early_stopping_rounds=args.xgb_early_stopping_rounds,
                  verbose=True) # Show training progress and eval metric
    end_train_time = time.time()
    training_duration = end_train_time - start_train_time
    logging.info(f"XGBoost training completed in {training_duration:.2f} seconds.")
    logging.info(f"Best iteration: {xgb_model.best_iteration}, Best AUC-PR: {xgb_model.best_score}")


    # --- 4. Predict on Test Set ---
    logging.info("Predicting on test set using best iteration...")
    # Predict using the best iteration found during early stopping
    y_pred_proba = xgb_model.predict_proba(X_test, iteration_range=(0, xgb_model.best_iteration + 1))[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int) # Threshold probabilities at 0.5

    # --- 5. Evaluate Model ---
    metrics, conf_matrix = evaluate_model(y_test, y_pred, y_pred_proba)

    # --- 6. Generate and Save Plots ---
    logging.info("Generating evaluation plots...")
    # Pass the fitted model directly
    plot_evaluation_charts(xgb_model, X_test, y_test, conf_matrix, 'XGBoost', GCS_BUCKET, GCS_OUTPUT_PREFIX)

    # --- 7. Perform SHAP Analysis ---
    shap_values = None
    shap_feature_importance = None
    if args.run_shap:
        # Pass the fitted XGBoost model and test data (preferably as DataFrame)
        shap_values, shap_feature_importance = perform_shap_analysis(
            xgb_model, X_test, processed_feature_names,
            GCS_BUCKET, GCS_OUTPUT_PREFIX
        )
    else:
        logging.info("SHAP analysis skipped as per --run-shap flag.")

    # --- 8. Save Model ---
    # Use XGBoost's native save method
    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/xgboost_model.json" # Save in json format
    save_model_to_gcs(xgb_model, GCS_BUCKET, model_blob_name)

    # --- 9. Save Logs and Metrics ---
    logging.info("Saving logs and metrics...")
    log_summary = {
        "model_type": "XGBoost",
        "script_args": vars(args),
        "metadata_source": METADATA_URI,
        "training_duration_seconds": training_duration,
        "best_iteration": xgb_model.best_iteration,
        "best_eval_score_aucpr": xgb_model.best_score,
        "evaluation_metrics": metrics,
        "shap_analysis_run": args.run_shap,
        "shap_top_features": shap_feature_importance,
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }

    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/xgboost_training_log.json"
    try:
        log_string = json.dumps(log_summary, indent=4)
        gcs_utils.upload_string_to_gcs(GCS_BUCKET, log_string, log_blob_name, content_type='application/json')
        logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")
    except Exception as e:
        logging.error(f"Failed to save log summary: {e}")

    logging.info("--- XGBoost Training Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an XGBoost model using data from GCS.")
    parser.add_argument("--gcs-bucket", type=str, required=True,
                        help="GCS bucket name for input metadata/data and output artifacts.")
    parser.add_argument("--metadata-uri", type=str, required=True,
                        help="GCS URI (gs://bucket/path/to/preprocessing_metadata.json) of the metadata file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True,
                        help="GCS prefix (folder path) within the bucket to save outputs (logs, plots, model) (e.g., 'fraud_detection/xgb_run_1').")
    # Optional XGBoost Hyperparameters
    parser.add_argument("--xgb-n-estimators", type=int, default=500, help="Number of boosting rounds (trees).")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05, help="Step size shrinkage.")
    parser.add_argument("--xgb-max-depth", type=int, default=5, help="Maximum depth of a tree.")
    parser.add_argument("--xgb-subsample", type=float, default=0.8, help="Fraction of samples used per tree.")
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8, help="Fraction of features used per tree.")
    parser.add_argument("--xgb-gamma", type=float, default=0.1, help="Minimum loss reduction required to make a further partition.")
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=20, help="Number of rounds with no improvement after which training stops.")
    # SHAP arguments
    parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis.")

    args = parser.parse_args()
    main(args)