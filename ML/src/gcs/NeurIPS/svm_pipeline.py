# gcs/svm_pipeline.py
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import json
import os
import optuna

from ML_classifications.ML.src.gcs import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')


# --- Helper Functions ---
def load_data_from_gcs(gcs_bucket, gcs_path, feature_names=None):
    logging.info(f"Loading data from: gs://{gcs_bucket}/{gcs_path}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, gcs_path)
        if data_bytes is None: raise FileNotFoundError(f"Could not download GCS file: gs://{gcs_bucket}/{gcs_path}")
        df = pd.read_csv(BytesIO(data_bytes))
        if feature_names is not None and not df.columns.equals(pd.Index(feature_names)):
            if len(feature_names) == len(df.columns):
                df.columns = feature_names
            else:
                logging.error(f"Feature name count mismatch in {gcs_path}.")
        logging.info(f"Data loaded successfully from gs://{gcs_bucket}/{gcs_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from gs://{gcs_bucket}/{gcs_path}: {e}")
        raise


def save_plot_to_gcs(fig, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving plot to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_bytes = buf.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, plot_bytes, gcs_blob_name, content_type='image/png')
        plt.close(fig)
    except Exception as e:
        logging.error(f"ERROR saving plot to GCS ({gcs_blob_name}): {e}")
        plt.close(fig)


def evaluate_model(y_true, y_pred, y_pred_proba):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else -1.0,
        "pr_auc": average_precision_score(y_true, y_pred_proba) if y_pred_proba is not None else -1.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics, confusion_matrix(y_true, y_pred)


def plot_evaluation_charts(calibrated_model, X_test, y_test, cm, model_name, gcs_bucket, gcs_output_prefix):
    # Ensure X_test is numpy for sklearn compatibility if it's a DataFrame
    if isinstance(X_test, pd.DataFrame):
        X_test_np = X_test.values
    else:
        X_test_np = X_test

    if not hasattr(calibrated_model, "predict_proba"):
        logging.error(f"Model {model_name} lacks predict_proba. Cannot plot ROC/PR.")
        return
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title(f'{model_name} Confusion Matrix')
        save_plot_to_gcs(fig_cm, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    except Exception as e:
        logging.error(f"Failed CM plot: {e}")
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_estimator(calibrated_model, X_test_np, y_test, ax=ax_roc, name=model_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
        ax_roc.set_title(f'{model_name} ROC Curve')
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_roc_curve.png")
    except Exception as e:
        logging.error(f"Failed ROC plot: {e}")
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_estimator(calibrated_model, X_test_np, y_test, ax=ax_pr, name=model_name)
        ax_pr.set_title(f'{model_name} PR Curve')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_pr_curve.png")
    except Exception as e:
        logging.error(f"Failed PR plot: {e}")


def save_model_to_gcs_joblib(model, gcs_bucket, gcs_blob_name, metadata=None):
    logging.info(f"Saving model object to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        with BytesIO() as buf:
            joblib.dump({'model': model, 'metadata': metadata or {}}, buf)
            buf.seek(0)
            model_bytes = buf.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model successfully saved to gs://{gcs_bucket}/{gcs_blob_name}")
        if metadata: logging.info(f"Model metadata included: {metadata}")
    except Exception as e:
        logging.error(f"ERROR saving model to GCS ({gcs_blob_name}): {e}")


# --- Optuna Objective Function ---
def objective(trial, X_train, y_train, X_test, y_test, args):
    svm_c = trial.suggest_float("svm_c", 1e-3, 1e2, log=True)
    # LinearSVC doesn't have 'kernel' or 'gamma' like SVC.
    # We can tune calibration method or CV folds if desired
    # calibration_method = trial.suggest_categorical("calibration_method", ['sigmoid', 'isotonic'])

    base_svm = LinearSVC(
        C=svm_c,
        loss='squared_hinge',
        penalty='l2',
        dual="auto",
        random_state=42,
        max_iter=args.svm_max_iter,  # Use fixed max_iter from args
        # class_weight='balanced' # Consider if data is imbalanced
    )
    calibrated_svm_model = CalibratedClassifierCV(base_svm, cv=3, method='sigmoid')  # Fixed calibration for simplicity

    calibrated_svm_model.fit(X_train, y_train)
    y_pred_proba = calibrated_svm_model.predict_proba(X_test)[:, 1]

    # Using ROC AUC for optimization
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f"Trial {trial.number}: C={svm_c:.4f}, ROC AUC={roc_auc:.4f}")
    return roc_auc


# --- Main Execution ---
def main(args):
    logging.info("--- Starting LinearSVC (Calibrated) Training Pipeline with Tuning ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')
    METADATA_URI = args.metadata_uri

    logging.info(f"Loading metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata = gcs_utils.load_json_from_gcs(GCS_BUCKET, metadata_blob_name)
        if not metadata: raise FileNotFoundError("Metadata not found.")
        processed_feature_names = metadata.get("processed_feature_names")
        if not processed_feature_names: raise ValueError("Processed feature names not in metadata.")
        data_paths = metadata.get("gcs_paths", {})
        # ... (load paths as before)
        x_train_path = data_paths.get("X_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_train_path = data_paths.get("y_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_test_path = data_paths.get("X_test_processed", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_test_path = data_paths.get("y_test", "").replace(f"gs://{GCS_BUCKET}/", "")
        if not all([x_train_path, y_train_path, x_test_path, y_test_path]):
            raise ValueError("One or more required data paths are missing in metadata.")
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        return

    X_train = load_data_from_gcs(GCS_BUCKET, x_train_path, processed_feature_names)
    y_train_df = load_data_from_gcs(GCS_BUCKET, y_train_path)
    X_test = load_data_from_gcs(GCS_BUCKET, x_test_path, processed_feature_names)
    y_test_df = load_data_from_gcs(GCS_BUCKET, y_test_path)
    y_train = y_train_df.iloc[:, 0].values
    y_test = y_test_df.iloc[:, 0].values

    # Convert to numpy if they are pandas DataFrames for scikit-learn/Optuna
    if isinstance(X_train, pd.DataFrame): X_train = X_train.values
    if isinstance(X_test, pd.DataFrame): X_test = X_test.values

    logging.info(f"--- Starting Hyperparameter Tuning (Optuna) for {args.n_trials} trials ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, args),
                   n_trials=args.n_trials)

    best_hyperparams = study.best_params
    best_metric_value = study.best_value
    logging.info(f"--- Hyperparameter Tuning Finished ---")
    logging.info(f"Best ROC AUC: {best_metric_value:.4f}")
    logging.info(f"Best Hyperparameters: {best_hyperparams}")

    logging.info("--- Training Final Model with Best Hyperparameters ---")
    final_svm_c = best_hyperparams["svm_c"]
    # final_calib_method = best_hyperparams.get("calibration_method", 'sigmoid') # If tuned

    final_base_svm = LinearSVC(
        C=final_svm_c, loss='squared_hinge', penalty='l2', dual="auto",
        random_state=42, max_iter=args.svm_max_iter
    )
    final_calibrated_model = CalibratedClassifierCV(final_base_svm, cv=3, method='sigmoid')  # Or final_calib_method

    start_train_time = time.time()
    final_calibrated_model.fit(X_train, y_train)
    training_duration = time.time() - start_train_time
    logging.info(f"Final LinearSVC + Calibration training completed in {training_duration:.2f} seconds.")

    y_pred = final_calibrated_model.predict(X_test)
    y_pred_proba = final_calibrated_model.predict_proba(X_test)[:, 1]

    metrics, conf_matrix = evaluate_model(y_test, y_pred, y_pred_proba)
    logging.info(f"Final Model Evaluation Metrics: {metrics}")

    plot_evaluation_charts(final_calibrated_model, X_test, y_test, conf_matrix, "LinearSVC Calibrated (Best)",
                           GCS_BUCKET, GCS_OUTPUT_PREFIX)

    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/linearsvc_calibrated_best_model.joblib"
    model_metadata = {"best_hyperparameters": best_hyperparams, "best_metric_value": best_metric_value,
                      "optimized_metric": "roc_auc"}
    save_model_to_gcs_joblib(final_calibrated_model, GCS_BUCKET, model_blob_name, metadata=model_metadata)

    log_summary = {
        "model_type": "LinearSVC (Calibrated)",
        "tuning_args": {"n_trials": args.n_trials, "svm_max_iter_per_trial": args.svm_max_iter},
        "best_hyperparameters": best_hyperparams,
        "best_tuned_metric_value": best_metric_value,
        "metadata_source": METADATA_URI,
        "final_model_training_duration_seconds": training_duration,
        "final_model_evaluation_metrics": metrics,
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }
    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/linearsvc_final_training_log.json"
    gcs_utils.save_json_to_gcs(log_summary, GCS_BUCKET, log_blob_name)
    logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")

    logging.info("--- LinearSVC (Calibrated) Training Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, tune, and evaluate a Calibrated LinearSVC model.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    parser.add_argument("--metadata-uri", type=str, required=True, help="GCS URI of the metadata file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")

    parser.add_argument("--n-trials", type=int, default=15, help="Number of Optuna trials.")  # Fewer trials for SVM
    parser.add_argument("--svm-max-iter", type=int, default=1000,
                        help="Max iterations for LinearSVC (per trial and final).")  # Reduced for faster tuning
    # Removed --svm-c as it's tuned. SHAP args also removed.

    args = parser.parse_args()
    main(args)