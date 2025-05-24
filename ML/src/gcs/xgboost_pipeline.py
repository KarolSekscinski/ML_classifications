# gcs/xgboost_pipeline.py
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
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import os
import optuna

import gcs_utils

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
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics, confusion_matrix(y_true, y_pred)


def plot_evaluation_charts(estimator, X_test, y_test, cm, model_name, gcs_bucket, gcs_output_prefix):
    # Ensure X_test is numpy for sklearn compatibility if it's a DataFrame
    if isinstance(X_test, pd.DataFrame):
        X_test_np = X_test.values  # Or X_test directly if estimator handles DataFrames
    else:
        X_test_np = X_test

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
        RocCurveDisplay.from_estimator(estimator, X_test_np, y_test, ax=ax_roc, name=model_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)')
        ax_roc.set_title(f'{model_name} ROC Curve')
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_roc_curve.png")
    except Exception as e:
        logging.error(f"Failed ROC plot: {e}")
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_estimator(estimator, X_test_np, y_test, ax=ax_pr, name=model_name)
        ax_pr.set_title(f'{model_name} PR Curve')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{model_name.lower().replace(' ', '_')}_pr_curve.png")
    except Exception as e:
        logging.error(f"Failed PR plot: {e}")


def save_xgboost_model_to_gcs(model, gcs_bucket, gcs_blob_name, metadata=None):
    logging.info(f"Saving XGBoost model to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    temp_model_path = "temp_xgb_model.json"
    try:
        # Save model and metadata separately or embed metadata if model format supports it
        # For XGBoost, typically save model, then metadata as a separate JSON or in the log
        model.save_model(temp_model_path)
        with open(temp_model_path, 'rb') as f:
            model_bytes = f.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/json')
        logging.info(f"XGBoost model successfully saved to gs://{gcs_bucket}/{gcs_blob_name}")
        if metadata:  # Save metadata alongside if provided
            meta_blob_name = gcs_blob_name.replace(".json", "_train_meta.json")
            gcs_utils.save_json_to_gcs(metadata, gcs_bucket, meta_blob_name)
            logging.info(f"XGBoost training metadata saved to: gs://{gcs_bucket}/{meta_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving XGBoost model to GCS ({gcs_blob_name}): {e}")
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)


# --- Optuna Objective Function ---
def objective(trial, X_train, y_train, X_test, y_test, args):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': args.optimization_metric,  # 'aucpr' or 'auc'
        'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),  # L2 regularization
        'alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),  # L1 regularization
        'random_state': 42,
        'n_jobs': -1
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=args.xgb_early_stopping_rounds,
              verbose=False)  # Reduce verbosity during tuning

    # Use the best score from early stopping if available, otherwise predict and score
    if model.best_score:  # For metrics like 'auc', 'aucpr' this is directly available
        metric_value = model.best_score
    else:  # Fallback if best_score is not directly suitable (e.g. 'logloss')
        y_pred_proba = model.predict_proba(X_test, iteration_range=(0, model.best_iteration + 1))[:, 1]
        if args.optimization_metric == 'aucpr':
            metric_value = average_precision_score(y_test, y_pred_proba)
        elif args.optimization_metric == 'auc':  # Default to roc_auc if 'auc'
            metric_value = roc_auc_score(y_test, y_pred_proba)
        else:  # Should match one of XGBoost's eval_metrics
            # This path might not be needed if eval_metric is set correctly and gives best_score
            logging.warning(f"Relying on manual calculation for {args.optimization_metric}")
            metric_value = roc_auc_score(y_test, y_pred_proba)  # Defaulting for safety

    logging.info(
        f"Trial {trial.number}: Params={params}, Metric ({args.optimization_metric})={metric_value:.4f}, Best Iter={model.best_iteration}")
    return metric_value


# --- Main Execution ---
def main(args):
    logging.info("--- Starting XGBoost Training Pipeline with Tuning ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')
    METADATA_URI = args.metadata_uri

    logging.info(f"Loading metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata = gcs_utils.load_json_from_gcs(GCS_BUCKET, metadata_blob_name)
        if not metadata: raise FileNotFoundError("Metadata not found.")
        processed_feature_names = metadata.get("processed_feature_names")
        if not processed_feature_names: raise ValueError("Feature names not in metadata.")
        data_paths = metadata.get("gcs_paths", {})
        x_train_path = data_paths.get("X_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_train_path = data_paths.get("y_train_resampled", "").replace(f"gs://{GCS_BUCKET}/", "")
        x_test_path = data_paths.get("X_test_processed", "").replace(f"gs://{GCS_BUCKET}/", "")
        y_test_path = data_paths.get("y_test", "").replace(f"gs://{GCS_BUCKET}/", "")
        if not all([x_train_path, y_train_path, x_test_path, y_test_path]):
            raise ValueError("Missing data paths in metadata.")
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        return

    X_train_df = load_data_from_gcs(GCS_BUCKET, x_train_path, processed_feature_names)
    y_train_df = load_data_from_gcs(GCS_BUCKET, y_train_path)
    X_test_df = load_data_from_gcs(GCS_BUCKET, x_test_path, processed_feature_names)
    y_test_df = load_data_from_gcs(GCS_BUCKET, y_test_path)

    y_train = y_train_df.iloc[:, 0].values
    y_test = y_test_df.iloc[:, 0].values

    # XGBoost can handle DataFrame directly, feature names are preserved
    # X_train = X_train_df
    # X_test = X_test_df
    # Or convert to numpy if preferred by Optuna setup / consistency
    X_train_np = X_train_df.values
    X_test_np = X_test_df.values

    logging.info(f"--- Starting Hyperparameter Tuning (Optuna) for {args.n_trials} trials ---")
    study = optuna.create_study(direction="maximize")  # Maximize AUC or AUCPR
    # Pass numpy arrays to objective function for consistency, XGBoost handles them.
    study.optimize(lambda trial: objective(trial, X_train_np, y_train, X_test_np, y_test, args),
                   n_trials=args.n_trials)

    best_hyperparams = study.best_params
    best_metric_value = study.best_value
    logging.info(f"--- Hyperparameter Tuning Finished ---")
    logging.info(f"Best {args.optimization_metric}: {best_metric_value:.4f}")
    logging.info(f"Best Hyperparameters: {best_hyperparams}")

    logging.info("--- Training Final Model with Best Hyperparameters ---")
    final_params = {
        'objective': 'binary:logistic', 'eval_metric': args.optimization_metric, 'use_label_encoder': False,
        'random_state': 42, 'n_jobs': -1, **best_hyperparams
    }
    # Ensure n_estimators from best_params is used, it might be named differently if not from objective
    if 'n_estimators' not in final_params and 'best_n_estimators' in best_hyperparams:  # Example
        final_params['n_estimators'] = best_hyperparams['best_n_estimators']
    elif 'n_estimators' not in final_params:  # Default if not found in trial (should be there)
        final_params['n_estimators'] = 500

    final_xgb_model = xgb.XGBClassifier(**final_params)

    start_train_time = time.time()
    # For final model, fit on full training, use early stopping against test set.
    # The 'n_estimators' from tuning is often the max; early stopping finds optimal within that.
    final_xgb_model.fit(X_train_np, y_train,
                        eval_set=[(X_test_np, y_test)],
                        early_stopping_rounds=args.xgb_early_stopping_rounds,  # Apply early stopping to final fit
                        verbose=True)
    training_duration = time.time() - start_train_time
    logging.info(f"Final XGBoost training completed in {training_duration:.2f} seconds.")
    logging.info(
        f"Final model best iteration: {final_xgb_model.best_iteration}, Best score ({args.optimization_metric}): {final_xgb_model.best_score}")

    y_pred_proba = final_xgb_model.predict_proba(X_test_np, iteration_range=(0, final_xgb_model.best_iteration + 1))[:,
                   1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics, conf_matrix = evaluate_model(y_test, y_pred, y_pred_proba)
    logging.info(f"Final Model Evaluation Metrics: {metrics}")

    plot_evaluation_charts(final_xgb_model, X_test_df, y_test, conf_matrix, 'XGBoost (Best)', GCS_BUCKET,
                           GCS_OUTPUT_PREFIX)

    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/xgboost_best_model.json"
    # For XGBoost, metadata is primarily the hyperparameters
    model_metadata = {"best_hyperparameters": best_hyperparams,
                      "best_tuned_metric_value": best_metric_value,
                      "optimized_metric": args.optimization_metric,
                      "final_model_best_iteration": final_xgb_model.best_iteration,
                      "final_model_best_score": final_xgb_model.best_score
                      }
    save_xgboost_model_to_gcs(final_xgb_model, GCS_BUCKET, model_blob_name, metadata=model_metadata)

    log_summary = {
        "model_type": "XGBoost",
        "tuning_args": {"n_trials": args.n_trials, "early_stopping_rounds": args.xgb_early_stopping_rounds,
                        "optimization_metric": args.optimization_metric},
        "best_hyperparameters": best_hyperparams,
        "best_tuned_metric_value": best_metric_value,
        "metadata_source": METADATA_URI,
        "final_model_training_duration_seconds": training_duration,
        "final_model_best_iteration": final_xgb_model.best_iteration,
        "final_model_best_eval_score": final_xgb_model.best_score,
        "final_model_evaluation_metrics": metrics,
        "output_gcs_prefix": f"gs://{GCS_BUCKET}/{GCS_OUTPUT_PREFIX}",
        "saved_model_path": f"gs://{GCS_BUCKET}/{model_blob_name}"
    }
    log_blob_name = f"{GCS_OUTPUT_PREFIX}/logs/xgboost_final_training_log.json"
    gcs_utils.save_json_to_gcs(log_summary, GCS_BUCKET, log_blob_name)
    logging.info(f"Log summary saved to gs://{GCS_BUCKET}/{log_blob_name}")

    logging.info("--- XGBoost Training Pipeline Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, tune, and evaluate an XGBoost model.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    parser.add_argument("--metadata-uri", type=str, required=True, help="GCS URI of the metadata file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")

    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument("--optimization-metric", type=str, default="aucpr", choices=['aucpr', 'auc'],
                        help="XGBoost eval_metric for optimization.")
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=15, help="Early stopping rounds for XGBoost.")
    # Removed specific XGBoost hyperparams as they are tuned now. SHAP args also removed.

    args = parser.parse_args()
    main(args)