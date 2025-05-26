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
# import shap # SHAP Removed
import time
import json
import os
import optuna  # Added for hyperparameter tuning

import gcs_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')
MODEL_NAME = "FinBench_XGBoost_Tuned"
N_OPTUNA_TRIALS = 25  # Number of Optuna trials for XGBoost


def load_data_from_gcs(gcs_bucket, gcs_path_in_metadata, feature_names_list=None):
    if not gcs_path_in_metadata or not gcs_path_in_metadata.startswith(f"gs://{gcs_bucket}/"):
        logging.error(f"Invalid GCS path from metadata: {gcs_path_in_metadata} for bucket {gcs_bucket}")
        raise ValueError(f"Invalid GCS path: {gcs_path_in_metadata}")
    blob_name = gcs_path_in_metadata.replace(f"gs://{gcs_bucket}/", "")
    logging.info(f"Loading data from: gs://{gcs_bucket}/{blob_name}")
    try:
        data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, blob_name)
        if data_bytes is None: raise FileNotFoundError(f"Could not download GCS file: gs://{gcs_bucket}/{blob_name}")
        df = pd.read_csv(BytesIO(data_bytes))
        if feature_names_list is not None:
            if len(feature_names_list) == df.shape[1]:
                df.columns = feature_names_list
            else:
                logging.error(f"Feature name count mismatch for {blob_name}.")
        logging.info(f"Data loaded successfully from gs://{gcs_bucket}/{blob_name}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from gs://{gcs_bucket}/{blob_name}: {e}"); raise


def save_plot_to_gcs(fig, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving plot to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        buf = BytesIO();
        fig.savefig(buf, format='png', bbox_inches='tight');
        buf.seek(0)
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, buf.read(), gcs_blob_name, content_type='image/png')
        logging.info(f"Plot successfully saved to gs://{gcs_bucket}/{gcs_blob_name}");
        plt.close(fig)
    except Exception as e:
        logging.error(f"ERROR saving plot to GCS ({gcs_blob_name}): {e}"); plt.close(fig)


def evaluate_model(y_true, y_pred, y_pred_proba, dataset_name="Test Set", is_trial=False):
    # (Identical to svm_finbench_cf2.py's evaluate_model, adapted for trials)
    if not is_trial: logging.info(f"--- Model Evaluation on {dataset_name} ---")
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
            if not is_trial: logging.warning(f"Could not calculate AUC scores on {dataset_name}: {e}")
            roc_auc = 0.0;
            pr_auc = 0.0
    elif not is_trial:
        logging.warning(f"Probability scores not available for {dataset_name}. AUCs not calculated.")
    cm = confusion_matrix(y_true, y_pred)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "roc_auc": roc_auc,
               "pr_auc": pr_auc, "confusion_matrix": cm.tolist()}
    if not is_trial:
        logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        if y_pred_proba is not None: logging.info(f"AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")
    return metrics, cm


def plot_evaluation_charts(model, X_data, y_data, cm, plot_suffix, model_display_name, gcs_bucket, gcs_output_prefix):
    # (Identical to svm_finbench_cf2.py's plot_evaluation_charts)
    if cm is None:
        logging.warning(f"Skipping CM plot for {plot_suffix} as CM is None.")
    else:
        try:
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5));
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
            ax_cm.set_title(f'{model_display_name} CM ({plot_suffix})');
            ax_cm.set_xlabel('Predicted');
            ax_cm.set_ylabel('True')
            save_plot_to_gcs(fig_cm, gcs_bucket,
                             f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_confusion_matrix_{plot_suffix.lower().replace(' ', '_')}.png")
        except Exception as e:
            logging.error(f"Failed CM plot for {plot_suffix}: {e}")
    if not hasattr(model, "predict_proba"): logging.warning(
        f"No predict_proba for {model_display_name}. Skipping ROC/PR for {plot_suffix}."); return
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6));
        RocCurveDisplay.from_estimator(model, X_data, y_data, ax=ax_roc, name=model_display_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC=0.5)');
        ax_roc.set_title(f'{model_display_name} ROC ({plot_suffix})');
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_roc_curve_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed ROC plot for {plot_suffix}: {e}")
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6));
        PrecisionRecallDisplay.from_estimator(model, X_data, y_data, ax=ax_pr, name=model_display_name)
        ax_pr.set_title(f'{model_display_name} PR Curve ({plot_suffix})')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_pr_curve_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed PR plot for {plot_suffix}: {e}")


# SHAP function perform_shap_analysis_xgb removed

def save_model_to_gcs_xgb(model, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving XGBoost model to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    temp_model_path = "temp_xgb_model.json"
    try:
        model.save_model(temp_model_path)
        with open(temp_model_path, 'rb') as f:
            model_bytes = f.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/json')
        logging.info(f"XGBoost model successfully saved to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
        os.remove(temp_model_path)
    except Exception as e:
        logging.error(f"ERROR saving XGBoost model to GCS ({gcs_blob_name}): {e}")
        if os.path.exists(temp_model_path): os.remove(temp_model_path)


def main(args):
    logging.info(f"--- Starting {MODEL_NAME} Training Pipeline for FinBench cf2 ---")
    GCS_BUCKET = args.gcs_bucket;
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/');
    METADATA_URI = args.metadata_uri

    logging.info(f"Loading OHE metadata from: {METADATA_URI}")
    try:
        metadata_blob_name = METADATA_URI.replace(f"gs://{GCS_BUCKET}/", "")
        metadata_str = gcs_utils.download_blob_as_string(GCS_BUCKET, metadata_blob_name)
        if metadata_str is None: raise FileNotFoundError("Could not download metadata file.")
        metadata = json.loads(metadata_str);
        logging.info("OHE Metadata loaded successfully.")
        processed_feature_names = metadata.get("processed_feature_names_list")
        if not processed_feature_names: raise ValueError("Processed feature names list missing in metadata.")
        data_paths = metadata.get("gcs_paths", {});
        paths = {k: data_paths.get(k) for k in
                 ["X_train_resampled", "y_train_resampled", "X_val_processed", "y_val", "X_test_processed", "y_test"]}
        if not all(paths.values()): raise ValueError("One or more required data paths missing in OHE metadata.")
    except Exception as e:
        logging.error(f"Failed to load/parse OHE metadata: {e}"); return

    try:
        logging.info("Loading processed OHE data from GCS...")
        X_train = load_data_from_gcs(GCS_BUCKET, paths["X_train_resampled"], processed_feature_names)
        y_train_df = load_data_from_gcs(GCS_BUCKET, paths["y_train_resampled"]);
        y_train = y_train_df.iloc[:, 0].values
        X_val = load_data_from_gcs(GCS_BUCKET, paths["X_val_processed"], processed_feature_names)
        y_val_df = load_data_from_gcs(GCS_BUCKET, paths["y_val"]);
        y_val = y_val_df.iloc[:, 0].values
        X_test = load_data_from_gcs(GCS_BUCKET, paths["X_test_processed"], processed_feature_names)
        y_test_df = load_data_from_gcs(GCS_BUCKET, paths["y_test"]);
        y_test = y_test_df.iloc[:, 0].values
    except Exception as e:
        logging.error(f"Failed data loading from GCS: {e}"); return
    logging.info(f"Data loading complete. X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # --- Optuna Objective Function ---
    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': args.xgb_eval_metric,  # Use from args, e.g., 'aucpr'
            'use_label_encoder': False,
            'verbosity': 0,  # Silent during trials
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),  # Max estimators for trial
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),  # L1
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),  # L2
            'n_jobs': -1
        }

        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=args.xgb_early_stopping_rounds,  # Use from args
                  verbose=False)  # No verbose logging for each trial's fit

        # Get metric from the best iteration
        # The metric name in results_ is typically 'validation_0-<eval_metric>'
        # Example: if eval_metric is 'aucpr', key is 'validation_0-aucpr'
        metric_key = f'validation_0-{args.xgb_eval_metric}'

        # XGBoost's early stopping already finds the best score on eval_set
        # The score is stored in model.best_score if early stopping is used
        # If not, we can evaluate on X_val
        if hasattr(model, 'best_score') and model.best_score is not None:
            trial_metric_value = model.best_score
        else:  # Fallback if best_score not directly available (e.g. early stopping not triggered)
            y_val_pred_proba_trial = model.predict_proba(X_val)[:, 1]
            if args.xgb_eval_metric == 'aucpr':
                trial_metric_value = average_precision_score(y_val, y_val_pred_proba_trial)
            elif args.xgb_eval_metric == 'auc':
                trial_metric_value = roc_auc_score(y_val, y_val_pred_proba_trial)
            elif args.xgb_eval_metric == 'logloss':  # Lower is better for logloss
                # Optuna needs to maximize, so return negative logloss or handle direction in study
                # For simplicity, assuming eval_metric is something where higher is better (like auc, aucpr)
                from sklearn.metrics import log_loss
                trial_metric_value = -log_loss(y_val, y_val_pred_proba_trial)  # Negative logloss for maximization
            else:  # Default to PR AUC if metric unknown for optimization direction
                trial_metric_value = average_precision_score(y_val, y_val_pred_proba_trial)

        return trial_metric_value

    logging.info(f"--- Starting Optuna Hyperparameter Search ({N_OPTUNA_TRIALS} trials) ---")
    # Ensure direction is correct for the chosen xgb_eval_metric
    # 'auc', 'aucpr' are maximized. 'logloss', 'rmse' are minimized.
    direction = "maximize"
    if args.xgb_eval_metric in ["logloss", "rmse", "error"]:
        direction = "minimize"

    study = optuna.create_study(direction=direction, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best_hyperparameters = study.best_params
    best_trial_metric_value = study.best_value
    logging.info(f"Best trial {args.xgb_eval_metric}: {best_trial_metric_value:.4f}")
    logging.info(f"Best hyperparameters: {best_hyperparameters}")

    # --- Final Model Training with Best Hyperparameters ---
    logging.info(f"--- Training Final {MODEL_NAME} Model with Best Hyperparameters ---")
    final_xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': args.xgb_eval_metric,
        'use_label_encoder': False, 'random_state': 42, 'n_jobs': -1,
        'verbosity': (100 if args.xgb_verbose else 0)  # Control verbosity for final model
    }
    final_xgb_params.update(best_hyperparameters)  # Add tuned params

    final_xgb_model = xgb.XGBClassifier(**final_xgb_params)
    start_train_time = time.time()
    final_xgb_model.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=args.xgb_early_stopping_rounds,  # Use early stopping for final model too
                        verbose=args.xgb_verbose)  # args.xgb_verbose can be int or bool
    end_train_time = time.time();
    training_duration = end_train_time - start_train_time
    logging.info(f"Final {MODEL_NAME} training completed in {training_duration:.2f} seconds.")

    best_iteration_final = final_xgb_model.best_iteration if hasattr(final_xgb_model,
                                                                     'best_iteration') and final_xgb_model.best_iteration is not None else final_xgb_params.get(
        'n_estimators') - 1
    best_score_final = final_xgb_model.best_score if hasattr(final_xgb_model, 'best_score') else None
    if best_iteration_final is not None: logging.info(
        f"Final model best iteration: {best_iteration_final}, Best score: {best_score_final if best_score_final else 'N/A'}")

    logging.info("Predicting on Validation set (Final Model)...")
    y_val_pred_proba_final = final_xgb_model.predict_proba(X_val, iteration_range=(0, best_iteration_final + 1))[:, 1]
    y_val_pred_final = (y_val_pred_proba_final > 0.5).astype(int)
    val_metrics_final, val_cm_final = evaluate_model(y_val, y_val_pred_final, y_val_pred_proba_final,
                                                     dataset_name="Validation Set (Final)")
    # plot_evaluation_charts(final_xgb_model, X_val, y_val, val_cm_final, "Validation_Final", MODEL_NAME, GCS_BUCKET, GCS_OUTPUT_PREFIX)

    logging.info("Predicting on Test set (Final Model)...")
    y_test_pred_proba_final = final_xgb_model.predict_proba(X_test, iteration_range=(0, best_iteration_final + 1))[:, 1]
    y_test_pred_final = (y_test_pred_proba_final > 0.5).astype(int)
    test_metrics_final, test_cm_final = evaluate_model(y_test, y_test_pred_final, y_test_pred_proba_final,
                                                       dataset_name="Test Set (Final)")
    plot_evaluation_charts(final_xgb_model, X_test, y_test, test_cm_final, "Test_Final", MODEL_NAME, GCS_BUCKET,
                           GCS_OUTPUT_PREFIX)

    # SHAP analysis removed

    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/{MODEL_NAME.lower()}_model.json"
    save_model_to_gcs_xgb(final_xgb_model, GCS_BUCKET, model_blob_name)

    logging.info("Saving logs and metrics...")
    log_summary = {
        "model_type": MODEL_NAME, "script_args": vars(args), "metadata_source": METADATA_URI,
        "optuna_n_trials": N_OPTUNA_TRIALS,
        "optuna_best_trial_metric_value": best_trial_metric_value,
        "best_hyperparameters": best_hyperparameters,
        "final_model_training_duration_seconds": training_duration,
        "final_model_best_iteration": best_iteration_final,
        "final_model_best_eval_score_on_val": best_score_final,
        "final_model_validation_set_evaluation": val_metrics_final,
        "final_model_test_set_evaluation": test_metrics_final,
        # SHAP Removed "shap_analysis_run": False, "shap_top_features": None,
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
    parser = argparse.ArgumentParser(description=f"Train and evaluate an {MODEL_NAME} model with Optuna.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name.")
    parser.add_argument("--metadata-uri", type=str, required=True,
                        help="GCS URI of the OHE preprocessing_metadata.json file.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True, help="GCS prefix for saving outputs.")

    parser.add_argument("--xgb-n-estimators", type=int, default=500,
                        help="Default/max num boosting rounds for trials (overridden by Optuna's suggestion).")
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05,
                        help="Default learning rate (overridden by Optuna).")
    parser.add_argument("--xgb-max-depth", type=int, default=5, help="Default max depth (overridden by Optuna).")
    parser.add_argument("--xgb-subsample", type=float, default=0.8,
                        help="Default subsample ratio (overridden by Optuna).")
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8,
                        help="Default colsample_bytree (overridden by Optuna).")
    parser.add_argument("--xgb-gamma", type=float, default=0.1, help="Default gamma (overridden by Optuna).")
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=20,
                        help="Early stopping rounds for trials and final model.")
    parser.add_argument("--xgb-eval-metric", type=str, default="aucpr", help="Evaluation metric for XGBoost.")
    parser.add_argument("--xgb-verbose", type=int, default=0,
                        help="Verbosity for final XGBoost training (0, 1, N). Trials are silent.")

    # SHAP arguments removed
    # parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis.")
    args = parser.parse_args()
    main(args)