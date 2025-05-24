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
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
# import shap # SHAP Removed
import time
import joblib
import json
import optuna  # Added for hyperparameter tuning

from ML_classifications.ML.src.gcs import gcs_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('seaborn-v0_8-darkgrid')
MODEL_NAME = "FinBench_SVC_Tuned"
N_OPTUNA_TRIALS = 25  # Number of Optuna trials for SVC


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
                logging.error(
                    f"Feature name count ({len(feature_names_list)}) != column count ({df.shape[1]}) for {blob_name}.")
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
    if not is_trial: logging.info(f"--- Model Evaluation on {dataset_name} ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = -1.0;
    pr_auc = -1.0
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba); pr_auc = average_precision_score(y_true, y_pred_proba)
        except ValueError as e:
            if not is_trial: logging.warning(f"Could not calculate AUC scores on {dataset_name}: {e}")
            roc_auc = 0.0;
            pr_auc = 0.0  # Default for trial if error
    elif not is_trial:
        logging.warning(f"Probability scores not available for {dataset_name}. AUCs not calculated.")

    cm = confusion_matrix(y_true, y_pred)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
               "roc_auc": roc_auc, "pr_auc": pr_auc, "confusion_matrix": cm.tolist()}
    if not is_trial:
        logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        if y_pred_proba is not None: logging.info(f"AUC-ROC: {roc_auc:.4f}, AUC-PR: {pr_auc:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")
    return metrics, cm


def plot_evaluation_charts(model, X_data, y_data, cm, plot_suffix, model_display_name, gcs_bucket, gcs_output_prefix):
    if cm is None:  # cm might be None if called during a trial where it's not computed
        logging.warning(f"Skipping CM plot for {plot_suffix} as CM is None.")
    else:
        try:
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
            ax_cm.set_title(f'{model_display_name} Confusion Matrix ({plot_suffix})');
            ax_cm.set_xlabel('Predicted Label');
            ax_cm.set_ylabel('True Label')
            save_plot_to_gcs(fig_cm, gcs_bucket,
                             f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_confusion_matrix_{plot_suffix.lower().replace(' ', '_')}.png")
        except Exception as e:
            logging.error(f"Failed to generate/save Confusion Matrix plot for {plot_suffix}: {e}")

    if not hasattr(model, "predict_proba"): logging.warning(
        f"Model does not have predict_proba. Skipping ROC/PR plots for {plot_suffix}."); return
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_estimator(model, X_data, y_data, ax=ax_roc, name=model_display_name)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)');
        ax_roc.set_title(f'{model_display_name} ROC Curve ({plot_suffix})');
        ax_roc.legend()
        save_plot_to_gcs(fig_roc, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_roc_curve_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed to generate/save ROC Curve plot for {plot_suffix}: {e}")
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_estimator(model, X_data, y_data, ax=ax_pr, name=model_display_name)
        ax_pr.set_title(f'{model_display_name} PR Curve ({plot_suffix})')
        save_plot_to_gcs(fig_pr, gcs_bucket,
                         f"{gcs_output_prefix}/plots/{MODEL_NAME.lower()}_pr_curve_{plot_suffix.lower().replace(' ', '_')}.png")
    except Exception as e:
        logging.error(f"Failed to generate/save PR Curve plot for {plot_suffix}: {e}")


# SHAP function perform_shap_analysis removed

def save_model_to_gcs(model, gcs_bucket, gcs_blob_name):
    logging.info(f"Saving model to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    try:
        with BytesIO() as buf:
            joblib.dump(model, buf); buf.seek(0); model_bytes = buf.read()
        gcs_utils.upload_bytes_to_gcs(gcs_bucket, model_bytes, gcs_blob_name, content_type='application/octet-stream')
        logging.info(f"Model successfully saved to GCS: gs://{gcs_bucket}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving model to GCS ({gcs_blob_name}): {e}")


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
        data_paths = metadata.get("gcs_paths", {})
        paths = {k: data_paths.get(k) for k in
                 ["X_train_resampled", "y_train_resampled", "X_val_processed", "y_val", "X_test_processed", "y_test"]}
        if not all(paths.values()): raise ValueError("One or more required data paths are missing in OHE metadata.")
    except Exception as e:
        logging.error(f"Failed to load or parse OHE metadata: {e}"); return

    try:
        logging.info("Loading processed OHE data from GCS...")
        X_train = load_data_from_gcs(GCS_BUCKET, paths["X_train_resampled"], processed_feature_names)
        y_train_df = load_data_from_gcs(GCS_BUCKET, paths["y_train_resampled"])
        X_val = load_data_from_gcs(GCS_BUCKET, paths["X_val_processed"], processed_feature_names)
        y_val_df = load_data_from_gcs(GCS_BUCKET, paths["y_val"])
        X_test = load_data_from_gcs(GCS_BUCKET, paths["X_test_processed"], processed_feature_names)
        y_test_df = load_data_from_gcs(GCS_BUCKET, paths["y_test"])
    except Exception as e:
        logging.error(f"Failed during data loading from GCS: {e}"); return

    y_train = y_train_df.iloc[:, 0].values;
    y_val = y_val_df.iloc[:, 0].values;
    y_test = y_test_df.iloc[:, 0].values
    logging.info(f"Data loading complete. X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # --- Optuna Objective Function ---
    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid", "linear"])
        c_param = trial.suggest_float("C", 1e-2, 1e2, log=True)
        gamma_param = "scale"  # Default
        degree_param = 3  # Default for poly

        if kernel == "rbf" or kernel == "poly" or kernel == "sigmoid":
            gamma_param = trial.suggest_categorical("gamma", ["scale", "auto"])
            # Could also suggest float: trial.suggest_float("gamma_float", 1e-4, 1e-1, log=True)
            # And then choose one, but "scale" and "auto" are common.
        if kernel == "poly":
            degree_param = trial.suggest_int("degree", 2, 5)

        model = SVC(C=c_param, kernel=kernel, gamma=gamma_param, degree=degree_param,
                    probability=True, random_state=42, verbose=False)  # Verbose off for trials

        # SVMs can be slow, consider subsetting X_train for trials if too slow
        # For now, using full X_train for each trial.
        model.fit(X_train, y_train)

        y_val_pred_proba_trial = model.predict_proba(X_val)[:, 1]
        # y_val_pred_trial = (y_val_pred_proba_trial > 0.5).astype(int)
        # _, val_metrics_trial = evaluate_model(y_val, y_val_pred_trial, y_val_pred_proba_trial, "Validation (Trial)", is_trial=True)

        # Optimize for PR AUC
        # trial_pr_auc = val_metrics_trial.get('pr_auc', 0.0)
        trial_pr_auc = average_precision_score(y_val, y_val_pred_proba_trial)
        return trial_pr_auc

    logging.info(f"--- Starting Optuna Hyperparameter Search ({N_OPTUNA_TRIALS} trials) ---")
    study = optuna.create_study(direction="maximize")  # Maximize PR AUC
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best_hyperparameters = study.best_params
    best_trial_metric_value = study.best_value
    logging.info(f"Best trial PR AUC: {best_trial_metric_value:.4f}")
    logging.info(f"Best hyperparameters: {best_hyperparameters}")

    # --- Final Model Training with Best Hyperparameters ---
    logging.info(f"--- Training Final {MODEL_NAME} Model with Best Hyperparameters ---")
    final_svm_model = SVC(
        C=best_hyperparameters["C"], kernel=best_hyperparameters["kernel"],
        gamma=best_hyperparameters.get("gamma", "scale"),  # .get due to conditional suggestion
        degree=best_hyperparameters.get("degree", 3),
        probability=True, random_state=42, verbose=args.svm_verbose
    )
    start_train_time = time.time()
    final_svm_model.fit(X_train, y_train)
    end_train_time = time.time();
    training_duration = end_train_time - start_train_time
    logging.info(f"Final {MODEL_NAME} training completed in {training_duration:.2f} seconds.")

    logging.info("Predicting on Validation set (Final Model)...")
    y_val_pred_final = final_svm_model.predict(X_val)
    y_val_pred_proba_final = final_svm_model.predict_proba(X_val)[:, 1]
    val_metrics_final, val_cm_final = evaluate_model(y_val, y_val_pred_final, y_val_pred_proba_final,
                                                     dataset_name="Validation Set (Final)")
    # plot_evaluation_charts(final_svm_model, X_val, y_val, val_cm_final, "Validation_Final", MODEL_NAME, GCS_BUCKET, GCS_OUTPUT_PREFIX)

    logging.info("Predicting on Test set (Final Model)...")
    y_test_pred_final = final_svm_model.predict(X_test)
    y_test_pred_proba_final = final_svm_model.predict_proba(X_test)[:, 1]
    test_metrics_final, test_cm_final = evaluate_model(y_test, y_test_pred_final, y_test_pred_proba_final,
                                                       dataset_name="Test Set (Final)")
    plot_evaluation_charts(final_svm_model, X_test, y_test, test_cm_final, "Test_Final", MODEL_NAME, GCS_BUCKET,
                           GCS_OUTPUT_PREFIX)

    # SHAP analysis removed

    model_blob_name = f"{GCS_OUTPUT_PREFIX}/model/{MODEL_NAME.lower()}_model.joblib"
    save_model_to_gcs(final_svm_model, GCS_BUCKET, model_blob_name)

    logging.info("Saving logs and metrics...")
    log_summary = {
        "model_type": MODEL_NAME, "script_args": vars(args), "metadata_source": METADATA_URI,
        "optuna_n_trials": N_OPTUNA_TRIALS,
        "optuna_best_trial_metric_value_pr_auc": best_trial_metric_value,
        "best_hyperparameters": best_hyperparameters,
        "final_model_training_duration_seconds": training_duration,
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

    parser.add_argument("--svm-c", type=float, default=1.0,
                        help="Default SVC regularization C (used if not tuned or as part of search space).")
    parser.add_argument("--svm-kernel", type=str, default='rbf', help="Default SVC kernel (used if not tuned).")
    parser.add_argument("--svm-gamma", type=str, default='scale', help="Default SVC gamma (used if not tuned).")
    parser.add_argument("--svm-verbose", action='store_true', help="Enable verbose output for final SVC training.")

    # SHAP arguments removed
    # parser.add_argument("--run-shap", action='store_true', help="Run SHAP analysis.")
    # parser.add_argument("--shap-sample-size", type=int, default=10, help="Samples for SHAP background/explanation.")
    args = parser.parse_args()
    main(args)