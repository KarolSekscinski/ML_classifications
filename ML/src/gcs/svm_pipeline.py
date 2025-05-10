import time
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import shap


# Assuming gcs_utils contains custom_print, save_plot_to_gcs, upload_bytes_to_gcs
# These would be imported in the main script and passed as arguments, or gcs_utils imported here.
# For simplicity in this module, we'll assume they are passed or available.

def run_svm_pipeline(X_train_resampled, y_train_resampled, X_test_processed, y_test,
                     processed_feature_names, gcs_bucket_name, gcs_output_prefix,
                     custom_print_func, save_plot_func, upload_bytes_func):
    """
    Runs the SVM model training, tuning, evaluation, and SHAP interpretation.
    """
    custom_print_func("\n--- SVM Model Training and Hyperparameter Tuning ---")

    # Define the parameter grid for SVM
    param_grid_svm = {'C': [0.1, 1], 'gamma': ['scale'], 'kernel': ['rbf']}  # Reduced grid for quick demo
    cv_stratified = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Reduced folds for speed

    grid_search_svm = GridSearchCV(
        estimator=SVC(probability=True, random_state=42),
        param_grid=param_grid_svm,
        scoring='average_precision',
        cv=cv_stratified,
        verbose=1,
        n_jobs=-1
    )

    custom_print_func("Starting GridSearchCV for SVM...")
    start_time_svm_train = time.time()
    grid_search_svm.fit(X_train_resampled, y_train_resampled)
    svm_training_time = time.time() - start_time_svm_train
    best_svm_model = grid_search_svm.best_estimator_

    custom_print_func("\nBest SVM Parameters found:")
    custom_print_func(grid_search_svm.best_params_)
    custom_print_func(f"Best SVM Average Precision (PR AUC) during CV: {grid_search_svm.best_score_:.4f}")
    custom_print_func(f"SVM Training Time (GridSearchCV): {svm_training_time:.2f} seconds")

    # --- Evaluate Best SVM Model on Test Set ---
    custom_print_func("\n--- Evaluating Best SVM Model on Test Set ---")
    y_pred_svm = best_svm_model.predict(X_test_processed)
    y_proba_svm = best_svm_model.predict_proba(X_test_processed)[:, 1]

    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm_fraud = precision_score(y_test, y_pred_svm, pos_label=1, zero_division=0)
    recall_svm_fraud = recall_score(y_test, y_pred_svm, pos_label=1, zero_division=0)
    f1_svm_fraud = f1_score(y_test, y_pred_svm, pos_label=1, zero_division=0)

    custom_print_func("\nSVM Detailed Metrics on Test Set:")
    custom_print_func(f"Accuracy: {accuracy_svm:.4f}")
    custom_print_func(f"Precision (fraud): {precision_svm_fraud:.4f}")
    custom_print_func(f"Recall/Sensitivity (fraud): {recall_svm_fraud:.4f}")
    custom_print_func(f"F1-score (fraud): {f1_svm_fraud:.4f}")

    try:
        report_svm_str = classification_report(y_test, y_pred_svm, target_names=['Legit (0)', 'Fraud (1)'],
                                               zero_division=0)
        custom_print_func("\nSVM Classification Report on Test Set:\n" + report_svm_str)
    except ValueError:
        report_svm_str = classification_report(y_test, y_pred_svm, zero_division=0)
        custom_print_func(
            "\nSVM Classification Report on Test Set (single class in y_pred or y_true):\n" + report_svm_str)

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    custom_print_func("SVM Confusion Matrix on Test Set:\n" + str(cm_svm))

    roc_auc_svm = roc_auc_score(y_test, y_proba_svm)
    pr_auc_svm = average_precision_score(y_test, y_proba_svm)
    custom_print_func(f"\nSVM AUC-ROC on Test Set: {roc_auc_svm:.4f}")
    custom_print_func(f"SVM AUC-PR (Average Precision) on Test Set: {pr_auc_svm:.4f}")

    fig_roc_svm = plt.figure(figsize=(8, 6))
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
    plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC-ROC = {roc_auc_svm:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('SVM ROC Curve')
    plt.legend(loc='lower right');
    plt.grid(True)
    save_plot_func(fig_roc_svm, gcs_bucket_name, gcs_output_prefix, "svm_roc_curve")

    fig_pr_svm = plt.figure(figsize=(8, 6))
    precision_svm_curve, recall_svm_curve, _ = precision_recall_curve(y_test, y_proba_svm)
    plt.plot(recall_svm_curve, precision_svm_curve, label=f'SVM (AUC-PR = {pr_auc_svm:.2f})')
    plt.xlabel('Recall (Sensitivity)');
    plt.ylabel('Precision');
    plt.title('SVM Precision-Recall Curve')
    plt.legend(loc='lower left');
    plt.grid(True)
    save_plot_func(fig_pr_svm, gcs_bucket_name, gcs_output_prefix, "svm_pr_curve")

    # --- Save SVM Model to GCS ---
    custom_print_func("\nSaving SVM model to GCS...")
    try:
        model_bytes = pickle.dumps(best_svm_model)
        model_blob_name = f"{gcs_output_prefix}/models/best_svm_model.pkl"
        upload_bytes_func(gcs_bucket_name, model_bytes, model_blob_name, content_type='application/octet-stream')
    except Exception as e:
        custom_print_func(f"ERROR saving SVM model to GCS: {e}")

    # --- SVM Interpretability with SHAP ---
    custom_print_func("\nSVM Interpretability using SHAP:")
    custom_print_func("Note: SHAP with KernelExplainer can be computationally intensive.")

    if processed_feature_names:
        X_train_shap_background_df = pd.DataFrame(X_train_resampled, columns=processed_feature_names)
        X_test_shap_sample_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)
    else:
        X_train_shap_background_df = X_train_resampled
        X_test_shap_sample_df = X_test_processed
        custom_print_func("Warning: Feature names not available for SHAP plots; using column indices.")

    nsamples_background = min(100, X_train_shap_background_df.shape[0])
    if nsamples_background > 0:
        background_data = shap.sample(X_train_shap_background_df, nsamples_background, random_state=42)
        try:
            explainer_svm = shap.KernelExplainer(best_svm_model.predict_proba, background_data)
            custom_print_func("SHAP KernelExplainer initialized.")
            nsamples_test_shap = min(50, X_test_shap_sample_df.shape[0])
            if nsamples_test_shap > 0:
                if isinstance(X_test_shap_sample_df, pd.DataFrame):
                    test_data_for_shap = X_test_shap_sample_df.sample(nsamples_test_shap, random_state=42)
                else:
                    indices = np.random.choice(X_test_shap_sample_df.shape[0], nsamples_test_shap, replace=False)
                    test_data_for_shap = X_test_shap_sample_df[indices]

                custom_print_func(f"Calculating SHAP values for {nsamples_test_shap} SVM test instances...")
                start_time_shap_svm = time.time()
                shap_values_svm = explainer_svm.shap_values(test_data_for_shap, nsamples='auto')
                custom_print_func(f"SVM SHAP values calculation time: {time.time() - start_time_shap_svm:.2f} seconds")

                shap_values_for_positive_class = shap_values_svm[1] if isinstance(shap_values_svm, list) and len(
                    shap_values_svm) == 2 else shap_values_svm

                custom_print_func("\nGenerating SVM SHAP Summary Plot (Feature Importance)...")
                fig_shap_summary_svm = plt.figure()
                shap.summary_plot(shap_values_for_positive_class, test_data_for_shap,
                                  feature_names=processed_feature_names if processed_feature_names else None,
                                  show=False, plot_size=None)
                plt.title("SHAP Summary Plot for SVM (Fraud Class)")
                save_plot_func(fig_shap_summary_svm, gcs_bucket_name, gcs_output_prefix, "svm_shap_summary_plot")
            else:
                custom_print_func("Not enough test samples to generate SHAP explanations for SVM.")
        except Exception as e:
            custom_print_func(f"Error during SVM SHAP value calculation or plotting: {e}")
    else:
        custom_print_func("Not enough background samples for SHAP KernelExplainer for SVM.")

    custom_print_func("\n--- SVM Pipeline Complete ---")
