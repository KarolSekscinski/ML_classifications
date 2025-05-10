import time
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import shap


# Assuming gcs_utils contains custom_print, save_plot_to_gcs, upload_bytes_to_gcs
# These would be imported in the main script and passed as arguments.

def run_xgboost_pipeline(X_train_resampled, y_train_resampled, X_test_processed, y_test,
                         processed_feature_names, gcs_bucket_name, gcs_output_prefix,
                         custom_print_func, save_plot_func, upload_bytes_func):
    """
    Runs the XGBoost model training, tuning, evaluation, and SHAP interpretation.
    """
    custom_print_func("\n--- XGBoost Model Training and Hyperparameter Tuning ---")

    param_grid_xgb = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'gamma': [0, 0.1],
        'tree_method': ['gpu_hist']
    }
    cv_stratified = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Reduced folds for speed

    xgb_model_base = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='aucpr'
    )

    grid_search_xgb = GridSearchCV(
        estimator=xgb_model_base,
        param_grid=param_grid_xgb,
        scoring='average_precision',
        cv=cv_stratified,
        verbose=1,
        n_jobs=-1
    )

    custom_print_func("Starting GridSearchCV for XGBoost...")
    start_time_xgb_train = time.time()
    try:
        grid_search_xgb.fit(X_train_resampled, y_train_resampled)
        xgb_training_time = time.time() - start_time_xgb_train
        best_xgb_model = grid_search_xgb.best_estimator_

        custom_print_func("\nBest XGBoost Parameters found by GridSearchCV:")
        custom_print_func(grid_search_xgb.best_params_)
        custom_print_func(f"Best XGBoost Average Precision (PR AUC) score during CV: {grid_search_xgb.best_score_:.4f}")
        custom_print_func(f"XGBoost Training Time (including GridSearchCV): {xgb_training_time:.2f} seconds")

        # --- Evaluate Best XGBoost Model on Test Set ---
        custom_print_func("\n--- Evaluating Best XGBoost Model on Test Set ---")
        y_pred_xgb = best_xgb_model.predict(X_test_processed)
        y_proba_xgb = best_xgb_model.predict_proba(X_test_processed)[:, 1]

        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        precision_xgb_fraud = precision_score(y_test, y_pred_xgb, pos_label=1, zero_division=0)
        recall_xgb_fraud = recall_score(y_test, y_pred_xgb, pos_label=1, zero_division=0)
        f1_xgb_fraud = f1_score(y_test, y_pred_xgb, pos_label=1, zero_division=0)

        custom_print_func("\nXGBoost Detailed Metrics on Test Set:")
        custom_print_func(f"Accuracy: {accuracy_xgb:.4f}")
        custom_print_func(f"Precision (fraud): {precision_xgb_fraud:.4f}")
        custom_print_func(f"Recall/Sensitivity (fraud): {recall_xgb_fraud:.4f}")
        custom_print_func(f"F1-score (fraud): {f1_xgb_fraud:.4f}")
        try:
            report_xgb_str = classification_report(y_test, y_pred_xgb, target_names=['Legit (0)', 'Fraud (1)'],
                                                   zero_division=0)
            custom_print_func("\nXGBoost Classification Report on Test Set:\n" + report_xgb_str)
        except ValueError:
            report_xgb_str = classification_report(y_test, y_pred_xgb, zero_division=0)
            custom_print_func(
                "\nXGBoost Classification Report on Test Set (single class in y_pred or y_true):\n" + report_xgb_str)

        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
        custom_print_func("XGBoost Confusion Matrix on Test Set:\n" + str(cm_xgb))
        roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
        pr_auc_xgb = average_precision_score(y_test, y_proba_xgb)
        custom_print_func(f"\nXGBoost AUC-ROC on Test Set: {roc_auc_xgb:.4f}")
        custom_print_func(f"XGBoost AUC-PR (Average Precision) on Test Set: {pr_auc_xgb:.4f}")

        fig_roc_xgb = plt.figure(figsize=(8, 6))
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC-ROC = {roc_auc_xgb:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate');
        plt.title('XGBoost ROC Curve')
        plt.legend(loc='lower right');
        plt.grid(True)
        save_plot_func(fig_roc_xgb, gcs_bucket_name, gcs_output_prefix, "xgb_roc_curve")

        fig_pr_xgb = plt.figure(figsize=(8, 6))
        precision_xgb_curve, recall_xgb_curve, _ = precision_recall_curve(y_test, y_proba_xgb)
        plt.plot(recall_xgb_curve, precision_xgb_curve, label=f'XGBoost (AUC-PR = {pr_auc_xgb:.2f})')
        plt.xlabel('Recall (Sensitivity)');
        plt.ylabel('Precision');
        plt.title('XGBoost Precision-Recall Curve')
        plt.legend(loc='lower left');
        plt.grid(True)
        save_plot_func(fig_pr_xgb, gcs_bucket_name, gcs_output_prefix, "xgb_pr_curve")

        # --- Save XGBoost Model to GCS ---
        custom_print_func("\nSaving XGBoost model to GCS...")
        try:
            model_xgb_bytes = pickle.dumps(best_xgb_model)
            model_xgb_blob_name = f"{gcs_output_prefix}/models/best_xgb_model.pkl"
            upload_bytes_func(gcs_bucket_name, model_xgb_bytes, model_xgb_blob_name,
                              content_type='application/octet-stream')
        except Exception as e:
            custom_print_func(f"ERROR saving XGBoost model to GCS: {e}")

        # --- XGBoost Interpretability with SHAP ---
        custom_print_func("\nXGBoost Interpretability using SHAP:")
        custom_print_func("Using SHAP TreeExplainer for XGBoost.")
        try:
            # For TreeExplainer, it's often better to pass the original model before it's wrapped by GridSearchCV,
            # or ensure the best_estimator_ is the raw XGBoost model.
            # Also, background data (X_train_resampled) can be helpful for TreeExplainer.
            explainer_xgb = shap.TreeExplainer(best_xgb_model, data=X_train_resampled, model_output="probability")
            custom_print_func("SHAP TreeExplainer initialized for XGBoost.")

            nsamples_test_shap_xgb = min(200, X_test_processed.shape[0])
            if nsamples_test_shap_xgb > 0:
                if processed_feature_names and isinstance(X_test_processed,
                                                          np.ndarray):  # Convert to DataFrame if names available
                    test_data_for_shap_xgb_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)
                    test_data_for_shap_xgb = test_data_for_shap_xgb_df.sample(nsamples_test_shap_xgb, random_state=42)

                elif isinstance(X_test_processed, pd.DataFrame):
                    test_data_for_shap_xgb = X_test_processed.sample(nsamples_test_shap_xgb, random_state=42)
                else:  # Fallback to numpy array if no names or already numpy
                    indices_xgb = np.random.choice(X_test_processed.shape[0], nsamples_test_shap_xgb, replace=False)
                    test_data_for_shap_xgb = X_test_processed[indices_xgb]

                custom_print_func(f"Calculating SHAP values for {nsamples_test_shap_xgb} XGBoost test instances...")
                start_time_shap_xgb = time.time()
                # If test_data_for_shap_xgb is numpy, feature_names might be needed by shap_values if not inferred
                shap_values_xgb = explainer_xgb.shap_values(test_data_for_shap_xgb)
                custom_print_func(
                    f"XGBoost SHAP values calculation time: {time.time() - start_time_shap_xgb:.2f} seconds")

                if isinstance(shap_values_xgb, list) and len(shap_values_xgb) == 2:
                    shap_values_xgb_positive = shap_values_xgb[1]
                else:
                    shap_values_xgb_positive = shap_values_xgb

                custom_print_func("\nGenerating XGBoost SHAP Summary Plot (Feature Importance)...")
                fig_shap_summary_xgb = plt.figure()
                shap.summary_plot(shap_values_xgb_positive, test_data_for_shap_xgb,
                                  feature_names=processed_feature_names if processed_feature_names else None,
                                  show=False, plot_size=None)
                plt.title("SHAP Summary Plot for XGBoost (Fraud Class)")
                save_plot_func(fig_shap_summary_xgb, gcs_bucket_name, gcs_output_prefix, "xgb_shap_summary_plot")
            else:
                custom_print_func("Not enough test samples to generate SHAP explanations for XGBoost.")
        except Exception as e:
            custom_print_func(f"Error during XGBoost SHAP value calculation or plotting: {e}")

    except Exception as e:
        custom_print_func(f"An error occurred during XGBoost training/tuning: {e}")
        custom_print_func("Skipping XGBoost evaluation and SHAP.")

    custom_print_func("\n--- XGBoost Pipeline Complete ---")
