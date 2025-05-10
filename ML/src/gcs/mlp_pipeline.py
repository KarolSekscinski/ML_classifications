import time
import pickle
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
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

def run_mlp_pipeline(X_train_resampled, y_train_resampled, X_test_processed, y_test,
                     processed_feature_names, gcs_bucket_name, gcs_output_prefix,
                     custom_print_func, save_plot_func, upload_bytes_func):
    """
    Runs the MLP model training, tuning, evaluation, and SHAP interpretation.
    """
    custom_print_func("\n--- MLP Model Training and Hyperparameter Tuning ---")

    # Define the parameter grid for MLP
    # Note: This is a basic grid. For a full run, you might explore more layers, neurons, and solvers.
    # 'adam' solver is often a good default. 'alpha' is for L2 regularization.
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50, 25)],  # e.g., one layer of 50, one of 100, two layers
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],  # 'sgd' and 'lbfgs' are other options
        'alpha': [0.0001, 0.001, 0.01],  # L2 penalty (regularization term) parameter
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [300, 500],  # Max iterations for solver
        'early_stopping': [True],  # To prevent overfitting and speed up if convergence is slow
        'n_iter_no_change': [10]  # Number of iterations with no improvement to trigger early stopping
    }
    # For quicker demo, reduce grid:
    param_grid_mlp_demo = {
        'hidden_layer_sizes': [(50,)],
        'activation': ['relu'],
        'alpha': [0.001],
        'learning_rate_init': [0.001],
        'max_iter': [200],  # Reduced for faster demo
        'early_stopping': [True],
        'n_iter_no_change': [10]
    }

    cv_stratified = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Reduced folds for speed

    grid_search_mlp = GridSearchCV(
        estimator=MLPClassifier(random_state=42),
        param_grid=param_grid_mlp_demo,  # Using demo grid for speed; use param_grid_mlp for full search
        scoring='average_precision',
        cv=cv_stratified,
        verbose=1,
        n_jobs=-1
    )

    custom_print_func("Starting GridSearchCV for MLP...")
    start_time_mlp_train = time.time()
    grid_search_mlp.fit(X_train_resampled, y_train_resampled)
    mlp_training_time = time.time() - start_time_mlp_train
    best_mlp_model = grid_search_mlp.best_estimator_

    custom_print_func("\nBest MLP Parameters found by GridSearchCV:")
    custom_print_func(grid_search_mlp.best_params_)
    custom_print_func(f"Best MLP Average Precision (PR AUC) score during CV: {grid_search_mlp.best_score_:.4f}")
    custom_print_func(f"MLP Training Time (GridSearchCV): {mlp_training_time:.2f} seconds")

    # --- Evaluate Best MLP Model on Test Set ---
    custom_print_func("\n--- Evaluating Best MLP Model on Test Set ---")
    y_pred_mlp = best_mlp_model.predict(X_test_processed)
    y_proba_mlp = best_mlp_model.predict_proba(X_test_processed)[:, 1]

    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    precision_mlp_fraud = precision_score(y_test, y_pred_mlp, pos_label=1, zero_division=0)
    recall_mlp_fraud = recall_score(y_test, y_pred_mlp, pos_label=1, zero_division=0)  # Sensitivity
    f1_mlp_fraud = f1_score(y_test, y_pred_mlp, pos_label=1, zero_division=0)

    custom_print_func("\nMLP Detailed Metrics on Test Set:")
    custom_print_func(f"Accuracy: {accuracy_mlp:.4f}")
    custom_print_func(f"Precision (fraud): {precision_mlp_fraud:.4f}")
    custom_print_func(f"Recall/Sensitivity (fraud): {recall_mlp_fraud:.4f}")
    custom_print_func(f"F1-score (fraud): {f1_mlp_fraud:.4f}")

    try:
        report_mlp_str = classification_report(y_test, y_pred_mlp, target_names=['Legit (0)', 'Fraud (1)'],
                                               zero_division=0)
        custom_print_func("\nMLP Classification Report on Test Set:\n" + report_mlp_str)
    except ValueError:
        report_mlp_str = classification_report(y_test, y_pred_mlp, zero_division=0)
        custom_print_func(
            "\nMLP Classification Report on Test Set (single class in y_pred or y_true):\n" + report_mlp_str)

    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    custom_print_func("MLP Confusion Matrix on Test Set:\n" + str(cm_mlp))

    roc_auc_mlp = roc_auc_score(y_test, y_proba_mlp)
    pr_auc_mlp = average_precision_score(y_test, y_proba_mlp)
    custom_print_func(f"\nMLP AUC-ROC on Test Set: {roc_auc_mlp:.4f}")
    custom_print_func(f"MLP AUC-PR (Average Precision) on Test Set: {pr_auc_mlp:.4f}")

    fig_roc_mlp = plt.figure(figsize=(8, 6))
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_proba_mlp)
    plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC-ROC = {roc_auc_mlp:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MLP ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    save_plot_func(fig_roc_mlp, gcs_bucket_name, gcs_output_prefix, "mlp_roc_curve")

    fig_pr_mlp = plt.figure(figsize=(8, 6))
    precision_mlp_curve, recall_mlp_curve, _ = precision_recall_curve(y_test, y_proba_mlp)
    plt.plot(recall_mlp_curve, precision_mlp_curve, label=f'MLP (AUC-PR = {pr_auc_mlp:.2f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('MLP Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    save_plot_func(fig_pr_mlp, gcs_bucket_name, gcs_output_prefix, "mlp_pr_curve")

    # --- Save MLP Model to GCS ---
    custom_print_func("\nSaving MLP model to GCS...")
    try:
        model_bytes = pickle.dumps(best_mlp_model)
        model_blob_name = f"{gcs_output_prefix}/models/best_mlp_model.pkl"
        upload_bytes_func(gcs_bucket_name, model_bytes, model_blob_name, content_type='application/octet-stream')
    except Exception as e:
        custom_print_func(f"ERROR saving MLP model to GCS: {e}")

    # --- MLP Interpretability with SHAP ---
    custom_print_func("\nMLP Interpretability using SHAP:")
    custom_print_func("Note: SHAP with KernelExplainer for MLP can be computationally intensive.")

    if processed_feature_names:
        X_train_shap_background_df = pd.DataFrame(X_train_resampled, columns=processed_feature_names)
        X_test_shap_sample_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)
    else:
        X_train_shap_background_df = X_train_resampled
        X_test_shap_sample_df = X_test_processed
        custom_print_func("Warning: Feature names not available for SHAP plots; using column indices.")

    # For MLPs, KernelExplainer is generally used. It's model-agnostic.
    # Sample the background data (usually from training set)
    nsamples_background = min(100, X_train_shap_background_df.shape[0])
    if nsamples_background > 0:
        background_data_mlp = shap.sample(X_train_shap_background_df, nsamples_background, random_state=42)
        try:
            explainer_mlp = shap.KernelExplainer(best_mlp_model.predict_proba, background_data_mlp)
            custom_print_func("SHAP KernelExplainer initialized for MLP.")

            nsamples_test_shap = min(50, X_test_shap_sample_df.shape[0])
            if nsamples_test_shap > 0:
                if isinstance(X_test_shap_sample_df, pd.DataFrame):
                    test_data_for_shap_mlp = X_test_shap_sample_df.sample(nsamples_test_shap, random_state=42)
                else:
                    indices = np.random.choice(X_test_shap_sample_df.shape[0], nsamples_test_shap, replace=False)
                    test_data_for_shap_mlp = X_test_shap_sample_df[indices]

                custom_print_func(f"Calculating SHAP values for {nsamples_test_shap} MLP test instances...")
                start_time_shap_mlp = time.time()
                shap_values_mlp = explainer_mlp.shap_values(test_data_for_shap_mlp, nsamples='auto')
                custom_print_func(f"MLP SHAP values calculation time: {time.time() - start_time_shap_mlp:.2f} seconds")

                # shap_values_mlp[1] for the positive class (fraud)
                shap_values_for_positive_class_mlp = shap_values_mlp[1] if isinstance(shap_values_mlp, list) and len(
                    shap_values_mlp) == 2 else shap_values_mlp

                custom_print_func("\nGenerating MLP SHAP Summary Plot (Feature Importance)...")
                fig_shap_summary_mlp = plt.figure()
                shap.summary_plot(shap_values_for_positive_class_mlp, test_data_for_shap_mlp,
                                  feature_names=processed_feature_names if processed_feature_names else None,
                                  show=False, plot_size=None)
                plt.title("SHAP Summary Plot for MLP (Fraud Class)")
                save_plot_func(fig_shap_summary_mlp, gcs_bucket_name, gcs_output_prefix, "mlp_shap_summary_plot")
            else:
                custom_print_func("Not enough test samples to generate SHAP explanations for MLP.")
        except Exception as e:
            custom_print_func(f"Error during MLP SHAP value calculation or plotting: {e}")
    else:
        custom_print_func("Not enough background samples for SHAP KernelExplainer for MLP.")

    custom_print_func("\n--- MLP Pipeline Complete ---")
