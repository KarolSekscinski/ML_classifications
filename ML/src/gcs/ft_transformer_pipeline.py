import time
import pickle
import pandas as pd
import numpy as np

import torch  # PyTorch
import torch.nn as nn
import torch.optim as optim
from rtdl.modules import FTTransformer  # rtdl library for FT-Transformer
from sklearn.impute import SimpleImputer
from skorch import NeuralNetClassifier  # skorch for scikit-learn compatibility

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

def run_ft_transformer_pipeline(
        X_train_resampled, y_train_resampled, X_test_processed, y_test,
        processed_feature_names,  # Needed for SHAP if X_test_processed is numpy
        numerical_features, categorical_features,  # Needed to determine cat_cardinalities for FT-Transformer
        X_train,  # Original X_train to determine cardinalities before OHE
        gcs_bucket_name, gcs_output_prefix,
        custom_print_func, save_plot_func, upload_bytes_func
):
    """
    Runs the FT-Transformer model training, tuning, evaluation, and SHAP interpretation.
    """
    custom_print_func("\n--- FT-Transformer Model Training and Hyperparameter Tuning ---")

    # FT-Transformer specific preprocessing: Determine cardinalities for categorical features
    # This needs to be done on the data *before* one-hot encoding but *after* imputation.
    # For simplicity, if X_train (original, non-OHE'd) is passed, we can infer from it.
    # Otherwise, this step would need careful handling of how categorical features were encoded.

    cat_cardinalities = []
    if categorical_features:  # Only if there are categorical features defined
        # Create a temporary preprocessor for just imputing categoricals to get cardinalities
        temp_cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train_cat_imputed = pd.DataFrame(temp_cat_imputer.fit_transform(X_train[categorical_features]),
                                           columns=categorical_features)

        for col in categorical_features:
            # Ensure imputed data is treated as object/category for nunique
            cat_cardinalities.append(X_train_cat_imputed[col].astype('category').nunique())

    custom_print_func(f"Categorical features for FT-Transformer: {categorical_features}")
    custom_print_func(f"Determined cardinalities for categorical features: {cat_cardinalities}")

    # Ensure y_train_resampled and y_test are LongTensors for PyTorch
    y_train_torch = torch.tensor(y_train_resampled, dtype=torch.long)
    y_test_torch = torch.tensor(y_test.values, dtype=torch.long)  # y_test is pandas Series

    # Convert X data to float32, which is standard for PyTorch tensors
    X_train_torch = torch.tensor(X_train_resampled.astype(np.float32))
    X_test_torch = torch.tensor(X_test_processed.astype(np.float32))

    d_out = len(np.unique(y_train_resampled))  # Number of output classes (should be 2 for binary)

    # Define the parameter grid for FT-Transformer (via skorch)
    # This grid is simplified for demonstration. FT-Transformer has many more params.
    param_grid_ft = {
        'lr': [0.0001, 0.001],  # Learning rate
        'batch_size': [128, 256],  # Batch size for training
        'max_epochs': [10, 20],  # Reduced epochs for faster demo
        # FT-Transformer specific module parameters (prefixed with 'module__')
        'module__n_blocks': [2, 3],  # Number of Transformer blocks
        'module__d_token': [96, 128],  # Dimension of tokens/embeddings
        # 'module__ffn_d_hidden': [192], # Dimension of hidden layer in FFN
        # 'module__attention_dropout': [0.1, 0.2],
        # 'module__ffn_dropout': [0.0, 0.1],
    }
    # For a very quick demo:
    param_grid_ft_demo = {
        'lr': [0.001],
        'batch_size': [256],
        'max_epochs': [5],  # Very few epochs for demo
        'module__n_blocks': [1],
        'module__d_token': [64],
    }

    # Initialize FT-Transformer within skorch's NeuralNetClassifier
    # The module needs to be instantiated with parameters that are NOT tuned by GridSearchCV here,
    # or they need to be part of the search space.
    # d_in is the number of features after preprocessing.
    d_in = X_train_resampled.shape[1]

    # Ensure numerical_features only includes those present after preprocessing
    # The FTTransformer module expects num_numerical_features count
    # If processed_feature_names is available and accurate, we can use it.
    # Otherwise, we assume X_train_resampled columns are ordered: numerical, OHE_categorical, binary_passthrough

    # Infer num_numerical_features:
    # This is tricky because X_train_resampled is already processed.
    # We need to know how many of its columns correspond to the original numerical features.
    # If processed_feature_names is correctly reconstructed, we can count.
    # A simpler way for FT-Transformer is to tell it how many *original* numerical features there were.
    # The FT-Transformer's tokenizer will handle numerical and categorical features differently.
    # It's often easier to pass X as a dict {'numerical': X_num, 'categorical': X_cat} to FT-Transformer
    # or use its built-in tokenizer which requires original numerical and categorical data.
    # For this setup with a pre-processed X_train_resampled (all numeric after OHE),
    # we treat all input features as 'numerical' from FT-Transformer's perspective if we don't use its tokenizer.
    # This is a simplification. A proper FT-Transformer setup would use its tokenizer.

    # Simplified approach: Treat all columns in X_train_resampled as numerical inputs to a generic transformer
    # This is NOT how FT-Transformer is typically used with its special tokenizer.
    # A more correct approach would involve rtdl.NumericalFeatureTokenizer and rtdl.CategoricalFeatureTokenizer
    # or passing original numerical and categorical data to a custom skorch module.

    # For this example, we'll pass cat_cardinalities. FTTransformer expects this.
    # It also expects `d_numerical` if numerical features are present and tokenized separately.
    # If all features are passed as one block (after OHE), then `d_numerical` would be 0,
    # and all features would be treated as if they are to be tokenized by the categorical tokenizer,
    # which is not ideal.

    # Let's assume X_train_resampled has numerical features first, then OHE categorical.
    # The FTTransformer module should be configured with the number of original numerical features
    # and the cardinalities of the original categorical features.

    # For FTTransformer, it's best to use its own tokenization.
    # However, to fit into the current preprocessor pipeline, we'd need a more complex skorch setup.
    # As a compromise for this example, we'll define a basic FT-Transformer.
    # Note: This will not leverage FT-Transformer's full potential without its specific tokenizers.

    net = NeuralNetClassifier(
        module=FTTransformer,
        module__d_numerical=len(numerical_features),  # Number of original numerical features
        module__cat_cardinalities=cat_cardinalities if cat_cardinalities else None,
        # Cardinalities of original categorical features
        module__d_out=d_out,  # Number of output classes
        # Other FT-Transformer params like n_blocks, d_token will be tuned by GridSearchCV
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        # train_split=None, # Use skorch's default or provide a validation set
        verbose=0,  # Skorch verbosity
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Use CUDA if available
    )
    custom_print_func(f"FT-Transformer will run on: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    cv_stratified = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # Reduced folds for speed

    grid_search_ft = GridSearchCV(
        estimator=net,
        param_grid=param_grid_ft_demo,  # Using demo grid; use param_grid_ft for full search
        scoring='average_precision',
        cv=cv_stratified,
        verbose=1,
        n_jobs=-1  # Can be problematic with CUDA if not handled carefully by skorch/GridSearchCV
    )

    custom_print_func("Starting GridSearchCV for FT-Transformer...")
    start_time_ft_train = time.time()
    try:
        # Important: FT-Transformer expects specific input format if using its tokenizers.
        # If passing pre-OHE'd data, it needs to be structured for the model.
        # Here, X_train_resampled is fully numeric after OHE.
        # We need to ensure the FTTransformer module is compatible with this flat numeric input.
        # The `rtdl.FTTransformer` expects either:
        # 1. x_numerical and x_categorical as separate inputs to its forward method.
        # 2. A setup where its internal tokenizers are used.
        # Skorch by default passes X as a single tensor.
        # To make this work, we might need a custom skorch module that splits X or uses
        # a version of FT-Transformer that accepts a single flat tensor and knows
        # which parts are numerical and which are categorical (e.g., via column indices).

        # For this example, we assume the FTTransformer module (or a wrapper) can handle
        # the flat X_train_resampled and internally distinguish based on d_numerical.
        # This is a simplification. A robust FT-Transformer setup is more involved.

        # Convert y_train_resampled to numpy for skorch compatibility if it's a tensor
        if isinstance(y_train_resampled, torch.Tensor):
            y_train_resampled_np = y_train_resampled.cpu().numpy()
        else:
            y_train_resampled_np = y_train_resampled

        grid_search_ft.fit(X_train_resampled.astype(np.float32),
                           y_train_resampled_np)  # Skorch expects numpy arrays or tensors

        ft_training_time = time.time() - start_time_ft_train
        best_ft_model = grid_search_ft.best_estimator_

        custom_print_func("\nBest FT-Transformer Parameters found by GridSearchCV:")
        custom_print_func(grid_search_ft.best_params_)
        custom_print_func(
            f"Best FT-Transformer Average Precision (PR AUC) score during CV: {grid_search_ft.best_score_:.4f}")
        custom_print_func(f"FT-Transformer Training Time (GridSearchCV): {ft_training_time:.2f} seconds")

        # --- Evaluate Best FT-Transformer Model on Test Set ---
        custom_print_func("\n--- Evaluating Best FT-Transformer Model on Test Set ---")
        y_pred_ft = best_ft_model.predict(X_test_torch)  # skorch handles tensor input for predict
        y_proba_ft = best_ft_model.predict_proba(X_test_torch)[:, 1]

        # Convert y_test_torch back to numpy for sklearn metrics
        y_test_np = y_test_torch.cpu().numpy()

        accuracy_ft = accuracy_score(y_test_np, y_pred_ft)
        precision_ft_fraud = precision_score(y_test_np, y_pred_ft, pos_label=1, zero_division=0)
        recall_ft_fraud = recall_score(y_test_np, y_pred_ft, pos_label=1, zero_division=0)  # Sensitivity
        f1_ft_fraud = f1_score(y_test_np, y_pred_ft, pos_label=1, zero_division=0)

        custom_print_func("\nFT-Transformer Detailed Metrics on Test Set:")
        custom_print_func(f"Accuracy: {accuracy_ft:.4f}")
        custom_print_func(f"Precision (fraud): {precision_ft_fraud:.4f}")
        custom_print_func(f"Recall/Sensitivity (fraud): {recall_ft_fraud:.4f}")
        custom_print_func(f"F1-score (fraud): {f1_ft_fraud:.4f}")

        try:
            report_ft_str = classification_report(y_test_np, y_pred_ft, target_names=['Legit (0)', 'Fraud (1)'],
                                                  zero_division=0)
            custom_print_func("\nFT-Transformer Classification Report on Test Set:\n" + report_ft_str)
        except ValueError:
            report_ft_str = classification_report(y_test_np, y_pred_ft, zero_division=0)
            custom_print_func(
                "\nFT-Transformer Classification Report on Test Set (single class in y_pred or y_true):\n" + report_ft_str)

        cm_ft = confusion_matrix(y_test_np, y_pred_ft)
        custom_print_func("FT-Transformer Confusion Matrix on Test Set:\n" + str(cm_ft))

        roc_auc_ft = roc_auc_score(y_test_np, y_proba_ft)
        pr_auc_ft = average_precision_score(y_test_np, y_proba_ft)
        custom_print_func(f"\nFT-Transformer AUC-ROC on Test Set: {roc_auc_ft:.4f}")
        custom_print_func(f"FT-Transformer AUC-PR (Average Precision) on Test Set: {pr_auc_ft:.4f}")

        fig_roc_ft = plt.figure(figsize=(8, 6))
        fpr_ft, tpr_ft, _ = roc_curve(y_test_np, y_proba_ft)
        plt.plot(fpr_ft, tpr_ft, label=f'FT-T (AUC-ROC = {roc_auc_ft:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate');
        plt.title('FT-Transformer ROC Curve')
        plt.legend(loc='lower right');
        plt.grid(True)
        save_plot_func(fig_roc_ft, gcs_bucket_name, gcs_output_prefix, "ft_transformer_roc_curve")

        fig_pr_ft = plt.figure(figsize=(8, 6))
        precision_ft_curve, recall_ft_curve, _ = precision_recall_curve(y_test_np, y_proba_ft)
        plt.plot(recall_ft_curve, precision_ft_curve, label=f'FT-T (AUC-PR = {pr_auc_ft:.2f})')
        plt.xlabel('Recall (Sensitivity)');
        plt.ylabel('Precision');
        plt.title('FT-Transformer Precision-Recall Curve')
        plt.legend(loc='lower left');
        plt.grid(True)
        save_plot_func(fig_pr_ft, gcs_bucket_name, gcs_output_prefix, "ft_transformer_pr_curve")

        # --- Save FT-Transformer Model to GCS ---
        custom_print_func("\nSaving FT-Transformer model to GCS...")
        try:
            # Skorch models can be pickled.
            model_bytes = pickle.dumps(best_ft_model)
            model_blob_name = f"{gcs_output_prefix}/models/best_ft_transformer_model.pkl"
            upload_bytes_func(gcs_bucket_name, model_bytes, model_blob_name, content_type='application/octet-stream')
        except Exception as e:
            custom_print_func(f"ERROR saving FT-Transformer model to GCS: {e}")

        # --- FT-Transformer Interpretability with SHAP ---
        custom_print_func("\nFT-Transformer Interpretability using SHAP:")
        custom_print_func("Note: SHAP for PyTorch models often uses DeepExplainer or GradientExplainer.")
        custom_print_func("KernelExplainer can be a fallback but is slow. TreeExplainer is not for NNs.")

        # For PyTorch models wrapped with skorch, SHAP can be tricky.
        # We need to ensure the explainer can work with the underlying PyTorch model or its predict_proba.
        # shap.Explainer might auto-detect, or we might need DeepExplainer.
        # DeepExplainer requires a background dataset of tensors.

        # Sample background data (from training set) and test data for SHAP
        nsamples_background_shap = min(100, X_train_resampled.shape[0])
        if nsamples_background_shap > 0:
            # Convert numpy to tensor for DeepExplainer if that's chosen by shap.Explainer
            background_data_torch = torch.tensor(
                shap.sample(X_train_resampled, nsamples_background_shap, random_state=42).astype(np.float32)).to(
                best_ft_model.device)

            nsamples_test_shap = min(50, X_test_processed.shape[0])
            if nsamples_test_shap > 0:
                test_data_for_shap_torch = torch.tensor(
                    shap.sample(X_test_processed, nsamples_test_shap, random_state=42).astype(np.float32)).to(
                    best_ft_model.device)

                try:
                    custom_print_func(f"Initializing SHAP Explainer for FT-Transformer (this might take a moment)...")
                    # Using shap.Explainer which should try to pick an appropriate one.
                    # For PyTorch, it might pick DeepExplainer or GradientExplainer.
                    # The model passed should be the underlying PyTorch model if possible, or predict_proba.
                    # Skorch's .module_ gives the PyTorch module.
                    # However, shap.Explainer might work better with the skorch wrapper directly if it understands its predict_proba.

                    # Option 1: Try with skorch wrapper (might fall back to KernelExplainer if not directly supported)
                    # explainer_ft = shap.Explainer(best_ft_model.predict_proba, background_data_torch)

                    # Option 2: Try with underlying PyTorch module (might need DeepExplainer explicitly)
                    # Need to handle the fact that predict_proba from skorch is not the same as forward from torch module
                    # For DeepExplainer, the model should be the torch module, and output should be logits or probabilities.
                    # This requires careful handling of the model's output format.

                    # Let's try KernelExplainer as a robust fallback for skorch-wrapped PyTorch models.
                    # It uses predict_proba, which skorch provides.
                    explainer_ft = shap.KernelExplainer(best_ft_model.predict_proba,
                                                        background_data.cpu().numpy())  # KernelExplainer needs numpy

                    custom_print_func(
                        f"Calculating SHAP values for {nsamples_test_shap} FT-Transformer test instances...")
                    start_time_shap_ft = time.time()
                    # KernelExplainer expects numpy array for explanation data
                    shap_values_ft = explainer_ft.shap_values(test_data_for_shap_torch.cpu().numpy(), nsamples='auto')
                    custom_print_func(
                        f"FT-Transformer SHAP values calculation time: {time.time() - start_time_shap_ft:.2f} seconds")

                    shap_values_for_positive_class_ft = shap_values_ft[1] if isinstance(shap_values_ft, list) and len(
                        shap_values_ft) == 2 else shap_values_ft

                    custom_print_func("\nGenerating FT-Transformer SHAP Summary Plot...")
                    fig_shap_summary_ft = plt.figure()
                    # For summary_plot, data can be DataFrame or numpy array. If numpy, feature_names are crucial.
                    # If test_data_for_shap_torch was created from a DataFrame with processed_feature_names:
                    if processed_feature_names and isinstance(test_data_for_shap_torch, torch.Tensor):
                        test_data_display = pd.DataFrame(test_data_for_shap_torch.cpu().numpy(),
                                                         columns=processed_feature_names)
                    else:
                        test_data_display = test_data_for_shap_torch.cpu().numpy()

                    shap.summary_plot(shap_values_for_positive_class_ft, test_data_display,
                                      feature_names=processed_feature_names if processed_feature_names else None,
                                      show=False, plot_size=None)
                    plt.title("SHAP Summary Plot for FT-Transformer (Fraud Class)")
                    save_plot_func(fig_shap_summary_ft, gcs_bucket_name, gcs_output_prefix,
                                   "ft_transformer_shap_summary_plot")
                except Exception as e:
                    custom_print_func(f"Error during FT-Transformer SHAP value calculation or plotting: {e}")
            else:
                custom_print_func("Not enough test samples to generate SHAP explanations for FT-Transformer.")
        else:
            custom_print_func("Not enough background samples for SHAP explainer for FT-Transformer.")

    except Exception as e:
        custom_print_func(f"An error occurred during FT-Transformer training/tuning: {e}")
        custom_print_func("Skipping FT-Transformer evaluation and SHAP.")

    custom_print_func("\n--- FT-Transformer Pipeline Complete ---")

