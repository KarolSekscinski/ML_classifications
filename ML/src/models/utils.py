# -*- coding: utf-8 -*-
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np # Added numpy import
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    make_scorer # If needed for custom scoring
)

# Import configuration constants
from src import config

logger = logging.getLogger(__name__)

# --- Load Data --- (Keep as before)
def load_data(file_path: Path) -> pd.DataFrame:
    """Loads data from a CSV file."""
    logger.info(f'Loading data from {file_path}')
    try:
        df = pd.read_csv(file_path)
        logger.info(f'Data loaded successfully. Shape: {df.shape}')
        return df
    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {file_path}")
        raise # Re-raise to stop execution

# --- Split Data (Modified for PyTorch target type) ---
def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Splits data into training and testing sets based on config.
    Ensures target 'y' is float32 for PyTorch BCE loss compatibility.
    """
    logger.info(f'Splitting data (test_size={config.TEST_SIZE}, random_state={config.RANDOM_STATE})')
    try:
        # Ensure y is suitable for stratify (no NaNs)
        if y.isnull().any():
             logger.warning("Target variable 'y' contains NaN values before split. Attempting to drop them.")
             valid_indices = y.dropna().index
             X = X.loc[valid_indices]
             y = y.loc[valid_indices]
             if X.empty:
                 raise ValueError("No valid data remaining after dropping NaNs in target.")

        # --- Convert target to float32 for PyTorch BCEWithLogitsLoss ---
        # Also reshape to (n_samples, 1) as often expected by PyTorch loss functions
        y_processed = y.astype(np.float32).values.reshape(-1, 1)
        logger.info(f"Converted target 'y' to numpy array of type {y_processed.dtype} and shape {y_processed.shape}")


        X_train, X_test, y_train, y_test = train_test_split(
            X, y_processed, # Use processed y
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y # Stratify based on the *original* y before reshaping/type change
        )
        logger.info(f'Training set shape: X={X_train.shape}, y={y_train.shape}')
        logger.info(f'Test set shape: X={X_test.shape}, y={y_test.shape}')
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during train/test split: {e}", exc_info=True)
        logger.error(f"Input shapes - X: {X.shape}, y: {y.shape}")
        if y.isnull().any():
            logger.error("Original target variable 'y' contains NaN values.")
        if len(y.unique()) < 2:
             logger.error(f"Original target variable 'y' has < 2 unique values: {y.unique()}. Stratify requires >= 2.")
        raise

# --- Create Model Pipeline --- (Keep as before)
def create_model_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Combines a preprocessor and a model into a single pipeline."""
    model_name = type(model).__name__
    logger.info(f'Creating full pipeline with preprocessor and model: {model_name}')
    if model is None:
        raise ValueError("Model provided to create_model_pipeline cannot be None.")
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model) # Step name 'classifier' is used in param_grid
    ])

# --- Tune Pipeline (Modified for CV object) ---
def tune_pipeline(pipeline: Pipeline, param_grid: dict, X_train: pd.DataFrame, y_train: np.ndarray, cv) -> Pipeline:
    """
    Tunes the pipeline hyperparameters using GridSearchCV based on config.
    Accepts a cross-validation strategy object (cv).
    Accepts y_train as numpy array.
    Returns the best fitted pipeline found.
    """
    # Skorch expects numpy arrays or tensors, not pandas Series for y
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
        logger.warning("y_train was a pandas Series, converted to numpy array for tuning.")
    # Ensure y_train is the correct shape/type if needed (skorch usually handles conversion)


    model_name = pipeline.steps[-1][1].__class__.__name__ # Get model name from pipeline
    logger.info(f'Tuning hyperparameters for {model_name} using GridSearchCV...')
    logger.info(f'Parameter grid: {param_grid}')
    if isinstance(cv, int):
        logger.info(f'Cross-validation folds: {cv} (default stratification)')
    else:
         logger.info(f'Cross-validation strategy: {type(cv).__name__} with {cv.get_n_splits()} splits')
    logger.info(f'Scoring metric: {config.TUNING_SCORING}')


    start_time = time.time()
    try:
        # Ensure X_train is suitable (Pandas DataFrame is usually fine as skorch converts)
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv, # Use the passed CV strategy/object
            scoring=config.TUNING_SCORING,
            n_jobs=-1, # Use all available CPU cores
            verbose=1,
            error_score='raise', # Raise errors during CV fitting
            refit=True # Ensure the best model is refit on the whole training data
        )
        search.fit(X_train, y_train) # Fit requires numpy arrays or pandas DataFrames usually
    except Exception as e:
         logger.error(f"Error during GridSearchCV for {model_name}: {e}", exc_info=True)
         logger.error(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
         # logger.error(f"Target value counts in training data:\n{pd.Series(y_train.ravel()).value_counts()}") # If y_train is numpy
         raise # Re-raise the exception

    end_time = time.time()

    logger.info(f'GridSearchCV finished for {model_name}. Time taken: {end_time - start_time:.2f} seconds.')
    logger.info(f'Best parameters found: {search.best_params_}')
    try:
        best_score = search.best_score_
        logger.info(f'Best {config.TUNING_SCORING} score on validation sets: {best_score:.4f}')
    except AttributeError:
        logger.warning("Could not retrieve best_score_ from GridSearchCV results.")

    if not hasattr(search, 'best_estimator_'):
         logger.error(f"GridSearchCV for {model_name} did not produce a best_estimator_.")
         raise ValueError(f"Could not find best estimator for {model_name}")

    return search.best_estimator_

# --- Save Pipeline --- (Keep as before)
def save_pipeline(pipeline: Pipeline, file_path: Path):
    """Saves the trained pipeline to a file using joblib."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving pipeline to {file_path}')
        # Note: Saving pipelines with skorch/pytorch models might require care
        # Joblib might work, but saving state_dict and architecture separately is safer
        # For simplicity here, we'll try joblib, but be aware it might be fragile.
        joblib.dump(pipeline, file_path)
        logger.info('Pipeline saved successfully (using joblib).')
    except Exception as e:
        logger.error(f"Error saving pipeline to {file_path}: {e}")
        # Consider alternative saving methods if joblib fails with PyTorch models
        raise

# --- Evaluate Pipeline (Modified for PyTorch target type) ---
def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray, model_name: str, positive_label: int):
    """
    Evaluates the final pipeline on the test set and logs detailed metrics.
    Accepts y_test as numpy array. Uses the positive_label for metric calculations.
    """
    logger.info(f'--- Evaluating final {model_name} pipeline on the test set ---')
    try:
        # Ensure y_test is in the correct format (1D array) for scikit-learn metrics
        if y_test.ndim > 1 and y_test.shape[1] == 1:
            y_test_eval = y_test.ravel()
        else:
            y_test_eval = y_test # Assume it's already 1D or metrics handle it

        # Get predictions (skorch usually returns class labels)
        y_pred = pipeline.predict(X_test)

        # ROC AUC
        roc_auc = None
        if hasattr(pipeline, "predict_proba"):
            try:
                # skorch predict_proba returns probabilities for each class [prob_0, prob_1]
                y_pred_proba = pipeline.predict_proba(X_test)

                # Ensure y_pred_proba has 2 columns for binary case
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
                     # Get probability of the positive class (usually index 1)
                     positive_class_index = 1 # Assuming [class_0, class_1] output
                     # Check pipeline.classes_ if available and needed
                     # classes_list = list(getattr(pipeline, 'classes_', [0, 1]))
                     # positive_class_index = classes_list.index(positive_label)

                     roc_auc = roc_auc_score(y_test_eval, y_pred_proba[:, positive_class_index])
                     logger.info(f'{model_name} Test ROC AUC: {roc_auc:.4f}')
                elif y_pred_proba.ndim == 1: # Handle case where it might return only positive class prob
                     roc_auc = roc_auc_score(y_test_eval, y_pred_proba)
                     logger.info(f'{model_name} Test ROC AUC: {roc_auc:.4f}')
                else:
                    logger.warning(f"Unexpected shape for predict_proba output: {y_pred_proba.shape}. Skipping ROC AUC.")

            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC for {model_name}. Error: {e}", exc_info=True)
        else:
            logger.warning(f"Model {model_name} or pipeline does not support predict_proba. Skipping ROC AUC.")

        # Core metrics
        accuracy = accuracy_score(y_test_eval, y_pred)
        precision = precision_score(y_test_eval, y_pred, pos_label=positive_label, zero_division=0)
        recall = recall_score(y_test_eval, y_pred, pos_label=positive_label, zero_division=0)
        f1 = f1_score(y_test_eval, y_pred, pos_label=positive_label, zero_division=0)
        conf_matrix = confusion_matrix(y_test_eval, y_pred)

        logger.info(f'{model_name} Test Accuracy: {accuracy:.4f}')
        logger.info(f'{model_name} Test Precision (Class {positive_label}): {precision:.4f}')
        logger.info(f'{model_name} Test Recall (Class {positive_label}): {recall:.4f}')
        logger.info(f'{model_name} Test F1-Score (Class {positive_label}): {f1:.4f}')
        logger.info(f'{model_name} Confusion Matrix:\n{conf_matrix}')
        report = classification_report(y_test_eval, y_pred, zero_division=0)
        logger.info(f'{model_name} Classification Report:\n{report}')

    except Exception as e:
        logger.error(f"Error during evaluation of {model_name}: {e}", exc_info=True)
        logger.error(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
        # logger.error(f"Target value counts in test data:\n{pd.Series(y_test_eval).value_counts()}")

    finally:
        logger.info(f'--- Finished evaluating {model_name} ---')
