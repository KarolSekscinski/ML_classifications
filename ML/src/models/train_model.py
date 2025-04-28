# -*- coding: utf-8 -*-
import logging
import logging.handlers
import sys
import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline # Import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin # For custom transformer

# XGBoost
from xgboost import XGBClassifier

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim

# --- Skorch Wrapper ---
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

# Import configuration and utility functions
from src import config
from src.features import build_features
from src.models import utils

# --- Logging Setup ---
# Ensure log directory exists
config.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE_PATH, mode='a'), # Log to file (append mode)
        logging.StreamHandler(sys.stdout) # Keep logging to console
    ]
)
logger = logging.getLogger(__name__) # Get logger for this module


# --- Custom Transformer for Float32 Conversion ---
class Float32Transformer(BaseEstimator, TransformerMixin):
    """Converts input data to numpy float32."""
    def fit(self, X, y=None):
        # No fitting necessary
        return self

    def transform(self, X):
        # Convert to float32
        # Check if input is sparse, convert to dense if necessary
        if hasattr(X, "toarray"): # Checks for sparse matrix
             X = X.toarray()
        Xt = X.astype(np.float32)
        logger.debug(f"Transformed data to dtype: {Xt.dtype}")
        return Xt

# --- PyTorch MLP Model Definition ---
class PyTorchMLP(nn.Module):
    """Simple PyTorch MLP for binary classification."""
    def __init__(self, n_features_in, hidden_layer_sizes=(64, 32), activation_fn=nn.ReLU, dropout_rate=0.3):
        super().__init__()
        if not isinstance(n_features_in, int) or n_features_in <= 0:
             raise ValueError(f"n_features_in must be a positive integer, got {n_features_in}")

        layers = []
        current_dim = n_features_in
        for hidden_units in hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, hidden_units))
            if callable(activation_fn):
                 layers.append(activation_fn())
            else:
                 logger.warning(f"Activation function {activation_fn} is not callable, using ReLU.")
                 layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_units
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)
        logger.info(f"Initialized PyTorchMLP with n_features_in={n_features_in}")

    def forward(self, X):
        # Input X should now be FloatTensor thanks to Float32Transformer & skorch
        return self.network(X)

# --- Main Execution ---
def main():
    """
    Main function using PyTorch MLP via skorch. Includes fix for dtype mismatch
    and EarlyStopping monitor.
    """
    logger.info('Starting model training pipeline (with PyTorch MLP)...')

    # 1. Load Data
    df = utils.load_data(config.RAW_DATA_PATH)

    # 2. Engineer Features
    df_engineered = build_features.engineer_features(df)

    # 3. Preprocess Data
    X, y, numerical_features, categorical_features, positive_label = build_features.preprocess_data(
        df_engineered
    )

    # 4. Split Data
    X_train, X_test, y_train, y_test = utils.split_data(X, y)

    # 5. Create Preprocessor
    preprocessor = build_features.create_preprocessor(numerical_features, categorical_features)

    # --- Determine n_features_in AFTER preprocessing ---
    logger.info("Fitting preprocessor on X_train to determine output feature dimension...")
    try:
        # Fit requires X, y is optional but can help if preprocessor uses target
        y_fit = y_train if isinstance(y_train, pd.Series) else None
        preprocessor.fit(X_train, y_fit)
        # Transform a small sample to get the output shape
        X_train_transformed_sample = preprocessor.transform(X_train.head(1))
        n_output_features = X_train_transformed_sample.shape[1]
        if n_output_features <= 0:
             raise ValueError("Preprocessor output 0 features.")
        logger.info(f"Determined preprocessor output feature dimension: {n_output_features}")
    except Exception as e:
        logger.error(f"Failed to fit preprocessor or determine output features: {e}", exc_info=True)
        return

    # --- Define Models and Parameter Grids ---
    cv_strategy = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    # --- FIX: Monitor 'train_loss' for EarlyStopping within GridSearchCV ---
    skorch_early_stopping = EarlyStopping(
        monitor='train_loss', # Monitor training loss as valid_loss isn't generated by default here
        patience=10,
        threshold=1e-4,
        threshold_mode='rel',
        lower_is_better=True
    )
    logger.info("Configured skorch EarlyStopping to monitor 'train_loss'.")


    models_and_params = {
        'svm': {
            'model_instance': SVC(probability=True, random_state=config.RANDOM_STATE),
            'params': { 'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 'auto'], 'classifier__kernel': ['rbf'] },
            'cv': cv_strategy,
            'use_generic_pipeline': True
        },
        'xgboost': {
            'model_instance': XGBClassifier( objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=config.RANDOM_STATE ),
            'params': { 'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.05, 0.1], 'classifier__max_depth': [3, 5], },
            'cv': cv_strategy,
            'use_generic_pipeline': True
        },
        'pytorch_mlp': {
            'model_instance': NeuralNetClassifier(
                module=PyTorchMLP,
                module__n_features_in=n_output_features,
                criterion=nn.BCEWithLogitsLoss,
                optimizer=optim.Adam,
                max_epochs=100,
                batch_size=32,
                train_split=None, # Crucial: GridSearchCV handles splitting
                callbacks=[('early_stopping', skorch_early_stopping)], # Use updated callback
                verbose=0
            ),
            'params': {
                'classifier__module__hidden_layer_sizes': [(64,), (64, 32)],
                'classifier__module__dropout_rate': [0.2, 0.4],
                'classifier__lr': [0.001, 0.0005],
                'classifier__batch_size': [32, 64],
            },
             'cv': cv_strategy,
             'use_generic_pipeline': False
        }
    }

    # --- Loop through models: create pipeline, tune, save, evaluate ---
    for model_name, model_config in models_and_params.items():
        logger.info(f"--- Processing Model: {model_name.upper()} ---")
        try:
            # --- Create the appropriate pipeline ---
            if model_config.get('use_generic_pipeline', False):
                pipeline = utils.create_model_pipeline(preprocessor, model_config['model_instance'])
                logger.info(f"Using generic pipeline for {model_name}")
            elif model_name == 'pytorch_mlp':
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('to_float32', Float32Transformer()),
                    ('classifier', model_config['model_instance'])
                ])
                logger.info(f"Using custom pipeline with Float32Transformer for {model_name}")
            else:
                 logger.error(f"Unknown pipeline configuration for model: {model_name}")
                 continue

            current_cv = model_config.get('cv', config.CV_FOLDS)
            # Ensure y_train is numpy array for skorch/GridSearchCV
            y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.values
            # Also ensure y_train is correct shape (N,) or (N, 1) depending on loss/model
            if y_train_np.ndim == 1:
                 y_train_np = y_train_np.reshape(-1, 1) # Reshape if 1D

            best_pipeline = utils.tune_pipeline(
                pipeline=pipeline,
                param_grid=model_config['params'],
                X_train=X_train,
                y_train=y_train_np, # Pass numpy array
                cv=current_cv
            )

            output_path = config.MODELS_DIR / f"{model_name}_tuned_pipeline.pkl"
            utils.save_pipeline(best_pipeline, output_path)

            # Ensure y_test is numpy for evaluation
            y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.values
            if y_test_np.ndim == 1:
                 y_test_np = y_test_np.reshape(-1, 1) # Reshape if 1D

            utils.evaluate_pipeline(best_pipeline, X_test, y_test_np, model_name, positive_label)

        except Exception as e:
            logger.error(f"Failed to process model {model_name}. Error: {e}", exc_info=True)
        finally:
             logger.info(f"--- Finished Processing Model: {model_name.upper()} ---")

    logger.info('Model training pipeline finished.')

if __name__ == '__main__':
    main()
