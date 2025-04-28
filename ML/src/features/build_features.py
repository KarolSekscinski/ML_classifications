# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Import configuration constants - assumes running from project root
# Or adjust path if running script directly from src/features/
# from .. import config # Example relative import if needed
from src import config # Assumes src is in PYTHONPATH or running from root

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features from existing ones based on config."""
    logger.info("Engineering new features...")
    df_eng = df.copy()

    # 1. Interaction Term (Example)
    # Ensure columns exist before creating interaction
    if 'DebtToIncomeRatio' in df_eng.columns and 'PercentTradesWBalance' in df_eng.columns:
        # Simple fillna(1) - consider more robust imputation if needed before interaction
        df_eng['DTIxPercTradesWBalance'] = df_eng['DebtToIncomeRatio'].fillna(1) * df_eng['PercentTradesWBalance'].fillna(1)
        logger.info("Added feature: DTIxPercTradesWBalance")
    else:
        logger.warning("Skipping DTIxPercTradesWBalance interaction: required columns not found.")


    # 2. Indicator variables for special values
    logger.info(f"Creating indicator features for special values {config.SPECIAL_VALUES} in columns: {config.INDICATOR_COLS}")
    for col in config.INDICATOR_COLS:
        if col in df_eng.columns:
            for val in config.SPECIAL_VALUES:
                indicator_name = f'{col}_is_{val}'
                df_eng[indicator_name] = (df_eng[col] == val).astype(int)
                # logger.debug(f"Added indicator feature: {indicator_name}") # Use debug for verbosity
        else:
             logger.warning(f"Column '{col}' specified for indicators not found in DataFrame.")


    logger.info(f"Feature engineering complete. New shape: {df_eng.shape}")
    return df_eng

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list, int]:
    """
    Performs target mapping, feature/target separation, special value handling,
    and identifies feature types.
    Returns: X, y, numerical_features, categorical_features, updated positive_label
    """
    logger.info('Performing initial data preparation and type identification...')
    df_processed = df.copy()
    target_col = config.TARGET_VARIABLE
    special_vals = config.SPECIAL_VALUES
    positive_label = config.POSITIVE_LABEL # Start with default

    # Map target variable 'Good'/'Bad' to 0/1
    if target_col not in df_processed.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if df_processed[target_col].dtype == 'object':
        logger.info(f'Mapping target variable {target_col}: Bad=1, Good=0')
        mapping = {'Bad': 1, 'Good': 0}
        df_processed[target_col] = df_processed[target_col].map(mapping)
        positive_label = mapping.get('Bad', 1) # Update positive label based on mapping
        if df_processed[target_col].isnull().any():
             logger.warning("NaNs produced during target mapping. Check unique values.")

    # Separate features and target
    logger.info(f'Separating features and target variable ({target_col})')
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]

    # Replace special values with NaN *in feature columns only*
    logger.info(f'Replacing special values {special_vals} with NaN in numerical feature columns (excluding indicators)')
    num_cols_to_process = X.select_dtypes(include=np.number).columns
    indicator_prefixes = tuple(f'{c}_is_' for c in config.INDICATOR_COLS) # Create prefixes once

    for col in num_cols_to_process:
         # Avoid replacing in indicator columns we just created
         if not col.startswith(indicator_prefixes):
              X[col] = X[col].replace(special_vals, np.nan)

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    logger.info(f'Identified {len(numerical_features)} numerical features.')
    logger.info(f'Identified {len(categorical_features)} categorical features.')
    if categorical_features:
        logger.warning(f"Categorical features found: {categorical_features}. Ensure preprocessing handles them.")

    return X, y, numerical_features, categorical_features, positive_label


def create_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Creates a preprocessing ColumnTransformer for numerical and categorical features.
    """
    logger.info('Creating preprocessing pipeline with ColumnTransformer...')

    transformers = []

    # Define pipeline for numerical features if any exist
    if numerical_features:
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numerical_transformer, numerical_features))
        logger.info("Added numerical transformer (Imputer + Scaler).")
    else:
        logger.info("No numerical features found to preprocess.")


    # Define pipeline for categorical features if any exist
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False often easier downstream
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
        logger.info("Added categorical transformer (Imputer + OneHotEncoder).")
    else:
        logger.info("No categorical features found to encode.")

    # Create ColumnTransformer
    if not transformers:
        logger.warning("No features to preprocess. Returning empty ColumnTransformer.")
        # Return an empty transformer or handle as appropriate for the workflow
        return ColumnTransformer(transformers=[], remainder='passthrough')


    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' # Drop columns not specified in transformers
        # Or use remainder='passthrough' if you want to keep other columns unchanged
    )

    return preprocessor

