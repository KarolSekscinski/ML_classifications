# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from io import StringIO, BytesIO
import pickle
import argparse
import logging
import json

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
from ML_classifications.ML.src.gcs import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Columns where -1 should be treated as NaN
COLS_WITH_MINUS_ONE_AS_NAN = [
    'prev_address_months_count', 'current_address_months_count',
    'bank_months_count', 'session_length_in_minutes'
]

# Feature definitions (ensure these match your Base.csv columns)
TARGET_COLUMN = 'fraud_bool'
NUMERICAL_FEATURES_ORIG = [
    'income', 'name_email_similarity', 'prev_address_months_count',
    'current_address_months_count', 'days_since_request', 'intended_balcon_amount',
    'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w',
    'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score',
    'bank_months_count', 'proposed_credit_limit', 'session_length_in_minutes',
    'device_distinct_emails_8w'
]
CATEGORICAL_FEATURES_ORIG = [
    'customer_age', 'payment_type', 'employment_status', 'housing_status',
    'source', 'device_os', 'month'
]

# SMOTE configuration
SMOTE_K_NEIGHBORS_DEFAULT = 5

def load_data(data_uri, gcs_bucket):
    """Loads data from local path or GCS URI."""
    logging.info(f"Attempting to load data from: {data_uri}")
    try:
        if data_uri.startswith("gs://"):
            if not gcs_bucket:
                raise ValueError("GCS Bucket name must be provided for GCS URI")
            blob_name = data_uri.replace(f"gs://{gcs_bucket}/", "")
            logging.info(f"Downloading from GCS: bucket='{gcs_bucket}', blob='{blob_name}'")
            data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, blob_name)
            if data_bytes is None:
                raise FileNotFoundError(f"Could not download GCS file: {data_uri}")
            data_source = BytesIO(data_bytes)
            logging.info("GCS file downloaded to memory.")
        else:
            logging.info(f"Loading from local path: {data_uri}")
            data_source = data_uri # Assume it's a local file path string

        df = pd.read_csv(data_source)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"ERROR: Input data file not found at: {data_uri}.")
        raise
    except Exception as e:
        logging.error(f"Error loading data from {data_uri}: {e}")
        raise

def save_dataframe_to_gcs(df, gcs_bucket_name, gcs_blob_name):
    """Saves a pandas DataFrame to GCS as CSV."""
    logging.info(f"Saving DataFrame to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        gcs_utils.upload_string_to_gcs(gcs_bucket_name, csv_string, gcs_blob_name, content_type='text/csv')
        logging.info(f"Successfully saved to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving DataFrame to GCS ({gcs_blob_name}): {e}")
        raise

def save_object_to_gcs(obj, gcs_bucket_name, gcs_blob_name, content_type='application/octet-stream'):
    """Saves a Python object (like a preprocessor or metadata) to GCS using pickle or JSON."""
    logging.info(f"Saving object to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    try:
        if gcs_blob_name.endswith(".pkl"):
            obj_bytes = pickle.dumps(obj)
            gcs_utils.upload_bytes_to_gcs(gcs_bucket_name, obj_bytes, gcs_blob_name, content_type=content_type)
        elif gcs_blob_name.endswith(".json"):
             # Convert numpy types if present for JSON serialization
            def default_serializer(o):
                if isinstance(o, np.integer): return int(o)
                if isinstance(o, np.floating): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
            obj_string = json.dumps(obj, indent=4, default=default_serializer)
            gcs_utils.upload_string_to_gcs(gcs_bucket_name, obj_string, gcs_blob_name, content_type='application/json')
        else:
             raise ValueError("Unsupported file type for saving object. Use .pkl or .json")
        logging.info(f"Successfully saved object to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving object to GCS ({gcs_blob_name}): {e}")
        raise

def main(args):
    """Main preprocessing pipeline function."""
    logging.info("--- Starting Data Loading and Preprocessing ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/') # Remove trailing slash if exists

    # --- 1. Load Data ---
    df = load_data(args.input_csv_uri, GCS_BUCKET)
    if df is None:
        return # Error logged in load_data

    # --- 2. Initial Data Cleaning ---
    logging.info("Performing initial data cleaning (handling -1 as NaN)...")
    for col in COLS_WITH_MINUS_ONE_AS_NAN:
        if col in df.columns:
            count_before = (df[col] == -1).sum()
            if count_before > 0:
                 logging.info(f"Replacing {count_before} instances of -1 with NaN in column '{col}'.")
                 df[col] = df[col].replace(-1, np.nan)
            else:
                 logging.info(f"No -1 values found in column '{col}'.")
        else:
            logging.warning(f"Column '{col}' specified in COLS_WITH_MINUS_ONE_AS_NAN not found in DataFrame.")

    # --- 3. Define Feature Types & Check Target ---
    logging.info("Defining feature types and validating columns...")
    if TARGET_COLUMN not in df.columns:
        logging.error(f"ERROR: Target column '{TARGET_COLUMN}' not found in the dataframe.")
        return

    # Filter feature lists based on actual columns present in the loaded dataframe
    numerical_features = [col for col in NUMERICAL_FEATURES_ORIG if col in df.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES_ORIG if col in df.columns]
    logging.info(f"Using {len(numerical_features)} numerical features: {numerical_features}")
    logging.info(f"Using {len(categorical_features)} categorical features: {categorical_features}")

    missing_numerical = [col for col in NUMERICAL_FEATURES_ORIG if col not in df.columns]
    missing_categorical = [col for col in CATEGORICAL_FEATURES_ORIG if col not in df.columns]
    if missing_numerical:
        logging.warning(f"Expected numerical features not found: {missing_numerical}")
    if missing_categorical:
        logging.warning(f"Expected categorical features not found: {missing_categorical}")

    all_features = numerical_features + categorical_features
    X = df[all_features]
    y = df[TARGET_COLUMN]
    logging.info(f"Shape of X (features): {X.shape}, Shape of y (target): {y.shape}")
    logging.info(f"Target distribution before split:\n{y.value_counts(normalize=True)}")


    # --- 4. Split Data (Raw) ---
    logging.info("Splitting raw data into training and testing sets (80/20 split)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"X_train_raw shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test_raw shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")
    logging.info(f"Target distribution in y_train:\n{y_train.value_counts(normalize=True)}")
    logging.info(f"Target distribution in y_test:\n{y_test.value_counts(normalize=True)}")

    # --- 5. Calculate Cardinalities (Before OHE, After Imputation on Train) ---
    cat_cardinalities = []
    if categorical_features:
        logging.info("Calculating cardinalities for categorical features (post-imputation)...")
        # Temporarily impute on training data to calculate accurate cardinalities
        temp_cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train_cat_imputed_for_card = X_train_raw[categorical_features].copy()
        X_train_cat_imputed_for_card = pd.DataFrame(
            temp_cat_imputer.fit_transform(X_train_cat_imputed_for_card),
            columns=categorical_features,
            index=X_train_raw.index # Preserve index if needed later, though not strictly necessary here
        )
        for col in categorical_features:
            nunique = X_train_cat_imputed_for_card[col].astype('category').nunique()
            cat_cardinalities.append(nunique)
            logging.info(f"  - Column '{col}': {nunique} unique values")
    logging.info(f"Determined cardinalities: {cat_cardinalities}")

    # --- 6. Preprocessing Pipelines Setup ---
    logging.info("Setting up preprocessing pipelines...")
    numerical_pipeline = Pipeline([
        ('imputer_median', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer_mode', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features), # Use filtered list
            ('cat', categorical_pipeline, categorical_features) # Use filtered list
        ],
        remainder='drop' # Drop any columns not explicitly handled
    )
    logging.info("Preprocessor configured.")

    # --- 7. Fit Preprocessor and Transform Data ---
    logging.info("Fitting preprocessor on X_train_raw and transforming train/test data...")
    # Fit *only* on the training data
    preprocessor.fit(X_train_raw)
    logging.info("Preprocessor fitted on X_train_raw.")

    # Transform both training and testing data
    X_train_processed = preprocessor.transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    logging.info(f"X_train transformed. Shape: {X_train_processed.shape}")
    logging.info(f"X_test transformed. Shape: {X_test_processed.shape}")

    # Save the fitted preprocessor
    preprocessor_blob_name = f"{GCS_OUTPUT_PREFIX}/preprocessor/preprocessor.pkl"
    save_object_to_gcs(preprocessor, GCS_BUCKET, preprocessor_blob_name)

    # --- 8. Reconstruct Feature Names (Post-Transformation) ---
    processed_feature_names = []
    try:
        # Get numerical feature names (they remain the same)
        processed_feature_names.extend(numerical_features)

        # Get OneHotEncoded feature names
        ohe_feature_names = []
        if categorical_features: # Only if there were categorical features to encode
             cat_pipeline_fitted = preprocessor.named_transformers_['cat']
             ohe_step = cat_pipeline_fitted.named_steps['onehot']
             ohe_feature_names = list(ohe_step.get_feature_names_out(categorical_features))
             processed_feature_names.extend(ohe_feature_names)

        logging.info(f"Processed feature names reconstructed. Total features: {len(processed_feature_names)}")
        # Verify length
        if len(processed_feature_names) != X_train_processed.shape[1]:
             logging.warning(f"Mismatch in reconstructed feature names ({len(processed_feature_names)}) and processed data columns ({X_train_processed.shape[1]}). Check ColumnTransformer remainder setting.")
    except Exception as e:
        logging.error(f"Could not reliably reconstruct feature names after OHE: {e}. Using generic names for saving.")
        processed_feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]


    # --- 9. Resampling (SMOTE) on Processed Training Data ---
    logging.info("Applying SMOTE to the processed training data...")
    minority_class_count = y_train.value_counts().min()
    k_neighbors_smote = 1 # Default minimum

    if minority_class_count <= 1:
         logging.warning(f"Minority class in y_train has only {minority_class_count} sample(s). SMOTE cannot be applied.")
         X_train_resampled = X_train_processed
         y_train_resampled = y_train.copy() # Ensure it's a copy
    elif minority_class_count <= SMOTE_K_NEIGHBORS_DEFAULT:
         k_neighbors_smote = minority_class_count - 1
         logging.warning(f"Minority class count ({minority_class_count}) is less than default k_neighbors ({SMOTE_K_NEIGHBORS_DEFAULT}). Setting k_neighbors={k_neighbors_smote}.")
         smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
         X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
         logging.info("SMOTE applied successfully.")
    else:
         k_neighbors_smote = SMOTE_K_NEIGHBORS_DEFAULT
         logging.info(f"Applying SMOTE with k_neighbors={k_neighbors_smote}.")
         smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
         X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
         logging.info("SMOTE applied successfully.")

    logging.info(f"Shape of X_train_resampled: {X_train_resampled.shape}, y_train_resampled: {y_train_resampled.shape}")
    logging.info(f"Fraud distribution in y_train_resampled:\n{pd.Series(y_train_resampled).value_counts(normalize=True)}")


    # --- 10. Save Processed Data and Metadata to GCS ---
    logging.info("Saving processed data splits and metadata to GCS...")

    # Convert processed numpy arrays back to DataFrames for saving with headers
    df_x_train_resampled = pd.DataFrame(X_train_resampled, columns=processed_feature_names)
    df_y_train_resampled = pd.Series(y_train_resampled, name=TARGET_COLUMN).to_frame() # Save as DataFrame with header
    df_x_test_processed = pd.DataFrame(X_test_processed, columns=processed_feature_names)
    df_y_test = pd.Series(y_test, name=TARGET_COLUMN).to_frame() # Save as DataFrame with header

    # Define GCS paths
    gcs_path_x_train = f"{GCS_OUTPUT_PREFIX}/processed_data/X_train_resampled.csv"
    gcs_path_y_train = f"{GCS_OUTPUT_PREFIX}/processed_data/y_train_resampled.csv"
    gcs_path_x_test = f"{GCS_OUTPUT_PREFIX}/processed_data/X_test_processed.csv"
    gcs_path_y_test = f"{GCS_OUTPUT_PREFIX}/processed_data/y_test.csv"
    gcs_path_metadata = f"{GCS_OUTPUT_PREFIX}/metadata/preprocessing_metadata.json"

    # Save dataframes
    save_dataframe_to_gcs(df_x_train_resampled, GCS_BUCKET, gcs_path_x_train)
    save_dataframe_to_gcs(df_y_train_resampled, GCS_BUCKET, gcs_path_y_train)
    save_dataframe_to_gcs(df_x_test_processed, GCS_BUCKET, gcs_path_x_test)
    save_dataframe_to_gcs(df_y_test, GCS_BUCKET, gcs_path_y_test)

    # Prepare and save metadata
    metadata = {
        "original_data_uri": args.input_csv_uri,
        "gcs_output_prefix": GCS_OUTPUT_PREFIX,
        "target_column": TARGET_COLUMN,
        "numerical_features_used": numerical_features,
        "categorical_features_used": categorical_features,
        "cat_feature_cardinalities": cat_cardinalities, # List of cardinalities in order of categorical_features
        "processed_feature_names": processed_feature_names,
        "smote_applied": "True" if minority_class_count > 1 else "False",
        "smote_k_neighbors_used": k_neighbors_smote if minority_class_count > 1 else None,
        "X_train_resampled_shape": df_x_train_resampled.shape,
        "y_train_resampled_shape": df_y_train_resampled.shape,
        "X_test_processed_shape": df_x_test_processed.shape,
        "y_test_shape": df_y_test.shape,
        "gcs_paths": {
            "X_train_resampled": f"gs://{GCS_BUCKET}/{gcs_path_x_train}",
            "y_train_resampled": f"gs://{GCS_BUCKET}/{gcs_path_y_train}",
            "X_test_processed": f"gs://{GCS_BUCKET}/{gcs_path_x_test}",
            "y_test": f"gs://{GCS_BUCKET}/{gcs_path_y_test}",
            "preprocessor": f"gs://{GCS_BUCKET}/{preprocessor_blob_name}",
            "metadata": f"gs://{GCS_BUCKET}/{gcs_path_metadata}"
        }
    }
    save_object_to_gcs(metadata, GCS_BUCKET, gcs_path_metadata) # Save as JSON

    logging.info("--- Data Loading and Preprocessing Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, preprocess, and save data to GCS.")
    parser.add_argument("--input-csv-uri", type=str, required=True,
                        help="Path or GCS URI (gs://bucket/path/to/Base.csv) to the input CSV file.")
    parser.add_argument("--gcs-bucket", type=str, required=True,
                        help="GCS bucket name for input (if GCS URI used) and output.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True,
                        help="GCS prefix (folder path) within the bucket to save processed data and artifacts (e.g., 'fraud_detection/processed').")

    args = parser.parse_args()
    main(args)