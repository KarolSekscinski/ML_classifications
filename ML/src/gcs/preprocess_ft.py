# preprocess_ft.py
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from io import StringIO, BytesIO
import joblib # Using joblib for potentially better handling of sklearn objects
import json
import os

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Columns where -1 should be treated as NaN (Same as before)
COLS_WITH_MINUS_ONE_AS_NAN = [
    'prev_address_months_count', 'current_address_months_count',
    'bank_months_count', 'session_length_in_minutes'
]

# Feature definitions (Ensure these match your Base.csv columns)
TARGET_COLUMN = 'fraud_bool'
# Assuming the same feature lists as before
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

# --- Helper Functions --- (Reusing from previous scripts)

def load_data(data_uri, gcs_bucket):
    """Loads data from local path or GCS URI."""
    logging.info(f"Attempting to load data from: {data_uri}")
    try:
        if data_uri.startswith("gs://"):
            if not gcs_bucket: raise ValueError("GCS Bucket name must be provided for GCS URI")
            blob_name = data_uri.replace(f"gs://{gcs_bucket}/", "")
            logging.info(f"Downloading from GCS: bucket='{gcs_bucket}', blob='{blob_name}'")
            data_bytes = gcs_utils.download_blob_to_memory(gcs_bucket, blob_name)
            if data_bytes is None: raise FileNotFoundError(f"Could not download GCS file: {data_uri}")
            data_source = BytesIO(data_bytes)
            logging.info("GCS file downloaded to memory.")
        else:
            logging.info(f"Loading from local path: {data_uri}")
            data_source = data_uri
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
        # Ensure integer columns are saved as integers if possible (esp. for indices)
        float_cols = df.select_dtypes(include='float').columns
        int_cols = df.select_dtypes(include=['int', 'Int64']).columns # Include nullable Int
        df[float_cols] = df[float_cols].astype(float) # Standard float
        # Attempt to convert potential float indices back to Int64 (nullable integer)
        for col in int_cols:
            try:
                 # Check if conversion back to Int64 is possible without losing info
                 if df[col].notna().all() and (df[col] == df[col].round()).all():
                      df[col] = df[col].astype(pd.Int64Dtype())
            except Exception:
                 pass # Keep as float if conversion fails

        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        gcs_utils.upload_string_to_gcs(gcs_bucket_name, csv_string, gcs_blob_name, content_type='text/csv')
        logging.info(f"Successfully saved to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving DataFrame to GCS ({gcs_blob_name}): {e}")
        raise

def save_object_to_gcs(obj, gcs_bucket_name, gcs_blob_name, use_joblib=True):
    """Saves a Python object (like a preprocessor or metadata) to GCS using joblib or JSON."""
    logging.info(f"Saving object to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    try:
        if gcs_blob_name.endswith(".joblib") and use_joblib:
            with BytesIO() as buf:
                joblib.dump(obj, buf)
                buf.seek(0)
                obj_bytes = buf.read()
            gcs_utils.upload_bytes_to_gcs(gcs_bucket_name, obj_bytes, gcs_blob_name, content_type='application/octet-stream')
        elif gcs_blob_name.endswith(".json"):
             def default_serializer(o):
                 if isinstance(o, (np.integer, np.int64)): return int(o)
                 if isinstance(o, (np.floating, np.float64)): return float(o)
                 if isinstance(o, np.ndarray): return o.tolist()
                 if isinstance(o, (np.bool_, bool)): return bool(o)
                 if pd.isna(o): return None # Handle pandas NaNs
                 raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
             obj_string = json.dumps(obj, indent=4, default=default_serializer)
             gcs_utils.upload_string_to_gcs(gcs_bucket_name, obj_string, gcs_blob_name, content_type='application/json')
        else:
             raise ValueError("Unsupported file type for saving object. Use .joblib or .json")
        logging.info(f"Successfully saved object to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving object to GCS ({gcs_blob_name}): {e}")
        raise


# --- Main Preprocessing Function for FT-Transformer ---

def main(args):
    """Main preprocessing pipeline function for FT-Transformer."""
    logging.info("--- Starting Data Loading and Preprocessing for FT-Transformer ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')

    # --- 1. Load Data ---
    df = load_data(args.input_csv_uri, GCS_BUCKET)
    if df is None: return

    # --- 2. Initial Data Cleaning ---
    logging.info("Performing initial data cleaning (handling -1 as NaN)...")
    for col in COLS_WITH_MINUS_ONE_AS_NAN:
        if col in df.columns:
            count_before = (df[col] == -1).sum()
            if count_before > 0:
                 logging.info(f"Replacing {count_before} instances of -1 with NaN in column '{col}'.")
                 df[col] = df[col].replace(-1, np.nan)
        else:
            logging.warning(f"Column '{col}' specified in COLS_WITH_MINUS_ONE_AS_NAN not found.")

    # --- 3. Define Feature Types & Check Target ---
    logging.info("Defining feature types and validating columns...")
    if TARGET_COLUMN not in df.columns:
        logging.error(f"ERROR: Target column '{TARGET_COLUMN}' not found.")
        return

    numerical_features = [col for col in NUMERICAL_FEATURES_ORIG if col in df.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES_ORIG if col in df.columns]
    logging.info(f"Using {len(numerical_features)} numerical features: {numerical_features}")
    logging.info(f"Using {len(categorical_features)} categorical features: {categorical_features}")

    missing_num = [col for col in NUMERICAL_FEATURES_ORIG if col not in df.columns]
    missing_cat = [col for col in CATEGORICAL_FEATURES_ORIG if col not in df.columns]
    if missing_num: logging.warning(f"Expected numerical features not found: {missing_num}")
    if missing_cat: logging.warning(f"Expected categorical features not found: {missing_cat}")

    all_features = numerical_features + categorical_features
    X = df[all_features]
    y = df[TARGET_COLUMN]
    logging.info(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
    logging.info(f"Target distribution before split:\n{y.value_counts(normalize=True)}")

    # --- 4. Split Data (Raw) ---
    logging.info("Splitting raw data into training and testing sets (80/20 split)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"X_train_raw shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test_raw shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")

    # --- 5. Preprocessing Pipelines Setup ---
    logging.info("Setting up preprocessing pipelines (Num: Impute->Scale, Cat: Impute->OrdinalEncode)...")
    # Numerical Pipeline
    numerical_pipeline = Pipeline([
        ('imputer_median', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Categorical Pipeline
    categorical_pipeline = Pipeline([
        ('imputer_mode', SimpleImputer(strategy='most_frequent')),
        # OrdinalEncoder assigns an integer to each category. handle_unknown='use_encoded_value', unknown_value=-1 handles test set categories not seen in train.
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)) # Ensure integer output
    ])

    # --- 6. Fit Pipelines on Training Data ---
    logging.info("Fitting numerical pipeline on X_train_raw numerical features...")
    numerical_pipeline.fit(X_train_raw[numerical_features])
    logging.info("Numerical pipeline fitted.")

    logging.info("Fitting categorical pipeline on X_train_raw categorical features...")
    categorical_pipeline.fit(X_train_raw[categorical_features])
    logging.info("Categorical pipeline fitted.")

    # Extract cardinalities and mappings from the fitted OrdinalEncoder
    ordinal_encoder_fitted = categorical_pipeline.named_steps['ordinal']
    cat_cardinalities = [len(cats) for cats in ordinal_encoder_fitted.categories_]
    # Create a mapping for metadata: {feature_name: {category_value: index, ...}, ...}
    category_mappings = {}
    for i, feature in enumerate(categorical_features):
         mapping = {cat: index for index, cat in enumerate(ordinal_encoder_fitted.categories_[i])}
         category_mappings[feature] = mapping
         # Add mapping for unknown value if used
         if ordinal_encoder_fitted.unknown_value != -1:
              # Note: The actual index used might vary based on sklearn version behavior.
              # Typically it appends, so index would be len(categories).
              # Let's assume -1 maps to index = len(categories) if used, or handle based on actual output if needed.
              # For simplicity, we won't explicitly map the unknown_value here but rely on the encoder's output.
              pass

    logging.info(f"Determined cardinalities: {cat_cardinalities}")
    # logging.debug(f"Category mappings: {category_mappings}") # Can be very verbose

    # --- 7. Transform Train and Test Data ---
    logging.info("Transforming numerical features for train and test sets...")
    X_train_num_processed = numerical_pipeline.transform(X_train_raw[numerical_features])
    X_test_num_processed = numerical_pipeline.transform(X_test_raw[numerical_features])
    logging.info(f"X_train_num_processed shape: {X_train_num_processed.shape}")
    logging.info(f"X_test_num_processed shape: {X_test_num_processed.shape}")

    logging.info("Transforming categorical features for train and test sets...")
    X_train_cat_indices = categorical_pipeline.transform(X_train_raw[categorical_features])
    X_test_cat_indices = categorical_pipeline.transform(X_test_raw[categorical_features])
     # Check for unknown values encoded as -1
    unknown_train = (X_train_cat_indices == -1).sum()
    unknown_test = (X_test_cat_indices == -1).sum()
    if unknown_train > 0: logging.warning(f"{unknown_train} unknown category values encoded as -1 in TRAINING categorical data (should not happen if fitted correctly).")
    if unknown_test > 0: logging.warning(f"{unknown_test} unknown category values encoded as -1 in TEST categorical data.")

    logging.info(f"X_train_cat_indices shape: {X_train_cat_indices.shape}")
    logging.info(f"X_test_cat_indices shape: {X_test_cat_indices.shape}")


    # --- 8. Combine Processed Train Data for SMOTE ---
    # Combine scaled numerical and ordinal encoded categorical features for SMOTE input
    # Ensure indices align if conversion to numpy dropped them
    logging.info("Combining processed training features for SMOTE...")
    X_train_processed_combined = np.concatenate([X_train_num_processed, X_train_cat_indices], axis=1)
    logging.info(f"X_train_processed_combined shape: {X_train_processed_combined.shape}")

    # --- 9. Apply SMOTE to Combined Processed Training Data ---
    logging.info("Applying SMOTE to the processed & combined training data...")
    minority_class_count = y_train.value_counts().min()
    k_neighbors_smote = 1 # Default minimum

    X_train_resampled = X_train_processed_combined
    y_train_resampled = y_train.copy() # Start with original y_train

    if minority_class_count <= 1:
        logging.warning(f"Minority class in y_train has only {minority_class_count} sample(s). SMOTE cannot be applied.")
    elif minority_class_count <= SMOTE_K_NEIGHBORS_DEFAULT:
        k_neighbors_smote = minority_class_count - 1
        logging.warning(f"Minority class count ({minority_class_count}) is less than default k_neighbors ({SMOTE_K_NEIGHBORS_DEFAULT}). Setting k_neighbors={k_neighbors_smote}.")
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed_combined, y_train)
        logging.info("SMOTE applied successfully.")
    else:
        k_neighbors_smote = SMOTE_K_NEIGHBORS_DEFAULT
        logging.info(f"Applying SMOTE with k_neighbors={k_neighbors_smote}.")
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed_combined, y_train)
        logging.info("SMOTE applied successfully.")

    logging.info(f"Shape of X_train_resampled (combined): {X_train_resampled.shape}, y_train_resampled: {y_train_resampled.shape}")
    logging.info(f"Fraud distribution in y_train_resampled:\n{pd.Series(y_train_resampled).value_counts(normalize=True)}")


    # --- 10. Separate Resampled Data into Numerical and Categorical Indices ---
    logging.info("Separating SMOTE output back into numerical and categorical index features...")
    num_feature_count = len(numerical_features)
    X_train_num_resampled = X_train_resampled[:, :num_feature_count]
    # IMPORTANT: SMOTE generates float values. Round categorical indices back to integers.
    X_train_cat_resampled_float = X_train_resampled[:, num_feature_count:]
    X_train_cat_indices_resampled = np.round(X_train_cat_resampled_float).astype(int)

    # Clip values to be within valid ordinal encoded range [0, max_index] for each column
    # Or handle potential -1s if unknown was used and somehow generated by SMOTE rounding
    for i in range(X_train_cat_indices_resampled.shape[1]):
         max_valid_index = cat_cardinalities[i] - 1
         # Clip values outside the learned categories during fit.
         # Also handle potential -1s if unknown_value was used and generated by SMOTE
         X_train_cat_indices_resampled[:, i] = np.clip(X_train_cat_indices_resampled[:, i], 0, max_valid_index)
         # If -1 was used for unknowns, check if any are present after rounding
         if (X_train_cat_indices_resampled[:, i] == -1).any():
              logging.warning(f"Generated index -1 found in resampled categorical column {i}. Clipping to 0.")
              X_train_cat_indices_resampled[X_train_cat_indices_resampled[:, i] == -1, i] = 0 # Or map to a default category index


    logging.info(f"Final X_train_num_resampled shape: {X_train_num_resampled.shape}")
    logging.info(f"Final X_train_cat_indices_resampled shape: {X_train_cat_indices_resampled.shape}")


    # --- 11. Save Processed Data and Artifacts to GCS ---
    logging.info("Saving processed data splits, pipelines, and metadata to GCS...")

    # Convert final numpy arrays to DataFrames for saving with headers
    df_x_train_num = pd.DataFrame(X_train_num_resampled, columns=numerical_features)
    df_x_train_cat = pd.DataFrame(X_train_cat_indices_resampled, columns=categorical_features)
    df_y_train = pd.Series(y_train_resampled, name=TARGET_COLUMN).to_frame()

    df_x_test_num = pd.DataFrame(X_test_num_processed, columns=numerical_features)
    df_x_test_cat = pd.DataFrame(X_test_cat_indices, columns=categorical_features)
    df_y_test = pd.Series(y_test, name=TARGET_COLUMN).to_frame()

    # Define GCS paths
    gcs_path_x_train_num = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_train_num_scaled.csv"
    gcs_path_x_train_cat = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_train_cat_indices.csv"
    gcs_path_y_train = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/y_train_resampled.csv"
    gcs_path_x_test_num = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_test_num_scaled.csv"
    gcs_path_x_test_cat = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_test_cat_indices.csv"
    gcs_path_y_test = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/y_test.csv"
    gcs_path_metadata = f"{GCS_OUTPUT_PREFIX}/metadata/preprocessing_ft_metadata.json"
    gcs_path_num_pipeline = f"{GCS_OUTPUT_PREFIX}/preprocessor_ft/numerical_pipeline.joblib"
    gcs_path_cat_pipeline = f"{GCS_OUTPUT_PREFIX}/preprocessor_ft/categorical_pipeline.joblib"


    # Save dataframes
    save_dataframe_to_gcs(df_x_train_num, GCS_BUCKET, gcs_path_x_train_num)
    save_dataframe_to_gcs(df_x_train_cat, GCS_BUCKET, gcs_path_x_train_cat)
    save_dataframe_to_gcs(df_y_train, GCS_BUCKET, gcs_path_y_train)
    save_dataframe_to_gcs(df_x_test_num, GCS_BUCKET, gcs_path_x_test_num)
    save_dataframe_to_gcs(df_x_test_cat, GCS_BUCKET, gcs_path_x_test_cat)
    save_dataframe_to_gcs(df_y_test, GCS_BUCKET, gcs_path_y_test)

    # Save fitted pipelines
    save_object_to_gcs(numerical_pipeline, GCS_BUCKET, gcs_path_num_pipeline, use_joblib=True)
    save_object_to_gcs(categorical_pipeline, GCS_BUCKET, gcs_path_cat_pipeline, use_joblib=True)


    # Prepare and save metadata
    metadata = {
        "original_data_uri": args.input_csv_uri,
        "gcs_output_prefix": GCS_OUTPUT_PREFIX,
        "data_format": "FT-Transformer Ready (Num Scaled, Cat Indices)",
        "target_column": TARGET_COLUMN,
        "numerical_features_used": numerical_features,
        "categorical_features_used": categorical_features,
        "cat_feature_cardinalities": cat_cardinalities,
        "category_mappings": category_mappings, # Include the mapping details
        "smote_applied_to_train": minority_class_count > 1,
        "smote_k_neighbors_used": k_neighbors_smote if minority_class_count > 1 else None,
        "X_train_num_shape": df_x_train_num.shape,
        "X_train_cat_shape": df_x_train_cat.shape,
        "y_train_resampled_shape": df_y_train.shape,
        "X_test_num_shape": df_x_test_num.shape,
        "X_test_cat_shape": df_x_test_cat.shape,
        "y_test_shape": df_y_test.shape,
        "gcs_paths": {
            "X_train_num_scaled": f"gs://{GCS_BUCKET}/{gcs_path_x_train_num}",
            "X_train_cat_indices": f"gs://{GCS_BUCKET}/{gcs_path_x_train_cat}",
            "y_train_resampled": f"gs://{GCS_BUCKET}/{gcs_path_y_train}",
            "X_test_num_scaled": f"gs://{GCS_BUCKET}/{gcs_path_x_test_num}",
            "X_test_cat_indices": f"gs://{GCS_BUCKET}/{gcs_path_x_test_cat}",
            "y_test": f"gs://{GCS_BUCKET}/{gcs_path_y_test}",
            "numerical_pipeline": f"gs://{GCS_BUCKET}/{gcs_path_num_pipeline}",
            "categorical_pipeline": f"gs://{GCS_BUCKET}/{gcs_path_cat_pipeline}",
            "metadata": f"gs://{GCS_BUCKET}/{gcs_path_metadata}"
        }
    }
    save_object_to_gcs(metadata, GCS_BUCKET, gcs_path_metadata, use_joblib=False) # Save metadata as JSON

    logging.info("--- Data Loading and Preprocessing for FT-Transformer Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, preprocess data for FT-Transformer, and save to GCS.")
    parser.add_argument("--input-csv-uri", type=str, required=True,
                        help="Path or GCS URI (gs://bucket/path/to/Base.csv) to the input CSV file.")
    parser.add_argument("--gcs-bucket", type=str, required=True,
                        help="GCS bucket name for input (if GCS URI used) and output.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True,
                        help="GCS prefix (folder path) within the bucket to save processed data and artifacts (e.g., 'fraud_detection/ft_processed').")

    args = parser.parse_args()
    main(args)