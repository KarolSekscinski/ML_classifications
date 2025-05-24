# preprocess_finbench_cf2_ohe.py
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from io import StringIO, BytesIO
import joblib
import json
import ast  # For safely evaluating string literals

# Assuming gcs_utils.py is in the same directory or accessible via PYTHONPATH
from ML_classifications.ML.src.gcs import gcs_utils

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Target column name from the dataset
TARGET_COLUMN_CF2 = 'y'

# SMOTE configuration
SMOTE_K_NEIGHBORS_DEFAULT = 5


# --- Helper Functions ---
def parse_string_to_list(s):
    """Safely parses a string representation of a list/array."""
    try:
        # Handle numpy array string format by replacing spaces with commas,
        # removing multiple spaces, and ensuring it's a valid list format.
        s_cleaned = s.strip()
        if s_cleaned.startswith('[') and s_cleaned.endswith(']'):
            # For "[ 1.  2.  3. ]" style arrays
            s_content = s_cleaned[1:-1].strip()
            s_content = ' '.join(s_content.split())  # Normalize spaces
            s_content = s_content.replace(' ', ', ')  # Replace space with comma-space
            s_eval_ready = f"[{s_content}]"
            return ast.literal_eval(s_eval_ready)
        else:  # Assume it's a simple list or non-array string
            return ast.literal_eval(s)
    except Exception as e:
        logging.warning(f"Could not parse string '{s}' as list/array: {e}. Returning as is or empty list.")
        # Attempt to split by space if it's a simple space-separated list of numbers without brackets
        try:
            return [float(x) for x in s.split()]
        except ValueError:
            return []  # Fallback for unparseable strings not matching known formats


def parse_cat_str_column(cat_str_series_element):
    """
    Parses elements from the 'cat_str' column.
    Example input: "[array(['2019-10-06', ..., '2020-07-31'], dtype=object) \n array(['2020-04-23', ..., '2020-11-08'], dtype=object)]"
    We want to extract the first element of each array for the respective categorical feature.
    """
    try:
        # This is a very specific parser for the given cat_str format.
        # It assumes the string contains representations of numpy arrays.
        # We're interested in the *values* for the current row, not the full dictionary.
        # For FinBench cf2, cat_str seems to contain the *actual values* for that row's categorical features.
        # For example, if cat_idx is [0,1], cat_str would be "[val_for_cat_feat_0 val_for_cat_feat_1]" (simplified)
        # or "[array(['val_for_cat_feat_0']) array(['val_for_cat_feat_1'])]"

        # Let's assume cat_str for a row looks like: "[ 'val1' 'val2' ... ]" or similar
        # For the sample: "[array(['2019-10-06', ...], dtype=object) array(['2020-04-23', ...], dtype=object)]"
        # This implies cat_str itself needs to be parsed carefully to extract *the specific values for that row*.
        # The sample `cat_str` contains two arrays of many dates. This doesn't seem to be the *instance value*
        # but rather the *possible values/dictionary* for each categorical feature.
        # This interpretation might be wrong. Let's assume for now `cat_str` directly gives the value
        # or that the features at `cat_idx` in `X_ml_unscale` are directly usable (even if numbers)
        # and should be *treated* as categorical.

        # **Revisiting FinBench structure:**
        # The `X_ml_unscale` likely contains numerical representations *even for categorical features*.
        # The `cat_str` column in the Hugging Face viewer for FinBench *often shows the dictionary mapping*,
        # not the instance value. The instance value is in `X_ml_unscale` at the `cat_idx` positions.
        # These numerical values in `X_ml_unscale` at `cat_idx` are likely already integer-encoded.
        # So, we will treat features at `cat_idx` from `X_ml_unscale` as categorical.
        # The actual `cat_str` column from the CSV will be ignored for now for simplicity in parsing,
        # as `X_ml_unscale` should contain the usable feature values.

        # This function is therefore NOT needed if we take categorical features directly from X_ml_unscale.
        pass
    except Exception as e:
        logging.error(f"Error parsing cat_str element: {e}")
        return []


def reconstruct_dataframe(df_raw, all_feature_names, cat_indices):
    """
    Reconstructs a flat DataFrame from the FinBench cf2 format.
    Numerical features are taken directly.
    Categorical features (identified by cat_indices in X_ml_unscale) are also taken from X_ml_unscale
    and will be explicitly treated as 'category' dtype.
    """
    reconstructed_data = []
    for _, row in df_raw.iterrows():
        try:
            # Parse X_ml_unscale (contains all 19 feature values for the row)
            x_unscale_values = parse_string_to_list(row['X_ml_unscale'])
            if len(x_unscale_values) != len(all_feature_names):
                logging.error(
                    f"Row {row.index}: Mismatch in length of X_ml_unscale ({len(x_unscale_values)}) and col_name ({len(all_feature_names)}). Skipping row.")
                continue

            current_row_data = dict(zip(all_feature_names, x_unscale_values))
            current_row_data[TARGET_COLUMN_CF2] = row[TARGET_COLUMN_CF2]
            reconstructed_data.append(current_row_data)
        except Exception as e:
            logging.error(f"Error processing row: {row.name} - {e}. Skipping row.")
            continue

    flat_df = pd.DataFrame(reconstructed_data)

    # Explicitly set dtype for categorical columns
    for i in cat_indices:
        col_name = all_feature_names[i]
        if col_name in flat_df.columns:
            # Convert to string first, then category, to handle numerical representations of categories
            flat_df[col_name] = flat_df[col_name].astype(str).astype('category')

    return flat_df


def load_and_prepare_data(file_path, common_all_feature_names, common_cat_indices):
    """Loads a single CSV, parses it, and reconstructs the flat DataFrame."""
    raw_df = pd.read_csv(file_path)  # Load the raw CSV with stringified lists
    if raw_df.empty:
        logging.error(f"File {file_path} is empty or could not be read into DataFrame.")
        return None

    # Use common_all_feature_names and common_cat_indices derived from the first file (e.g., train)
    flat_df = reconstruct_dataframe(raw_df, common_all_feature_names, common_cat_indices)
    return flat_df


# (save_dataframe_to_gcs, save_object_to_gcs - keep from previous script)
def save_dataframe_to_gcs(df, gcs_bucket_name, gcs_blob_name):
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


def save_object_to_gcs(obj, gcs_bucket_name, gcs_blob_name, use_joblib=True):
    logging.info(f"Saving object to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    try:
        if gcs_blob_name.endswith(".joblib") and use_joblib:
            with BytesIO() as buf:
                joblib.dump(obj, buf)
                buf.seek(0)
                obj_bytes = buf.read()
            gcs_utils.upload_bytes_to_gcs(gcs_bucket_name, obj_bytes, gcs_blob_name,
                                          content_type='application/octet-stream')
        elif gcs_blob_name.endswith(".json"):
            def default_serializer(o):
                if isinstance(o, (np.integer, np.int64)): return int(o)
                if isinstance(o, (np.floating, np.float64)): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, (np.bool_, bool)): return bool(o)
                if pd.isna(o): return None
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

            obj_string = json.dumps(obj, indent=4, default=default_serializer)
            gcs_utils.upload_string_to_gcs(gcs_bucket_name, obj_string, gcs_blob_name, content_type='application/json')
        else:
            raise ValueError("Unsupported file type for saving object. Use .joblib or .json")
        logging.info(f"Successfully saved object to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving object to GCS ({gcs_blob_name}): {e}")
        raise


def main(args):
    logging.info("--- Starting Data Preprocessing for FinBench cf2 (OHE for SVM, XGBoost, MLP) ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')

    # --- 1. Load Structure from Training Data ---
    logging.info(f"Loading training data from {args.train_file_path} to determine structure...")
    raw_train_df_struct_only = pd.read_csv(args.train_file_path, nrows=1)
    if raw_train_df_struct_only.empty:
        logging.error("Training file is empty or unreadable. Cannot determine data structure.")
        return

    try:
        common_all_feature_names = parse_string_to_list(raw_train_df_struct_only['col_name'].iloc[0])
        common_cat_indices = [int(i) for i in parse_string_to_list(raw_train_df_struct_only['cat_idx'].iloc[0])]
    except Exception as e:
        logging.error(f"Failed to parse structural columns (col_name, cat_idx) from training data: {e}")
        return

    # --- Warn about duplicates in source col_name ---
    if len(common_all_feature_names) != len(set(common_all_feature_names)):
        logging.warning("Duplicate names found in 'col_name' from input CSV!")
        from collections import Counter
        counts = Counter(common_all_feature_names)
        duplicates_in_source = {name: count for name, count in counts.items() if count > 1}
        logging.warning(f"Duplicate source names and their counts: {duplicates_in_source}")
        logging.warning("The reconstructed DataFrame will use the last occurrence of duplicate column names.")

    # Derive initial feature lists (these might contain duplicates if source col_name had them)
    _temp_categorical_feature_names = [common_all_feature_names[i] for i in common_cat_indices]
    _temp_numerical_feature_names = [name for i, name in enumerate(common_all_feature_names) if
                                     i not in common_cat_indices]

    # Create lists of UNIQUE feature names for ColumnTransformer (preserves order)
    categorical_feature_names = list(dict.fromkeys(_temp_categorical_feature_names))
    numerical_feature_names = list(dict.fromkeys(_temp_numerical_feature_names))

    # Check for overlap between the unique numerical and categorical lists
    overlap = set(numerical_feature_names) & set(categorical_feature_names)
    if overlap:
        logging.error(
            f"CRITICAL: Overlap detected between derived unique numerical and categorical feature lists: {overlap}")
        logging.error("This indicates a fundamental issue in how cat_idx is defined relative to col_name.")
        return

    logging.info(f"Target column: {TARGET_COLUMN_CF2}")
    logging.info(
        f"Using UNIQUE Derived Numerical Features ({len(numerical_feature_names)}): {numerical_feature_names if numerical_feature_names else 'None'}")
    logging.info(
        f"Using UNIQUE Derived Categorical Features ({len(categorical_feature_names)}): {categorical_feature_names if categorical_feature_names else 'None'}")

    # --- 2. Load and Reconstruct Full Datasets ---
    # reconstruct_dataframe uses common_all_feature_names (with potential duplicates for dict keys)
    # but resulting flat_df will have unique columns (last duplicate key wins).
    train_df = load_and_prepare_data(args.train_file_path, common_all_feature_names, common_cat_indices)
    val_df = load_and_prepare_data(args.val_file_path, common_all_feature_names, common_cat_indices)
    test_df = load_and_prepare_data(args.test_file_path, common_all_feature_names, common_cat_indices)

    if train_df is None or val_df is None or test_df is None:
        logging.error("One or more data files could not be processed. Exiting.")
        return

    X_train_raw = train_df.drop(columns=[TARGET_COLUMN_CF2])
    y_train = train_df[TARGET_COLUMN_CF2]
    X_val_raw = val_df.drop(columns=[TARGET_COLUMN_CF2])
    y_val = val_df[TARGET_COLUMN_CF2]
    X_test_raw = test_df.drop(columns=[TARGET_COLUMN_CF2])
    y_test = test_df[TARGET_COLUMN_CF2]

    # --- 3. Preprocessing Pipelines Setup ---
    # (Pipelines definition remain the same)
    numerical_pipeline = Pipeline([
        ('imputer_median', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer_mode', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int))
    ])

    transformers_list = []
    # Use the de-duplicated lists for ColumnTransformer
    if numerical_feature_names:  # Ensure list is not empty
        transformers_list.append(('num', numerical_pipeline, numerical_feature_names))
    if categorical_feature_names:  # Ensure list is not empty
        transformers_list.append(('cat', categorical_pipeline, categorical_feature_names))

    if not transformers_list:
        logging.error("No numerical or categorical features defined to process after de-duplication.")
        return

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='drop'
    )

    # --- 4. Fit Preprocessor on Training Data & Transform All Sets ---
    logging.info("Fitting preprocessor on reconstructed X_train_raw and transforming all data splits...")

    # Create the DataFrame for fitting using only the unique feature names that will be processed
    # X_train_raw ALREADY has unique columns due to how reconstruct_dataframe works.
    # The lists numerical_feature_names and categorical_feature_names are now also unique.
    features_for_ct_processing = []
    if numerical_feature_names: features_for_ct_processing.extend(numerical_feature_names)
    if categorical_feature_names: features_for_ct_processing.extend(categorical_feature_names)

    # Ensure features_for_ct_processing is also unique (should be if no overlap previously)
    features_for_ct_processing = list(dict.fromkeys(features_for_ct_processing))

    # Select only the necessary columns from X_train_raw (which has unique columns)
    # These selected columns MUST exist in X_train_raw.
    missing_cols_in_X_train_raw = [col for col in features_for_ct_processing if col not in X_train_raw.columns]
    if missing_cols_in_X_train_raw:
        logging.error(
            f"FATAL: The following features are not in the reconstructed X_train_raw: {missing_cols_in_X_train_raw}")
        logging.error(f"X_train_raw columns: {X_train_raw.columns.tolist()}")
        return

    X_train_raw_for_fit = X_train_raw[features_for_ct_processing]
    preprocessor.fit(X_train_raw_for_fit)
    logging.info("Preprocessor fitted.")

    # Transform datasets
    X_train_processed = preprocessor.transform(X_train_raw_for_fit)
    if features_for_ct_processing:  # Ensure there are columns to select
        X_val_for_transform = X_val_raw[features_for_ct_processing]
        X_val_processed = preprocessor.transform(X_val_for_transform)
        X_test_for_transform = X_test_raw[features_for_ct_processing]
        X_test_processed = preprocessor.transform(X_test_for_transform)
    else:  # Handle cases where no features are selected for processing
        X_val_processed = np.array([[] for _ in range(len(X_val_raw))])
        X_test_processed = np.array([[] for _ in range(len(X_test_raw))])

    logging.info(f"X_train_processed shape: {X_train_processed.shape}")
    logging.info(f"X_val_processed shape: {X_val_processed.shape}")
    logging.info(f"X_test_processed shape: {X_test_processed.shape}")

    # Get feature names after OHE
    processed_feature_names = []
    if numerical_feature_names:  # Use the unique list
        processed_feature_names.extend(numerical_feature_names)
    if categorical_feature_names and 'cat' in preprocessor.named_transformers_:  # Use the unique list
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
                categorical_feature_names)
            processed_feature_names.extend(list(ohe_feature_names))
        except Exception as e:
            logging.warning(f"Could not get OHE feature names: {e}.")
            num_ohe_cols = X_train_processed.shape[1] - len(numerical_feature_names if numerical_feature_names else [])
            if num_ohe_cols > 0: processed_feature_names.extend([f"ohe_feat_{i}" for i in range(num_ohe_cols)])

    if not processed_feature_names and X_train_processed.shape[1] > 0:  # If all else fails for names
        processed_feature_names = [f"proc_feat_{i}" for i in range(X_train_processed.shape[1])]
    elif X_train_processed.shape[1] != len(processed_feature_names):
        logging.warning(
            f"Final feature name length mismatch. Data cols: {X_train_processed.shape[1]}, Names: {len(processed_feature_names)}. Using generic.")
        processed_feature_names = [f"proc_feat_{i}" for i in range(X_train_processed.shape[1])]

    # --- 5. Apply SMOTE to Training Data Only ---
    # ... (SMOTE logic remains the same, ensure X_train_processed has features) ...
    logging.info("Applying SMOTE to the processed training data...")
    minority_class_count = y_train.value_counts().min()
    k_neighbors_smote = 1
    X_train_resampled = X_train_processed
    y_train_resampled = y_train.copy()

    if X_train_processed.shape[0] > 0 and X_train_processed.shape[1] > 0:  # Check if data is not empty
        if minority_class_count <= 1:
            logging.warning(f"Minority class in y_train has {minority_class_count} sample(s). SMOTE not applied.")
        elif minority_class_count <= SMOTE_K_NEIGHBORS_DEFAULT:
            k_neighbors_smote = minority_class_count - 1
            if k_neighbors_smote < 1: k_neighbors_smote = 1  # k_neighbors must be at least 1
            logging.warning(
                f"Minority count ({minority_class_count}) <= default k_neighbors. Using k_neighbors={k_neighbors_smote}.")
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            logging.info("SMOTE applied.")
        else:
            k_neighbors_smote = SMOTE_K_NEIGHBORS_DEFAULT
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            logging.info("SMOTE applied.")
    else:
        logging.warning("Training data is empty or has no features after processing. SMOTE not applied.")

    # --- 6. Save Processed Data and Artifacts to GCS ---
    # ... (Saving logic remains the same, just ensure correct DataFrame construction and names)
    # Construct DataFrames with processed_feature_names, which should now be correct.
    # Handle empty data case for DataFrames.
    if X_train_resampled.shape[1] == 0 and len(processed_feature_names) > 0:
        logging.warning("X_train_resampled has no columns, but processed_feature_names is not empty. Adjusting names.")
        # This case should ideally not happen if processed_feature_names is derived correctly
        df_x_train_resampled = pd.DataFrame(X_train_resampled)  # No columns if X_train_resampled is empty
    else:
        df_x_train_resampled = pd.DataFrame(X_train_resampled,
                                            columns=processed_feature_names if X_train_resampled.shape[1] == len(
                                                processed_feature_names) else None)

    df_y_train_resampled = pd.Series(y_train_resampled, name=TARGET_COLUMN_CF2).to_frame()

    if X_val_processed.shape[1] == 0 and len(processed_feature_names) > 0:
        df_x_val_processed = pd.DataFrame(X_val_processed)
    else:
        df_x_val_processed = pd.DataFrame(X_val_processed,
                                          columns=processed_feature_names if X_val_processed.shape[1] == len(
                                              processed_feature_names) else None)

    df_y_val = pd.Series(y_val, name=TARGET_COLUMN_CF2).to_frame()

    if X_test_processed.shape[1] == 0 and len(processed_feature_names) > 0:
        df_x_test_processed = pd.DataFrame(X_test_processed)
    else:
        df_x_test_processed = pd.DataFrame(X_test_processed,
                                           columns=processed_feature_names if X_test_processed.shape[1] == len(
                                               processed_feature_names) else None)

    df_y_test = pd.Series(y_test, name=TARGET_COLUMN_CF2).to_frame()

    # ... (rest of GCS paths and saving logic) ...
    # (Ensure gcs_paths_... and metadata object use the unique numerical_feature_names and categorical_feature_names)
    gcs_path_x_train = f"{GCS_OUTPUT_PREFIX}/processed_data_ohe/X_train_resampled.csv"
    # ... (define all other GCS paths) ...
    gcs_path_y_train = f"{GCS_OUTPUT_PREFIX}/processed_data_ohe/y_train_resampled.csv"
    gcs_path_x_val = f"{GCS_OUTPUT_PREFIX}/processed_data_ohe/X_val_processed.csv"
    gcs_path_y_val = f"{GCS_OUTPUT_PREFIX}/processed_data_ohe/y_val.csv"
    gcs_path_x_test = f"{GCS_OUTPUT_PREFIX}/processed_data_ohe/X_test_processed.csv"
    gcs_path_y_test = f"{GCS_OUTPUT_PREFIX}/processed_data_ohe/y_test.csv"
    gcs_path_preprocessor = f"{GCS_OUTPUT_PREFIX}/preprocessor_ohe/preprocessor_ohe.joblib"
    gcs_path_metadata = f"{GCS_OUTPUT_PREFIX}/metadata/preprocessing_ohe_metadata.json"

    # Save non-empty dataframes
    if not df_x_train_resampled.empty: save_dataframe_to_gcs(df_x_train_resampled, GCS_BUCKET, gcs_path_x_train)
    if not df_y_train_resampled.empty: save_dataframe_to_gcs(df_y_train_resampled, GCS_BUCKET, gcs_path_y_train)
    if not df_x_val_processed.empty: save_dataframe_to_gcs(df_x_val_processed, GCS_BUCKET, gcs_path_x_val)
    if not df_y_val.empty: save_dataframe_to_gcs(df_y_val, GCS_BUCKET, gcs_path_y_val)
    if not df_x_test_processed.empty: save_dataframe_to_gcs(df_x_test_processed, GCS_BUCKET, gcs_path_x_test)
    if not df_y_test.empty: save_dataframe_to_gcs(df_y_test, GCS_BUCKET, gcs_path_y_test)

    save_object_to_gcs(preprocessor, GCS_BUCKET, gcs_path_preprocessor, use_joblib=True)

    metadata = {
        "dataset_name": "FinBench_cf2_OHE",
        "gcs_output_prefix": GCS_OUTPUT_PREFIX,
        "target_column": TARGET_COLUMN_CF2,
        "original_numerical_features": numerical_feature_names,  # These are now unique
        "original_categorical_features": categorical_feature_names,  # These are now unique
        "processed_feature_names_list": processed_feature_names if isinstance(processed_feature_names, list) else [],
        "preprocessing_type": "OneHotEncoding",
        # Ensure condition checks for non-empty processed data for SMOTE
        "smote_applied_to_train": minority_class_count > 1 and X_train_processed.shape[0] > 0 and
                                  X_train_processed.shape[1] > 0,
        "gcs_paths": {
            "X_train_resampled": f"gs://{GCS_BUCKET}/{gcs_path_x_train}" if not df_x_train_resampled.empty else None,
            "y_train_resampled": f"gs://{GCS_BUCKET}/{gcs_path_y_train}" if not df_y_train_resampled.empty else None,
            "X_val_processed": f"gs://{GCS_BUCKET}/{gcs_path_x_val}" if not df_x_val_processed.empty else None,
            "y_val": f"gs://{GCS_BUCKET}/{gcs_path_y_val}" if not df_y_val.empty else None,
            "X_test_processed": f"gs://{GCS_BUCKET}/{gcs_path_x_test}" if not df_x_test_processed.empty else None,
            "y_test": f"gs://{GCS_BUCKET}/{gcs_path_y_test}" if not df_y_test.empty else None,
            "preprocessor": f"gs://{GCS_BUCKET}/{gcs_path_preprocessor}",
            "metadata": f"gs://{GCS_BUCKET}/{gcs_path_metadata}"
        }
    }
    save_object_to_gcs(metadata, GCS_BUCKET, gcs_path_metadata, use_joblib=False)
    logging.info("--- OHE Preprocessing for FinBench cf2 Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load local FinBench cf2 data, preprocess (OHE), and save to GCS.")
    # ... (argparse remains the same) ...
    parser.add_argument("--train-file-path", type=str, required=True, help="Local file path to the training data CSV.")
    parser.add_argument("--val-file-path", type=str, required=True, help="Local file path to the validation data CSV.")
    parser.add_argument("--test-file-path", type=str, required=True, help="Local file path to the test data CSV.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name for output.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True,
                        help="GCS prefix (folder path) to save processed data and artifacts (e.g., 'finbench_cf2/processed_ohe').")
    args = parser.parse_args()
    main(args)
