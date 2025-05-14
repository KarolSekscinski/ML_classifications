# preprocess_finbench_cf2_ft.py
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from io import StringIO, BytesIO
import joblib
import json
import os
import ast

import gcs_utils  # Make sure this file is available

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TARGET_COLUMN_CF2 = 'y'
SMOTE_K_NEIGHBORS_DEFAULT = 5


# --- Helper Functions (Shared with _ohe script) ---
def parse_string_to_list(s):
    try:
        s_cleaned = s.strip()
        if s_cleaned.startswith('[') and s_cleaned.endswith(']'):
            s_content = s_cleaned[1:-1].strip()
            s_content = ' '.join(s_content.split())
            s_content = s_content.replace(' ', ', ')
            s_eval_ready = f"[{s_content}]"
            return ast.literal_eval(s_eval_ready)
        else:
            return ast.literal_eval(s)
    except Exception:
        try:
            return [float(x) for x in s.split()]
        except ValueError:
            return []


def reconstruct_dataframe(df_raw, all_feature_names, cat_indices):
    reconstructed_data = []
    for index, row in df_raw.iterrows():
        try:
            x_unscale_values = parse_string_to_list(row['X_ml_unscale'])
            if len(x_unscale_values) != len(all_feature_names):
                logging.error(
                    f"Row {index}: Mismatch X_ml_unscale ({len(x_unscale_values)}) vs col_name ({len(all_feature_names)}). Skipping.")
                continue
            current_row_data = dict(zip(all_feature_names, x_unscale_values))
            current_row_data[TARGET_COLUMN_CF2] = row[TARGET_COLUMN_CF2]
            reconstructed_data.append(current_row_data)
        except Exception as e:
            logging.error(f"Error processing row {index}: {e}. Skipping.")
            continue
    flat_df = pd.DataFrame(reconstructed_data)
    for i in cat_indices:  # Convert categorical columns to string then category for OrdinalEncoder
        col_name = all_feature_names[i]
        if col_name in flat_df.columns:
            flat_df[col_name] = flat_df[col_name].astype(str)  # Crucial for OrdinalEncoder to treat them as discrete
    return flat_df


def load_and_prepare_data(file_path, common_all_feature_names, common_cat_indices):
    raw_df = pd.read_csv(file_path)
    if raw_df.empty: return None
    return reconstruct_dataframe(raw_df, common_all_feature_names, common_cat_indices)


def save_dataframe_to_gcs(df_input, gcs_bucket_name, gcs_blob_name):  # Renamed df to df_input
    logging.info(f"Saving DataFrame to GCS: gs://{gcs_bucket_name}/{gcs_blob_name} (Input shape: {df_input.shape})")
    try:
        if df_input.empty:
            logging.warning(f"Input DataFrame for {gcs_blob_name} is empty. Skipping GCS upload.")
            # Optionally, upload an empty file or a file with just headers if required by downstream tasks
            # For now, we just skip.
            return

        # Work on a copy to avoid SettingWithCopyWarning and modifying original df
        df = df_input.copy()

        # Standardize float types to float64
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(np.float64)  # Explicitly use numpy.float64

        # Handle integer-like columns to save them as nullable integers (Int64) if possible,
        # or float64 if they contain NaNs and cannot be perfect integers.
        for col in df.columns:
            # Check if it's numeric but not already a definite float (could be int, or object containing numbers)
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_float_dtype(df[col]):
                # Attempt conversion to nullable integer if all non-NaN values are whole numbers
                try:
                    if df[col].isna().any():  # If there are NaNs
                        # Check if all non-NaNs are whole numbers
                        if (df[col].dropna() == df[col].dropna().round()).all():
                            df[col] = df[col].astype(pd.Int64Dtype())
                        else:  # Has NaNs and non-integer numbers, must be float
                            df[col] = df[col].astype(np.float64)
                    else:  # No NaNs
                        if (df[col] == df[col].round()).all():  # All are whole numbers
                            df[col] = df[col].astype(pd.Int64Dtype())
                        else:  # No NaNs but not all whole numbers (e.g. float stored as object), make it float
                            df[col] = df[col].astype(np.float64)
                except Exception as e_int_conv:
                    logging.warning(
                        f"Could not precisely convert column {col} to Int64/float64, leaving as is or trying general float: {e_int_conv}")
                    # Fallback if specific int conversion fails but it looks numeric
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except Exception:
                        pass  # Leave as is if all fails

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        gcs_utils.upload_string_to_gcs(gcs_bucket_name, csv_string, gcs_blob_name, content_type='text/csv')
        logging.info(
            f"Successfully saved {gcs_blob_name} to GCS. Shape: {df.shape}, Dtypes:\n{df.dtypes.value_counts().to_string()}")

    except Exception as e:
        logging.error(f"ERROR saving DataFrame to GCS ({gcs_blob_name}): {e}")
        import traceback
        logging.error(traceback.format_exc())  # Log full traceback for this error
        raise

def save_object_to_gcs(obj, gcs_bucket_name, gcs_blob_name, use_joblib=True):
    # (Identical to _ohe script, copied for completeness)
    logging.info(f"Saving object to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    try:
        if gcs_blob_name.endswith(".joblib") and use_joblib:
            with BytesIO() as buf:
                joblib.dump(obj, buf); buf.seek(0); obj_bytes = buf.read()
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
            raise ValueError("Unsupported file type. Use .joblib or .json")
        logging.info(f"Successfully saved object to GCS: gs://{gcs_bucket_name}/{gcs_blob_name}")
    except Exception as e:
        logging.error(f"ERROR saving object to GCS ({gcs_blob_name}): {e}")
        raise


def main(args):
    logging.info("--- Starting Data Preprocessing for FinBench cf2 (FT-Transformer) ---")
    GCS_BUCKET = args.gcs_bucket
    GCS_OUTPUT_PREFIX = args.gcs_output_prefix.strip('/')

    # --- 1. Load Structure from Training Data ---
    logging.info(f"Loading training data from {args.train_file_path} to determine structure...")
    raw_train_df_struct_only = pd.read_csv(args.train_file_path, nrows=1)
    if raw_train_df_struct_only.empty:
        logging.error("Training file is empty. Cannot determine data structure.");
        return
    try:
        common_all_feature_names = parse_string_to_list(raw_train_df_struct_only['col_name'].iloc[0])
        common_cat_indices = [int(i) for i in parse_string_to_list(raw_train_df_struct_only['cat_idx'].iloc[0])]
        # Get cardinalities directly from metadata for FT-Transformer consistency
        common_cat_dims = [int(d) for d in parse_string_to_list(raw_train_df_struct_only['cat_dim'].iloc[0])]
    except Exception as e:
        logging.error(f"Failed to parse structural columns from training data: {e}");
        return

    categorical_feature_names = [common_all_feature_names[i] for i in common_cat_indices]
    numerical_feature_names = [name for i, name in enumerate(common_all_feature_names) if i not in common_cat_indices]

    logging.info(f"Target column: {TARGET_COLUMN_CF2}")
    logging.info(f"Derived Numerical Features ({len(numerical_feature_names)}): {numerical_feature_names}")
    logging.info(f"Derived Categorical Features ({len(categorical_feature_names)}): {categorical_feature_names}")
    logging.info(f"Provided Categorical Dimensions: {common_cat_dims}")
    if len(categorical_feature_names) != len(common_cat_dims):
        logging.error(
            f"Mismatch: Num categorical features ({len(categorical_feature_names)}) != Num cat_dims ({len(common_cat_dims)}).")
        return

    # --- 2. Load and Reconstruct Full Datasets ---
    train_df = load_and_prepare_data(args.train_file_path, common_all_feature_names, common_cat_indices)
    val_df = load_and_prepare_data(args.val_file_path, common_all_feature_names, common_cat_indices)
    test_df = load_and_prepare_data(args.test_file_path, common_all_feature_names, common_cat_indices)
    if train_df is None or val_df is None or test_df is None: return

    y_train = train_df[TARGET_COLUMN_CF2]
    y_val = val_df[TARGET_COLUMN_CF2]
    y_test = test_df[TARGET_COLUMN_CF2]
    logging.info(f"Target distribution in y_train before SMOTE:\n{y_train.value_counts(normalize=True)}")

    # --- 3. Preprocessing Pipelines Setup ---
    numerical_pipeline = Pipeline([
        ('imputer_median', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # OrdinalEncoder for FT-Transformer. Ensure dtype is int.
    # Unknown values will be encoded as a new integer (-999 then mapped if needed).
    categorical_pipeline = Pipeline([
        ('imputer_mode', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-999, dtype=int))
    ])

    # --- 4. Fit & Transform Numerical Features ---
    X_train_num_df = train_df[numerical_feature_names]
    X_val_num_df = val_df[numerical_feature_names]
    X_test_num_df = test_df[numerical_feature_names]

    if numerical_feature_names:
        logging.info("Fitting and transforming numerical features...")
        numerical_pipeline.fit(X_train_num_df)
        X_train_num_scaled = numerical_pipeline.transform(X_train_num_df)
        X_val_num_scaled = numerical_pipeline.transform(X_val_num_df)
        X_test_num_scaled = numerical_pipeline.transform(X_test_num_df)
    else:  # Handle no numerical features
        X_train_num_scaled = np.array([[] for _ in range(len(X_train_num_df))])
        X_val_num_scaled = np.array([[] for _ in range(len(X_val_num_df))])
        X_test_num_scaled = np.array([[] for _ in range(len(X_test_num_df))])

    # --- 5. Fit & Transform Categorical Features ---
    X_train_cat_df = train_df[categorical_feature_names]
    X_val_cat_df = val_df[categorical_feature_names]
    X_test_cat_df = test_df[categorical_feature_names]

    cat_cardinalities_from_encoder = []
    category_mappings = {}

    if categorical_feature_names:
        logging.info("Fitting and transforming categorical features (OrdinalEncoding)...")
        # Important: OrdinalEncoder needs string inputs to correctly map categories.
        # reconstruct_dataframe already ensures this by converting cat columns to string then category.
        categorical_pipeline.fit(X_train_cat_df)
        X_train_cat_indices = categorical_pipeline.transform(X_train_cat_df)
        X_val_cat_indices = categorical_pipeline.transform(X_val_cat_df)
        X_test_cat_indices = categorical_pipeline.transform(X_test_cat_df)

        ordinal_encoder_fitted = categorical_pipeline.named_steps['ordinal']
        cat_cardinalities_from_encoder = [len(cats) for cats in ordinal_encoder_fitted.categories_]
        for i, feature in enumerate(categorical_feature_names):
            mapping = {cat_val: int(idx) for idx, cat_val in enumerate(ordinal_encoder_fitted.categories_[i])}
            category_mappings[feature] = mapping
            # Check if unknown_value was encountered and if it created an extra category
            if (X_train_cat_indices[:, i] == len(ordinal_encoder_fitted.categories_[i])).any() or \
                    (X_val_cat_indices[:, i] == len(ordinal_encoder_fitted.categories_[i])).any() or \
                    (X_test_cat_indices[:, i] == len(ordinal_encoder_fitted.categories_[i])).any():
                if ordinal_encoder_fitted.unknown_value == -999:  # If -999 was our placeholder
                    # This means a new category was effectively added for unknowns by the encoder
                    cat_cardinalities_from_encoder[i] += 1
                    mapping[f"__UNKNOWN_{ordinal_encoder_fitted.unknown_value}"] = len(
                        ordinal_encoder_fitted.categories_[i])

        logging.info(f"Cardinalities from OrdinalEncoder: {cat_cardinalities_from_encoder}")
        logging.info(f"Original cardinalities from cat_dim: {common_cat_dims}")
        # You might want to assert or warn if these differ significantly from common_cat_dims
        # For FT-Transformer, the actual cardinalities from the encoder are what matter.
    else:  # Handle no categorical features
        X_train_cat_indices = np.array([[] for _ in range(len(X_train_cat_df))])
        X_val_cat_indices = np.array([[] for _ in range(len(X_val_cat_df))])
        X_test_cat_indices = np.array([[] for _ in range(len(X_test_cat_df))])

    # --- 6. Combine Processed Train Data for SMOTE ---
    if X_train_num_scaled.shape[1] == 0 and X_train_cat_indices.shape[1] == 0:
        logging.error("No features to process for SMOTE. Exiting.")
        return
    elif X_train_num_scaled.shape[1] == 0:
        X_train_processed_combined = X_train_cat_indices
    elif X_train_cat_indices.shape[1] == 0:
        X_train_processed_combined = X_train_num_scaled
    else:
        X_train_processed_combined = np.concatenate([X_train_num_scaled, X_train_cat_indices], axis=1)
    logging.info(f"X_train_processed_combined shape for SMOTE: {X_train_processed_combined.shape}")

    # --- 7. Apply SMOTE to Combined Training Data ---
    # (SMOTE logic is similar to _ohe script)
    minority_class_count = y_train.value_counts().min()
    k_neighbors_smote = 1
    X_train_resampled_combined = X_train_processed_combined
    y_train_resampled = y_train.copy()

    if X_train_processed_combined.shape[0] > 0 and X_train_processed_combined.shape[1] > 0:
        if minority_class_count <= 1:
            logging.warning(f"Minority class has {minority_class_count} sample(s). SMOTE not applied.")
        elif minority_class_count <= SMOTE_K_NEIGHBORS_DEFAULT:
            k_neighbors_smote = minority_class_count - 1
            if k_neighbors_smote < 1: k_neighbors_smote = 1
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
            X_train_resampled_combined, y_train_resampled = smote.fit_resample(X_train_processed_combined, y_train)
            logging.info(f"SMOTE applied with k_neighbors={k_neighbors_smote}.")
        else:
            k_neighbors_smote = SMOTE_K_NEIGHBORS_DEFAULT
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
            X_train_resampled_combined, y_train_resampled = smote.fit_resample(X_train_processed_combined, y_train)
            logging.info(f"SMOTE applied with k_neighbors={k_neighbors_smote}.")
    else:
        logging.warning("Training data empty or no features after processing. SMOTE not applied.")

    logging.info(f"Shapes after SMOTE: X_combined={X_train_resampled_combined.shape}, y={y_train_resampled.shape}")

    # --- 8. Separate Resampled Data ---
    num_feat_count = X_train_num_scaled.shape[1]
    X_train_num_resampled = X_train_resampled_combined[:, :num_feat_count] if num_feat_count > 0 else np.array(
        [[] for _ in range(X_train_resampled_combined.shape[0])])

    if X_train_cat_indices.shape[1] > 0:
        X_train_cat_indices_resampled_float = X_train_resampled_combined[:, num_feat_count:]
        X_train_cat_indices_resampled = np.round(X_train_cat_indices_resampled_float).astype(int)
        # Clip resampled categorical indices to valid ranges
        for i in range(X_train_cat_indices_resampled.shape[1]):
            max_val_encoder = cat_cardinalities_from_encoder[i] - 1  # Max index from encoder
            unknown_val_code = -999  # The placeholder used in OrdinalEncoder
            # If SMOTE generated something like the unknown_value placeholder, map it to a valid category (e.g. 0)
            X_train_cat_indices_resampled[X_train_cat_indices_resampled[:, i] == unknown_val_code, i] = 0
            X_train_cat_indices_resampled[:, i] = np.clip(X_train_cat_indices_resampled[:, i], 0, max_val_encoder)
    else:
        X_train_cat_indices_resampled = np.array([[] for _ in range(X_train_resampled_combined.shape[0])])

    # --- 9. Save Processed Data and Artifacts to GCS ---
    logging.info("Saving FT-Transformer processed data, pipelines, and metadata to GCS...")
    df_x_train_num = pd.DataFrame(X_train_num_resampled, columns=numerical_feature_names)
    df_x_train_cat = pd.DataFrame(X_train_cat_indices_resampled, columns=categorical_feature_names)
    df_y_train = pd.Series(y_train_resampled, name=TARGET_COLUMN_CF2).to_frame()

    df_x_val_num = pd.DataFrame(X_val_num_scaled, columns=numerical_feature_names)
    df_x_val_cat = pd.DataFrame(X_val_cat_indices, columns=categorical_feature_names)
    df_y_val = pd.Series(y_val, name=TARGET_COLUMN_CF2).to_frame()

    df_x_test_num = pd.DataFrame(X_test_num_scaled, columns=numerical_feature_names)
    df_x_test_cat = pd.DataFrame(X_test_cat_indices, columns=categorical_feature_names)
    df_y_test = pd.Series(y_test, name=TARGET_COLUMN_CF2).to_frame()

    gcs_path_x_train_num = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_train_num_scaled.csv"
    gcs_path_x_train_cat = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_train_cat_indices.csv"
    # ... (rest of the GCS paths as in _ohe script, but with _ft suffix)
    gcs_path_y_train = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/y_train_resampled.csv"
    gcs_path_x_val_num = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_val_num_scaled.csv"
    gcs_path_x_val_cat = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_val_cat_indices.csv"
    gcs_path_y_val = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/y_val.csv"
    gcs_path_x_test_num = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_test_num_scaled.csv"
    gcs_path_x_test_cat = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/X_test_cat_indices.csv"
    gcs_path_y_test = f"{GCS_OUTPUT_PREFIX}/processed_data_ft/y_test.csv"
    gcs_path_num_pipeline = f"{GCS_OUTPUT_PREFIX}/preprocessor_ft/numerical_pipeline_ft.joblib"
    gcs_path_cat_pipeline = f"{GCS_OUTPUT_PREFIX}/preprocessor_ft/categorical_pipeline_ft.joblib"
    gcs_path_metadata = f"{GCS_OUTPUT_PREFIX}/metadata/preprocessing_ft_metadata.json"

    if df_x_train_num.shape[1] > 0: save_dataframe_to_gcs(df_x_train_num, GCS_BUCKET, gcs_path_x_train_num)
    if df_x_train_cat.shape[1] > 0: save_dataframe_to_gcs(df_x_train_cat, GCS_BUCKET, gcs_path_x_train_cat)
    save_dataframe_to_gcs(df_y_train, GCS_BUCKET, gcs_path_y_train)
    if df_x_val_num.shape[1] > 0: save_dataframe_to_gcs(df_x_val_num, GCS_BUCKET, gcs_path_x_val_num)
    if df_x_val_cat.shape[1] > 0: save_dataframe_to_gcs(df_x_val_cat, GCS_BUCKET, gcs_path_x_val_cat)
    save_dataframe_to_gcs(df_y_val, GCS_BUCKET, gcs_path_y_val)
    if df_x_test_num.shape[1] > 0: save_dataframe_to_gcs(df_x_test_num, GCS_BUCKET, gcs_path_x_test_num)
    if df_x_test_cat.shape[1] > 0: save_dataframe_to_gcs(df_x_test_cat, GCS_BUCKET, gcs_path_x_test_cat)
    save_dataframe_to_gcs(df_y_test, GCS_BUCKET, gcs_path_y_test)

    if numerical_feature_names: save_object_to_gcs(numerical_pipeline, GCS_BUCKET, gcs_path_num_pipeline)
    if categorical_feature_names: save_object_to_gcs(categorical_pipeline, GCS_BUCKET, gcs_path_cat_pipeline)

    metadata = {
        "dataset_name": "FinBench_cf2_FT",
        "gcs_output_prefix": GCS_OUTPUT_PREFIX,
        "target_column": TARGET_COLUMN_CF2,
        "original_numerical_features": numerical_feature_names,
        "original_categorical_features": categorical_feature_names,
        "cat_feature_cardinalities_from_encoder": cat_cardinalities_from_encoder if categorical_feature_names else [],
        "cat_feature_cardinalities_from_dataset_metadata": common_cat_dims if categorical_feature_names else [],
        "category_mappings_from_encoder": category_mappings if categorical_feature_names else {},
        "preprocessing_type": "OrdinalEncoding",
        "smote_applied_to_train": minority_class_count > 1 and X_train_processed_combined.shape[0] > 0 and
                                  X_train_processed_combined.shape[1] > 0,
        "gcs_paths": {  # Paths to individual FT files
            "X_train_num_scaled": f"gs://{GCS_BUCKET}/{gcs_path_x_train_num}" if numerical_feature_names else None,
            "X_train_cat_indices": f"gs://{GCS_BUCKET}/{gcs_path_x_train_cat}" if categorical_feature_names else None,
            "y_train_resampled": f"gs://{GCS_BUCKET}/{gcs_path_y_train}",
            "X_val_num_scaled": f"gs://{GCS_BUCKET}/{gcs_path_x_val_num}" if numerical_feature_names else None,
            "X_val_cat_indices": f"gs://{GCS_BUCKET}/{gcs_path_x_val_cat}" if categorical_feature_names else None,
            "y_val": f"gs://{GCS_BUCKET}/{gcs_path_y_val}",
            "X_test_num_scaled": f"gs://{GCS_BUCKET}/{gcs_path_x_test_num}" if numerical_feature_names else None,
            "X_test_cat_indices": f"gs://{GCS_BUCKET}/{gcs_path_x_test_cat}" if categorical_feature_names else None,
            "y_test": f"gs://{GCS_BUCKET}/{gcs_path_y_test}",
            "numerical_pipeline": f"gs://{GCS_BUCKET}/{gcs_path_num_pipeline}" if numerical_feature_names else None,
            "categorical_pipeline": f"gs://{GCS_BUCKET}/{gcs_path_cat_pipeline}" if categorical_feature_names else None,
            "metadata": f"gs://{GCS_BUCKET}/{gcs_path_metadata}"
        }
    }
    save_object_to_gcs(metadata, GCS_BUCKET, gcs_path_metadata, use_joblib=False)
    logging.info("--- FT-Transformer Preprocessing for FinBench cf2 Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load local FinBench cf2 data, preprocess for FT-Transformer, and save to GCS.")
    parser.add_argument("--train-file-path", type=str, required=True, help="Local file path to the training data CSV.")
    parser.add_argument("--val-file-path", type=str, required=True, help="Local file path to the validation data CSV.")
    parser.add_argument("--test-file-path", type=str, required=True, help="Local file path to the test data CSV.")
    parser.add_argument("--gcs-bucket", type=str, required=True, help="GCS bucket name for output.")
    parser.add_argument("--gcs-output-prefix", type=str, required=True,
                        help="GCS prefix (folder path) to save processed data and artifacts (e.g., 'finbench_cf2/processed_ft').")
    args = parser.parse_args()
    main(args)