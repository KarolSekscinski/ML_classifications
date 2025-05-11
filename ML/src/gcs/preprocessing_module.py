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


# Assuming gcs_utils is in the same directory or accessible via PYTHONPATH
# and contains custom_print, upload_bytes_to_gcs, upload_string_to_gcs
# For this module, we'll assume these functions are passed or gcs_utils is imported by the caller.

def load_and_preprocess_data(
        data_source_uri,
        gcs_bucket_name,
        gcs_output_prefix,
        custom_print_func,
        upload_bytes_func,
        upload_string_func  # For saving CSVs as strings if preferred, or use upload_bytes_func
):
    """
    Loads data, performs all preprocessing, applies SMOTE, saves artifacts,
    and returns processed data splits.
    """
    custom_print_func("--- Starting Data Loading and Preprocessing ---")

    # --- 1. Load Data ---
    custom_print_func("Loading data...")
    try:
        # If data_source_uri is a GCS URI, download it first.
        # For simplicity, this example assumes data_source_uri is a local path or StringIO/BytesIO.
        # In a container, you'd typically download from GCS to a local temp file or load into memory.
        if isinstance(data_source_uri, str) and data_source_uri.startswith("gs://"):
            # Placeholder for GCS download logic
            custom_print_func(f"Attempting to load from GCS URI: {data_source_uri} (not implemented in this snippet)")
            # Example:
            # from google.cloud import storage
            # client = storage.Client()
            # bucket_name_src, blob_name_src = data_source_uri.replace("gs://", "").split("/", 1)
            # bucket = client.bucket(bucket_name_src)
            # blob = bucket.blob(blob_name_src)
            # data_source_uri = BytesIO(blob.download_as_bytes())
            # For now, assuming it's handled before calling this or is a local path/buffer
            pass  # Pass if it's already a buffer from main script

        df = pd.read_csv(data_source_uri)
        custom_print_func(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        custom_print_func(f"ERROR: CSV file not found at: {data_source_uri}.")
        return None  # Or raise error
    except Exception as e:
        custom_print_func(f"Error loading data: {e}")
        return None  # Or raise error

    # --- 2. Initial Data Cleaning ---
    custom_print_func("\nPerforming initial data cleaning (handling -1 as NaN)...")
    cols_with_minus_one_as_nan = [
        'prev_address_months_count', 'current_address_months_count',
        'bank_months_count', 'session_length_in_minutes'
    ]
    for col in cols_with_minus_one_as_nan:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    # --- 3. Define Feature Types ---
    custom_print_func("\nDefining feature types...")
    target = 'fraud_bool'
    if target not in df.columns:
        custom_print_func(f"ERROR: Target column '{target}' not found.")
        return None

    numerical_features_orig = [  # Keep original list for FT-Transformer
        'income', 'name_email_similarity', 'prev_address_months_count',
        'current_address_months_count', 'days_since_request', 'intended_balcon_amount',
        'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w',
        'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score',
        'bank_months_count', 'proposed_credit_limit', 'session_length_in_minutes',
        'device_distinct_emails_8w'
    ]
    numerical_features = [col for col in numerical_features_orig if col in df.columns]

    categorical_features_orig = [  # Keep original list for FT-Transformer and cardinalities
        'customer_age', 'payment_type', 'employment_status', 'housing_status',
        'source', 'device_os', 'month'
    ]
    categorical_features = [col for col in categorical_features_orig if col in df.columns]

    # Calculate cardinalities for FT-Transformer (before OHE, after imputation)
    cat_cardinalities = []
    if categorical_features:
        temp_cat_imputer = SimpleImputer(strategy='most_frequent')
        # Impute on a copy to avoid changing df directly here if X_train is used later
        df_cat_imputed_for_card = df[categorical_features].copy()  # Use the filtered list
        df_cat_imputed_for_card = pd.DataFrame(temp_cat_imputer.fit_transform(df_cat_imputed_for_card),
                                               columns=categorical_features)
        for col in categorical_features:  # Use the filtered list
            cat_cardinalities.append(df_cat_imputed_for_card[col].astype('category').nunique())
    custom_print_func(f"Determined cardinalities for categorical features: {cat_cardinalities}")

    # --- 4. Separate Features (X) and Target (y) ---
    X = df.drop(columns=[target])
    y = df[target]
    custom_print_func(f"\nShape of X (raw): {X.shape}, Shape of y (raw): {y.shape}")

    # --- 5. Split Data (Raw) ---
    custom_print_func("\nSplitting raw data into training and testing sets (80/20 split)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    custom_print_func(f"X_train_raw shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
    custom_print_func(f"X_test_raw shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")

    # --- 6. Preprocessing Pipelines Setup ---
    custom_print_func("\nSetting up preprocessing pipelines...")
    numerical_pipeline = Pipeline([
        ('imputer_median', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer_mode', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Ensure correct feature lists for ColumnTransformer based on X_train_raw
    X_numerical_cols_for_ct = [col for col in numerical_features if col in X_train_raw.columns]
    X_categorical_cols_for_ct = [col for col in categorical_features if col in X_train_raw.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, X_numerical_cols_for_ct),
            ('cat', categorical_pipeline, X_categorical_cols_for_ct)
        ],
        remainder='passthrough'
    )
    custom_print_func("Preprocessor configured.")

    # --- 7. Fit Preprocessor and Transform Data ---
    custom_print_func("\nFitting preprocessor on X_train_raw and transforming data...")
    preprocessor.fit(X_train_raw)
    custom_print_func("Preprocessor fitted.")

    X_train_processed = preprocessor.transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    custom_print_func("X_train_raw and X_test_raw transformed.")

    # Save the fitted preprocessor
    custom_print_func("Saving fitted preprocessor to GCS...")
    try:
        preprocessor_bytes = pickle.dumps(preprocessor)
        preprocessor_blob_name = f"{gcs_output_prefix}/preprocessor/preprocessor.pkl"
        upload_bytes_func(gcs_bucket_name, preprocessor_bytes, preprocessor_blob_name,
                          content_type='application/octet-stream')
    except Exception as e:
        custom_print_func(f"ERROR saving preprocessor to GCS: {e}")

    # Reconstruct feature names
    processed_feature_names = None
    try:
        ohe_feature_names = []
        if 'cat' in preprocessor.named_transformers_:
            cat_pipeline_fitted = preprocessor.named_transformers_['cat']
            if 'onehot' in cat_pipeline_fitted.named_steps:
                ohe_feature_names = list(cat_pipeline_fitted.named_steps['onehot'].get_feature_names_out(
                    X_categorical_cols_for_ct))  # Use the list passed to CT

        remainder_transformer_fitted = preprocessor.named_transformers_.get('remainder', None)
        original_cols_for_remainder = []
        if hasattr(remainder_transformer_fitted, 'get_feature_names_out'):
            original_cols_for_remainder = list(remainder_transformer_fitted.get_feature_names_out())
        elif hasattr(remainder_transformer_fitted, 'feature_names_in_'):  # Older scikit-learn might use this
            original_cols_for_remainder = list(remainder_transformer_fitted.feature_names_in_)
        else:  # Fallback logic
            processed_by_num_cat = set(X_numerical_cols_for_ct + X_categorical_cols_for_ct)
            original_cols_for_remainder = [col for col in X_train_raw.columns if col not in processed_by_num_cat]
            if preprocessor.remainder == 'drop':
                original_cols_for_remainder = []

        processed_feature_names = X_numerical_cols_for_ct + ohe_feature_names + original_cols_for_remainder
        custom_print_func(f"Processed feature names reconstructed. Total features: {len(processed_feature_names)}")
    except Exception as e:
        custom_print_func(f"Could not reconstruct feature names after OHE: {e}")

    # --- 8. Resampling (SMOTE) on Processed Training Data ---
    custom_print_func("\nApplying SMOTE to the processed training data...")
    minority_class_count = y_train.value_counts().min()
    k_neighbors_smote = min(5, minority_class_count - 1) if minority_class_count > 1 else 1

    X_train_resampled = X_train_processed
    y_train_resampled = y_train.copy()

    if k_neighbors_smote < 1 and minority_class_count <= 1:
        custom_print_func(
            f"WARNING: Minority class in y_train has only {minority_class_count} sample(s). SMOTE cannot be applied effectively.")
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
        try:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
            custom_print_func("SMOTE applied successfully to training data.")
            custom_print_func(
                f"Shape of X_train_resampled: {X_train_resampled.shape}, y_train_resampled: {y_train_resampled.shape}")
            custom_print_func(
                f"Fraud distribution in y_train_resampled: \n{pd.Series(y_train_resampled).value_counts(normalize=True)}")
        except ValueError as e:
            custom_print_func(f"Error during SMOTE: {e}. Proceeding with original imbalanced processed training data.")

    # --- 9. Save Processed Data to GCS as CSVs ---
    custom_print_func("\nSaving processed data splits to GCS as CSVs...")
    try:
        # X_train_resampled
        if processed_feature_names:
            df_x_train_resampled = pd.DataFrame(X_train_resampled, columns=processed_feature_names)
        else:
            df_x_train_resampled = pd.DataFrame(X_train_resampled)
        csv_buffer = StringIO()
        df_x_train_resampled.to_csv(csv_buffer, index=False)
        upload_string_func(gcs_bucket_name, csv_buffer.getvalue(),
                           f"{gcs_output_prefix}/processed_data/X_train_resampled.csv", content_type='text/csv')

        # y_train_resampled
        df_y_train_resampled = pd.Series(y_train_resampled, name=target)
        csv_buffer = StringIO()
        df_y_train_resampled.to_csv(csv_buffer, index=False, header=True)
        upload_string_func(gcs_bucket_name, csv_buffer.getvalue(),
                           f"{gcs_output_prefix}/processed_data/y_train_resampled.csv", content_type='text/csv')

        # X_test_processed
        if processed_feature_names:
            df_x_test_processed = pd.DataFrame(X_test_processed, columns=processed_feature_names)
        else:
            df_x_test_processed = pd.DataFrame(X_test_processed)
        csv_buffer = StringIO()
        df_x_test_processed.to_csv(csv_buffer, index=False)
        upload_string_func(gcs_bucket_name, csv_buffer.getvalue(),
                           f"{gcs_output_prefix}/processed_data/X_test_processed.csv", content_type='text/csv')

        # y_test
        df_y_test = pd.Series(y_test, name=target)
        csv_buffer = StringIO()
        df_y_test.to_csv(csv_buffer, index=False, header=True)
        upload_string_func(gcs_bucket_name, csv_buffer.getvalue(), f"{gcs_output_prefix}/processed_data/y_test.csv",
                           content_type='text/csv')

        custom_print_func("Processed data CSVs saved to GCS.")
    except Exception as e:
        custom_print_func(f"ERROR saving processed data CSVs to GCS: {e}")

    custom_print_func("\n--- Data Loading and Preprocessing Complete ---")

    # Return the original list of categorical feature names (before OHE)
    # and the original list of numerical feature names
    return (X_train_resampled, y_train_resampled, X_test_processed, y_test, X_train_raw,
            processed_feature_names, cat_cardinalities,
            numerical_features,  # Original numerical feature names
            categorical_features  # Original categorical feature names
            )

