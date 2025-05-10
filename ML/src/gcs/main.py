import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from io import StringIO, BytesIO
import time
import pickle  # Added for pickling the preprocessor

# Import custom modules
import gcs_utils
import svm_pipeline
import xgboost_pipeline
import mlp_pipeline

# --- Configuration ---
GCS_BUCKET_NAME_MAIN = "licencjat_ml_classification"  # Ensure this matches gcs_utils or pass it
gcs_output_prefix = gcs_utils.get_gcs_output_prefix()  # Generate unique prefix for this run

# Script Execution Configuration
csv_data_sample = """fraud_bool,income,name_email_similarity,prev_address_months_count,current_address_months_count,customer_age,days_since_request,intended_balcon_amount,payment_type,zip_count_4w,velocity_6h,velocity_24h,velocity_4w,bank_branch_count_8w,date_of_birth_distinct_emails_4w,employment_status,credit_risk_score,email_is_free,housing_status,phone_home_valid,phone_mobile_valid,bank_months_count,has_other_cards,proposed_credit_limit,foreign_request,source,session_length_in_minutes,device_os,keep_alive_session,device_distinct_emails_8w,device_fraud_count,month
0,0.3,0.986506310633034,-1,25,40,0.0067353870811739,102.45371092469456,AA,1059,13096.035018400871,7850.955007125409,6742.080561007602,5,5,CB,163,1,BC,0,1,9,0,1500.0,0,INTERNET,16.224843433978073,linux,1,1,0,0
1,0.8,0.6174260062650061,-1,89,20,0.010095097878573,-0.8495509687507287,AD,1658,9223.283430930423,5745.251480643791,5941.6648588359885,3,18,CA,154,1,BC,1,1,2,0,1500.0,0,INTERNET,3.36385371062431,other,1,1,0,0
0,0.8,0.9967070206409232,9,14,40,0.0123163495250501,-1.4903855214855013,AB,1095,4471.472148765561,5471.988958014066,5992.555113248597,15,11,CA,89,1,BC,0,1,30,0,200.0,0,INTERNET,22.73055923496224,windows,0,1,0,0
0,0.6000000000000001,0.4750999462380287,11,14,30,0.0069908306305029,-1.8631006767118188,AB,3483,14431.993621381194,6755.34447929092,5970.336830507939,11,13,CA,90,1,BC,0,1,1,0,200.0,0,INTERNET,15.215816122068803,linux,1,1,0,0
1,0.9,0.8423068370138775,-1,29,40,5.742625847255586,47.152497798787614,AA,2339,7601.511579253147,5124.046929628144,5940.734211620649,1,6,CA,91,0,BC,1,1,26,0,200.0,0,INTERNET,3.743047928033851,other,0,1,0,0
0,0.3,0.11,-1,10,50,0.1,10,AA,1000,8000,6000,6000,10,10,CA,100,1,BC,1,1,10,1,1000,0,INTERNET,10,linux,0,1,0,0 
"""
data_source = StringIO(csv_data_sample)
# To load your full dataset from a local file:
# data_source = "Base.csv"
# To load from GCS:
# from google.cloud import storage
# client = storage.Client()
# bucket = client.bucket(GCS_BUCKET_NAME_MAIN) # Use the bucket name
# blob = bucket.blob("path/to/your/Base.csv") # Update with actual path in GCS
# data_source = BytesIO(blob.download_as_bytes())


# --- 1. Load Data ---
gcs_utils.custom_print("Starting Fraud Detection Pipeline...")
gcs_utils.custom_print(f"Output will be saved to GCS Bucket: {GCS_BUCKET_NAME_MAIN} under prefix: {gcs_output_prefix}")

gcs_utils.custom_print("Loading data...")
try:
    df = pd.read_csv(data_source)
    gcs_utils.custom_print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    gcs_utils.custom_print(f"ERROR: CSV file not found. Ensure 'Base.csv' is accessible or update 'data_source'.")
    gcs_utils.finalize_logs(GCS_BUCKET_NAME_MAIN, gcs_output_prefix)
    exit()
except Exception as e:
    gcs_utils.custom_print(f"Error loading data: {e}")
    gcs_utils.finalize_logs(GCS_BUCKET_NAME_MAIN, gcs_output_prefix)
    exit()

# --- 2. Initial Data Cleaning (from EDA) ---
gcs_utils.custom_print("\nPerforming initial data cleaning (handling -1 as NaN)...")
cols_with_minus_one_as_nan = [
    'prev_address_months_count',
    'current_address_months_count',
    'bank_months_count',
    'session_length_in_minutes'
]
for col in cols_with_minus_one_as_nan:
    if col in df.columns:
        df[col] = df[col].replace(-1, np.nan)

# --- 3. Define Feature Types (based on domain knowledge) ---
gcs_utils.custom_print("\nDefining feature types...")
target = 'fraud_bool'
numerical_features = [
    'income', 'name_email_similarity', 'prev_address_months_count',
    'current_address_months_count', 'days_since_request', 'intended_balcon_amount',
    'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w',
    'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score',
    'bank_months_count', 'proposed_credit_limit', 'session_length_in_minutes',
    'device_distinct_emails_8w'
]
numerical_features = [col for col in numerical_features if col in df.columns]
categorical_features = [
    'customer_age', 'payment_type', 'employment_status', 'housing_status',
    'source', 'device_os', 'month'
]
categorical_features = [col for col in categorical_features if col in df.columns]
# Binary features will be handled by 'remainder=passthrough'

gcs_utils.custom_print(f"Target: {target}")
# ... (optional: print feature list lengths)

# --- 4. Separate Features (X) and Target (y) ---
if target not in df.columns:
    gcs_utils.custom_print(f"ERROR: Target column '{target}' not found in the DataFrame.")
    gcs_utils.finalize_logs(GCS_BUCKET_NAME_MAIN, gcs_output_prefix)
    exit()
X = df.drop(columns=[target])
y = df[target]
gcs_utils.custom_print(f"\nShape of X: {X.shape}, Shape of y: {y.shape}")

# --- 5. Preprocessing Pipelines ---
gcs_utils.custom_print("\nSetting up preprocessing pipelines...")
numerical_pipeline = Pipeline([
    ('imputer_median', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer_mode', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
X_numerical = [col for col in numerical_features if col in X.columns]
X_categorical = [col for col in categorical_features if col in X.columns]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, X_numerical),
        ('cat', categorical_pipeline, X_categorical)
    ],
    remainder='passthrough'
)
gcs_utils.custom_print("Preprocessor configured.")

# --- 6. Split Data ---
gcs_utils.custom_print("\nSplitting data into training and testing sets (80/20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# ... (optional: print shapes and distributions)

# --- 7. Apply Preprocessing & Save Preprocessor ---
gcs_utils.custom_print("\nApplying preprocessing (fitting on train, transforming train and test)...")
# Fit the preprocessor on the training data
preprocessor.fit(X_train)
gcs_utils.custom_print("Preprocessor fitted on training data.")

# Save the fitted preprocessor to GCS
gcs_utils.custom_print("Saving fitted preprocessor to GCS...")
try:
    preprocessor_bytes = pickle.dumps(preprocessor)
    preprocessor_blob_name = f"{gcs_output_prefix}/preprocessor/preprocessor.pkl"
    gcs_utils.upload_bytes_to_gcs(GCS_BUCKET_NAME_MAIN, preprocessor_bytes, preprocessor_blob_name,
                                  content_type='application/octet-stream')
except Exception as e:
    gcs_utils.custom_print(f"ERROR saving preprocessor to GCS: {e}")

# Transform training and testing data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)
gcs_utils.custom_print("Training and testing data transformed.")

processed_feature_names = None
try:
    ohe_feature_names = []
    if 'cat' in preprocessor.named_transformers_:
        cat_pipeline = preprocessor.named_transformers_['cat']
        if 'onehot' in cat_pipeline.named_steps:
            ohe_feature_names = list(cat_pipeline.named_steps['onehot'].get_feature_names_out(X_categorical))

    remainder_transformer = preprocessor.named_transformers_.get('remainder', None)
    if hasattr(remainder_transformer, 'get_feature_names_out'):
        original_cols_for_remainder = list(remainder_transformer.get_feature_names_out())
    elif hasattr(remainder_transformer, 'feature_names_in_'):
        original_cols_for_remainder = list(remainder_transformer.feature_names_in_)
    else:
        processed_by_num_cat = set(X_numerical + X_categorical)
        original_cols_for_remainder = [col for col in X_train.columns if col not in processed_by_num_cat]
        if preprocessor.remainder == 'drop':
            original_cols_for_remainder = []
    processed_feature_names = X_numerical + ohe_feature_names + original_cols_for_remainder
    gcs_utils.custom_print(f"Processed feature names reconstructed. Total features: {len(processed_feature_names)}")
except Exception as e:
    gcs_utils.custom_print(f"Could not reconstruct feature names after OHE: {e}")
    gcs_utils.custom_print("Proceeding with NumPy arrays. SHAP plots might not have detailed feature names.")

# --- 8. Resampling (SMOTE) ---
gcs_utils.custom_print("\nApplying SMOTE to the training data...")
minority_class_count = y_train.value_counts().min()
k_neighbors_smote = min(5, minority_class_count - 1) if minority_class_count > 1 else 1
if k_neighbors_smote < 1 and minority_class_count <= 1:
    gcs_utils.custom_print(
        f"WARNING: Minority class in training set has only {minority_class_count} sample(s). SMOTE cannot be applied.")
    X_train_resampled, y_train_resampled = X_train_processed, y_train
else:
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        gcs_utils.custom_print("SMOTE applied successfully.")
        # ... (optional: print resampled shapes and distribution)
    except ValueError as e:
        gcs_utils.custom_print(f"Error during SMOTE: {e}. Proceeding with original imbalanced training data.")
        X_train_resampled, y_train_resampled = X_train_processed, y_train

# --- 9. Run SVM Pipeline ---
svm_pipeline.run_svm_pipeline(
    X_train_resampled, y_train_resampled, X_test_processed, y_test,
    processed_feature_names,
    GCS_BUCKET_NAME_MAIN, gcs_output_prefix,
    gcs_utils.custom_print, gcs_utils.save_plot_to_gcs, gcs_utils.upload_bytes_to_gcs
)

# --- 10. Run XGBoost Pipeline ---
xgboost_pipeline.run_xgboost_pipeline(
    X_train_resampled, y_train_resampled, X_test_processed, y_test,
    processed_feature_names,
    GCS_BUCKET_NAME_MAIN, gcs_output_prefix,
    gcs_utils.custom_print, gcs_utils.save_plot_to_gcs, gcs_utils.upload_bytes_to_gcs
)
# --- 11. Run MLP Pipeline ---
mlp_pipeline.run_mlp_pipeline(
    X_train_resampled, y_train_resampled, X_test_processed, y_test,
    processed_feature_names,
    GCS_BUCKET_NAME_MAIN, gcs_output_prefix,
    gcs_utils.custom_print, gcs_utils.save_plot_to_gcs, gcs_utils.upload_bytes_to_gcs
)

# --- Finalize Logging ---
gcs_utils.custom_print("\n--- Main Fraud Detection Pipeline Complete ---")
gcs_utils.finalize_logs(GCS_BUCKET_NAME_MAIN, gcs_output_prefix)

gcs_utils.custom_print("Script finished. All logs, plots, and models attempted to be saved to GCS.")

