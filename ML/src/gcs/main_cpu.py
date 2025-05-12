# main_cpu.py
import argparse
import pandas as pd
from io import StringIO, BytesIO
import sys
import os

# Add project root to path if needed, or ensure modules are installed/accessible
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gcs_utils import (
        custom_print,
        download_blob_to_memory,
        upload_string_to_gcs,
        upload_bytes_to_gcs # Needed by pipelines
    )
    from svm_pipeline import run_svm_pipeline
    from xgboost_pipeline import run_xgboost_pipeline
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    print("Ensure gcs_utils.py, svm_pipeline.py, and xgboost_pipeline.py are accessible.")
    sys.exit(1)


def load_csv_from_gcs(bucket_name, blob_name):
    """Downloads a CSV from GCS and loads it into a pandas DataFrame."""
    custom_print(f"Loading CSV from gs://{bucket_name}/{blob_name}")
    data_bytes = download_blob_to_memory(bucket_name, blob_name)
    if data_bytes is None:
        raise FileNotFoundError(f"Could not download {blob_name} from bucket {bucket_name}")
    try:
        df = pd.read_csv(BytesIO(data_bytes))
        custom_print(f"Loaded DataFrame from {blob_name}, shape: {df.shape}")
        return df
    except Exception as e:
        custom_print(f"Error reading CSV from {blob_name}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run CPU-based ML pipelines (SVM, XGBoost).")
    parser.add_argument("--gcs-bucket-name", required=True, help="GCS bucket name for input data and output results.")
    parser.add_argument("--gcs-prefix", required=True, help="GCS prefix where processed data resides and results will be stored.")
    parser.add_argument("--run-svm", action='store_true', help="Run the SVM pipeline.")
    parser.add_argument("--run-xgboost", action='store_true', help="Run the XGBoost pipeline.")

    args = parser.parse_args()
    gcs_bucket_name = args.gcs_bucket_name
    gcs_prefix = args.gcs_prefix.strip('/')
    run_svm = args.run_svm
    run_xgboost = args.run_xgboost

    if not run_svm and not run_xgboost:
        custom_print("No pipelines selected to run. Use --run-svm or --run-xgboost.")
        sys.exit(0)

    custom_print("--- Starting CPU ML Pipelines ---")
    custom_print(f"Using GCS Bucket: {gcs_bucket_name}")
    custom_print(f"Using GCS Prefix: {gcs_prefix}")

    # Define data paths (using the OHE data)
    data_folder = f"{gcs_prefix}/processed_data/ohe"
    x_train_path = f"{data_folder}/X_train_resampled.csv"
    y_train_path = f"{data_folder}/y_train_resampled.csv"
    x_test_path = f"{data_folder}/X_test.csv"
    y_test_path = f"{data_folder}/y_test.csv"

    # Load data
    try:
        custom_print("\nLoading data for CPU pipelines...")
        X_train = load_csv_from_gcs(gcs_bucket_name, x_train_path)
        y_train = load_csv_from_gcs(gcs_bucket_name, y_train_path).iloc[:, 0] # Load as Series
        X_test = load_csv_from_gcs(gcs_bucket_name, x_test_path)
        y_test = load_csv_from_gcs(gcs_bucket_name, y_test_path).iloc[:, 0] # Load as Series
        custom_print("Data loaded successfully.")
        custom_print(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")
    except Exception as e:
        custom_print(f"FATAL: Failed to load data. Error: {e}")
        sys.exit(1)

    # Define output base paths
    svm_output_prefix = f"{gcs_prefix}/results/svm"
    xgb_output_prefix = f"{gcs_prefix}/results/xgboost"

    # --- Run SVM Pipeline ---
    if run_svm:
        custom_print("\n--- Running SVM Pipeline ---")
        try:
            run_svm_pipeline(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                gcs_bucket_name=gcs_bucket_name,
                gcs_output_prefix=svm_output_prefix,
                custom_print_func=custom_print,
                upload_string_func=upload_string_to_gcs,
                upload_bytes_func=upload_bytes_to_gcs
            )
            custom_print("--- SVM Pipeline Finished ---")
        except Exception as e:
            custom_print(f"ERROR during SVM pipeline execution: {e}")

    # --- Run XGBoost Pipeline ---
    if run_xgboost:
        custom_print("\n--- Running XGBoost Pipeline ---")
        try:
            run_xgboost_pipeline(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                gcs_bucket_name=gcs_bucket_name,
                gcs_output_prefix=xgb_output_prefix,
                custom_print_func=custom_print,
                upload_string_func=upload_string_to_gcs,
                upload_bytes_func=upload_bytes_to_gcs
            )
            custom_print("--- XGBoost Pipeline Finished ---")
        except Exception as e:
            custom_print(f"ERROR during XGBoost pipeline execution: {e}")

    custom_print("\n--- CPU ML Pipelines Script Complete ---")

if __name__ == "__main__":
    main()