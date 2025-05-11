# Import custom modules
import gcs_utils
import svm_pipeline
import xgboost_pipeline
import mlp_pipeline
import preprocessing_module
import ft_transformer_pipeline

# --- Configuration ---
GCS_BUCKET_NAME_MAIN = "licencjat_ml_classification"  # Ensure this matches gcs_utils or pass it
gcs_output_prefix = gcs_utils.get_gcs_output_prefix()  # Generate unique prefix for this run
data_source = "gs://" + GCS_BUCKET_NAME_MAIN + "NeurIPS/Base.csv"

X_train_resampled, y_train_resampled, X_test_processed, y_test, X_train, processed_feature_names, cat_cardinalities, numerical_features, categorical_features = (
    preprocessing_module.
    load_and_preprocess_data(
        data_source, GCS_BUCKET_NAME_MAIN, gcs_output_prefix,
        gcs_utils.custom_print, gcs_utils.save_plot_to_gcs, gcs_utils.upload_bytes_to_gcs
    ))

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

ft_transformer_pipeline.run_ft_transformer_pipeline(
    X_train_resampled, y_train_resampled, X_test_processed, y_test,
    processed_feature_names,
    numerical_features, categorical_features, X_train=X_train,
    gcs_bucket_name=GCS_BUCKET_NAME_MAIN, gcs_output_prefix=gcs_output_prefix,
    custom_print_func=gcs_utils.custom_print, save_plot_func=gcs_utils.save_plot_to_gcs,
    upload_bytes_func=gcs_utils.upload_bytes_to_gcs
)

# --- Finalize Logging ---
gcs_utils.custom_print("\n--- Main Fraud Detection Pipeline Complete ---")
gcs_utils.finalize_logs(GCS_BUCKET_NAME_MAIN, gcs_output_prefix)

gcs_utils.custom_print("Script finished. All logs, plots, and models attempted to be saved to GCS.")
