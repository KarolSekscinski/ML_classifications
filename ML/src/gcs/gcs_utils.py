import time
from io import BytesIO
from google.cloud import storage
import matplotlib.pyplot as plt

# --- Configuration ---
# This can be set here or passed as arguments if more flexibility is needed.
GCS_BUCKET_NAME = "licencjat_ml_classification"  # <--- REPLACE with your GCS bucket name

# --- Logging ---
LOG_MESSAGES = []


def custom_print(message):
    """Prints to console and appends to global log list."""
    print(message)
    LOG_MESSAGES.append(str(message))


# --- GCS Helper Functions ---
def upload_string_to_gcs(bucket_name, string_content, destination_blob_name, content_type='text/plain'):
    """Uploads a string to a GCS bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(string_content, content_type=content_type)
        custom_print(f"Successfully uploaded string to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        custom_print(f"ERROR uploading string to GCS: {e}")


def upload_bytes_to_gcs(bucket_name, bytes_content, destination_blob_name, content_type='application/octet-stream'):
    """Uploads bytes content (e.g., image, pickled model) to a GCS bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(bytes_content, content_type=content_type)
        custom_print(f"Successfully uploaded bytes to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        custom_print(f"ERROR uploading bytes to GCS: {e}")


def save_plot_to_gcs(plt_figure, bucket_name, gcs_output_prefix, blob_name_suffix):
    """Saves the current matplotlib figure to GCS as PNG."""
    try:
        img_data = BytesIO()
        # Ensure figure is not empty if it was created outside
        if not plt_figure.axes:  # A simple check if the figure has any axes
            custom_print(f"Warning: Plot figure for {blob_name_suffix} appears to be empty. Skipping GCS upload.")
            plt.close(plt_figure)
            return

        plt_figure.savefig(img_data, format='png', bbox_inches='tight')
        img_data.seek(0)
        destination_blob_name = f"{gcs_output_prefix}/plots/{blob_name_suffix}.png"
        upload_bytes_to_gcs(bucket_name, img_data.getvalue(), destination_blob_name, content_type='image/png')
        # plt.close(plt_figure) # Close the figure to free memory - caller should manage this if fig is reused
    except Exception as e:
        custom_print(f"ERROR saving plot {blob_name_suffix} to GCS: {e}")
    finally:
        if plt_figure:  # Always try to close to prevent memory leaks
            plt.close(plt_figure)


def get_gcs_output_prefix():
    """Generates a unique GCS output prefix for the run."""
    return "fraud_detection_run_" + time.strftime("%Y%m%d_%H%M%S")


def finalize_logs(bucket_name, gcs_output_prefix):
    """Uploads the accumulated logs to GCS at the end of the script run."""
    custom_print("Finalizing logs and uploading to GCS...")
    log_file_content = "\n".join(LOG_MESSAGES)
    upload_string_to_gcs(bucket_name, log_file_content, f"{gcs_output_prefix}/run_log.txt")
    custom_print("Log upload complete.")
