# gcs_utils.py
import logging
from google.cloud import storage
from google.api_core import exceptions as google_exceptions # For specific exceptions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize storage client globally or within functions as needed.
# Global initialization is fine for scripts if credentials are set up environment-wide.
try:
    storage_client = storage.Client()
    logging.info("Google Cloud Storage client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Google Cloud Storage client: {e}")
    # Depending on how critical GCS is, you might raise an error here or proceed cautiously.
    storage_client = None # Set to None to check before use in functions

def _get_client():
    """Helper to get the storage client, raising an error if not initialized."""
    if storage_client is None:
        logging.error("Storage client is not initialized. Ensure credentials are set up.")
        raise ConnectionError("Storage client not initialized.")
    return storage_client

def upload_bytes_to_gcs(bucket_name, source_bytes, destination_blob_name, content_type='application/octet-stream'):
    """Uploads bytes to a GCS bucket."""
    client = _get_client()
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        logging.info(f"Uploading bytes to gs://{bucket_name}/{destination_blob_name}...")
        blob.upload_from_string(source_bytes, content_type=content_type)
        logging.info(f"Bytes successfully uploaded to gs://{bucket_name}/{destination_blob_name}")
        return True
    except google_exceptions.NotFound:
        logging.error(f"Error: Bucket '{bucket_name}' not found.")
        return False
    except Exception as e:
        logging.error(f"Failed to upload bytes to gs://{bucket_name}/{destination_blob_name}: {e}")
        return False

def upload_string_to_gcs(bucket_name, source_string, destination_blob_name, content_type='text/plain'):
    """Uploads a string to a GCS bucket."""
    client = _get_client()
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        logging.info(f"Uploading string to gs://{bucket_name}/{destination_blob_name}...")
        blob.upload_from_string(source_string, content_type=content_type)
        logging.info(f"String successfully uploaded to gs://{bucket_name}/{destination_blob_name}")
        return True
    except google_exceptions.NotFound:
        logging.error(f"Error: Bucket '{bucket_name}' not found.")
        return False
    except Exception as e:
        logging.error(f"Failed to upload string to gs://{bucket_name}/{destination_blob_name}: {e}")
        return False

def download_blob_to_memory(bucket_name, source_blob_name):
    """Downloads a blob from GCS into memory as bytes."""
    client = _get_client()
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to memory...")
        blob_bytes = blob.download_as_bytes()
        logging.info(f"Successfully downloaded gs://{bucket_name}/{source_blob_name} ({len(blob_bytes)} bytes).")
        return blob_bytes
    except google_exceptions.NotFound:
        logging.error(f"Error: Blob '{source_blob_name}' not found in bucket '{bucket_name}'.")
        return None
    except Exception as e:
        logging.error(f"Failed to download blob gs://{bucket_name}/{source_blob_name}: {e}")
        return None

def download_blob_as_string(bucket_name, source_blob_name, encoding='utf-8'):
    """Downloads a blob from GCS into memory as a string."""
    client = _get_client()
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} as string...")
        blob_string = blob.download_as_text(encoding=encoding)
        logging.info(f"Successfully downloaded gs://{bucket_name}/{source_blob_name} as string.")
        return blob_string
    except google_exceptions.NotFound:
        logging.error(f"Error: Blob '{source_blob_name}' not found in bucket '{bucket_name}'.")
        return None
    except Exception as e:
        logging.error(f"Failed to download blob gs://{bucket_name}/{source_blob_name} as string: {e}")
        return None

def list_blobs(bucket_name, prefix=None, delimiter=None):
    """Lists blobs in a given GCS bucket, optionally filtering by prefix."""
    client = _get_client()
    try:
        blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
        blob_names = [blob.name for blob in blobs]
        logging.info(f"Found {len(blob_names)} blobs in gs://{bucket_name}/{prefix or ''}")
        return blob_names
    except google_exceptions.NotFound:
        logging.error(f"Error: Bucket '{bucket_name}' not found.")
        return []
    except Exception as e:
        logging.error(f"Failed to list blobs in gs://{bucket_name}/{prefix or ''}: {e}")
        return []

# Example usage (optional, for testing the module directly)
if __name__ == '__main__':
    # Replace with your actual bucket name and desired paths for testing
    TEST_BUCKET = 'your-gcs-bucket-name' # <--- CHANGE THIS
    TEST_PREFIX = 'gcs_utils_test'

    if TEST_BUCKET == 'your-gcs-bucket-name':
        print("Please update TEST_BUCKET in gcs_utils.py before running the example.")
    elif storage_client: # Proceed only if client initialized
        print(f"--- Testing GCS Utils with bucket: {TEST_BUCKET} ---")

        # Test string upload
        print("\nTesting string upload...")
        test_str = "Hello GCS from gcs_utils!"
        str_blob = f"{TEST_PREFIX}/test_string.txt"
        upload_string_to_gcs(TEST_BUCKET, test_str, str_blob)

        # Test bytes upload (e.g., pickled object)
        print("\nTesting bytes upload...")
        import pickle
        test_obj = {'a': 1, 'b': [2, 3]}
        obj_bytes = pickle.dumps(test_obj)
        bytes_blob = f"{TEST_PREFIX}/test_object.pkl"
        upload_bytes_to_gcs(TEST_BUCKET, obj_bytes, bytes_blob)

        # Test listing blobs
        print("\nTesting list blobs...")
        blobs = list_blobs(TEST_BUCKET, prefix=TEST_PREFIX)
        print("Blobs found:")
        for b in blobs:
            print(f" - {b}")

        # Test string download
        print("\nTesting string download...")
        downloaded_str = download_blob_as_string(TEST_BUCKET, str_blob)
        if downloaded_str:
            print(f"Downloaded string: '{downloaded_str}'")
            assert downloaded_str == test_str

        # Test bytes download
        print("\nTesting bytes download...")
        downloaded_bytes = download_blob_to_memory(TEST_BUCKET, bytes_blob)
        if downloaded_bytes:
            downloaded_obj = pickle.loads(downloaded_bytes)
            print(f"Downloaded and unpickled object: {downloaded_obj}")
            assert downloaded_obj == test_obj

        print("\n--- GCS Utils Test Complete ---")
        print(f"Note: Test files created under gs://{TEST_BUCKET}/{TEST_PREFIX}/")