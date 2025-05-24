# gcs/gcs_utils.py
import logging
from google.cloud import storage
from io import BytesIO, StringIO
import json
import pickle # Added for saving/loading general objects if needed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_gcs_client():
    """Initializes and returns a GCS client."""
    try:
        client = storage.Client()
        return client
    except Exception as e:
        logging.error(f"Failed to initialize GCS client: {e}")
        raise

def download_blob_to_memory(bucket_name, source_blob_name):
    """Downloads a blob from GCS into memory as bytes."""
    client = _get_gcs_client()
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        if not blob.exists():
            logging.warning(f"Blob {source_blob_name} does not exist in bucket {bucket_name}.")
            return None
        logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to memory.")
        return blob.download_as_bytes()
    except Exception as e:
        logging.error(f"Error downloading blob gs://{bucket_name}/{source_blob_name}: {e}")
        raise

def download_blob_as_string(bucket_name, source_blob_name, encoding='utf-8'):
    """Downloads a blob from GCS into memory as a string."""
    try:
        file_bytes = download_blob_to_memory(bucket_name, source_blob_name)
        if file_bytes:
            return file_bytes.decode(encoding)
        return None
    except Exception as e:
        logging.error(f"Error decoding blob gs://{bucket_name}/{source_blob_name} to string: {e}")
        raise

def upload_bytes_to_gcs(bucket_name, data_bytes, destination_blob_name, content_type='application/octet-stream'):
    """Uploads bytes to GCS."""
    client = _get_gcs_client()
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(data_bytes, content_type=content_type)
        logging.info(f"Bytes successfully uploaded to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        logging.error(f"Error uploading bytes to gs://{bucket_name}/{destination_blob_name}: {e}")
        raise

def upload_string_to_gcs(bucket_name, data_string, destination_blob_name, content_type='text/plain', encoding='utf-8'):
    """Uploads a string to GCS."""
    try:
        data_bytes = data_string.encode(encoding)
        upload_bytes_to_gcs(bucket_name, data_bytes, destination_blob_name, content_type=content_type)
    except Exception as e:
        logging.error(f"Error encoding string for upload to gs://{bucket_name}/{destination_blob_name}: {e}")
        # No re-raise here, upload_bytes_to_gcs will handle its own errors and logging

def save_json_to_gcs(data_dict, bucket_name, destination_blob_name):
    """Saves a dictionary as a JSON file to GCS."""
    try:
        json_string = json.dumps(data_dict, indent=4, default=str) # default=str for non-serializable types
        upload_string_to_gcs(bucket_name, json_string, destination_blob_name, content_type='application/json')
    except Exception as e:
        logging.error(f"Error serializing dictionary to JSON for GCS upload ({destination_blob_name}): {e}")
        raise

def load_json_from_gcs(bucket_name, source_blob_name):
    """Loads a JSON file from GCS and returns it as a dictionary."""
    try:
        json_string = download_blob_as_string(bucket_name, source_blob_name)
        if json_string:
            return json.loads(json_string)
        return None
    except Exception as e:
        logging.error(f"Error loading or parsing JSON from gs://{bucket_name}/{source_blob_name}: {e}")
        raise

def save_pickle_to_gcs(obj, bucket_name, destination_blob_name):
    """Saves a Python object as a pickle file to GCS."""
    try:
        pickle_bytes = pickle.dumps(obj)
        upload_bytes_to_gcs(bucket_name, pickle_bytes, destination_blob_name, content_type='application/octet-stream')
    except Exception as e:
        logging.error(f"Error pickling object for GCS upload ({destination_blob_name}): {e}")
        raise

def load_pickle_from_gcs(bucket_name, source_blob_name):
    """Loads a pickle file from GCS and returns the Python object."""
    try:
        pickle_bytes = download_blob_to_memory(bucket_name, source_blob_name)
        if pickle_bytes:
            return pickle.loads(pickle_bytes)
        return None
    except Exception as e:
        logging.error(f"Error loading or unpickling object from gs://{bucket_name}/{source_blob_name}: {e}")
        raise