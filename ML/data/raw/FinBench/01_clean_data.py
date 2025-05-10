import pandas as pd
import re
import numpy as np  # Used for safely creating NaN if needed
import os  # Added for directory operations
from io import StringIO  # Keep for potential testing with string data if needed


# --- Core Cleaning Functions (from previous response, unchanged) ---
def clean_array_like_string_to_semicolon_separated(s):
    """
    Cleans strings like "[ 1 2 3 ]" or "['a' 'b']" to "1;2;3" or "a;b".
    Handles numbers and quoted strings.
    """
    if pd.isna(s):
        return s
    s = str(s).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()

    items = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)  # Try numbers first
    if not items:
        items = re.findall(r"'([^']*)'", s)  # Then quoted strings
    if not items:
        if s:  # Fallback for space-separated non-quoted items
            items = [item for item in re.split(r'\s+', s) if item]
        else:
            return ""
    return ';'.join(items)


def clean_cat_str(s):
    """
    Cleans the 'cat_str' column.
    From: "[array(['car', 'home'], dtype=object)\n array(['female', 'male'], dtype=object)...]"
    To: "car,home;female,male;..."
    """
    if pd.isna(s):
        return s

    array_contents = re.findall(r"array\(\s*\[(.*?)\][^]]*?\)", str(s), re.DOTALL)

    processed_arrays = []
    for content in array_contents:
        items = re.findall(r"'([^']*)'", content)
        processed_arrays.append(','.join(items))

    return ';'.join(processed_arrays)


def clean_text_newlines(s):
    """Replaces newlines with spaces in text columns."""
    if pd.isna(s) or not isinstance(s, str):
        return s
    return s.replace('\n', ' ').replace('\r', '')


# --- Main CSV Processing Function (logic is same) ---
def process_single_csv_file(input_filepath, output_filepath):
    print(f"Processing {input_filepath}...")
    try:
        df = pd.read_csv(input_filepath)
    except Exception as e:
        print(f"  Error reading {input_filepath}: {e}")
        return

    # Clean columns that become semicolon-separated strings
    for col_name in ['num_idx', 'cat_idx', 'cat_dim', 'col_name']:
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(clean_array_like_string_to_semicolon_separated)

    if 'cat_str' in df.columns:
        df['cat_str'] = df['cat_str'].apply(clean_cat_str)

    # Clean text columns (replace newlines)
    for col_name in ['X_instruction_for_profile', 'X_profile']:
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(clean_text_newlines)

    # Expand X_ml and X_ml_unscale
    max_features = 0
    if 'num_features' in df.columns and not df['num_features'].empty and df['num_features'].notna().all():
        try:
            max_features = int(df['num_features'].max())
        except ValueError:
            print(
                f"  Warning: Could not convert 'num_features' to int for max value in {input_filepath}. Skipping X_ml/X_ml_unscale expansion.")
            max_features = 0
    else:
        if 'X_ml' in df.columns and not df['X_ml'].empty:
            print(
                f"  'num_features' column not reliable in {input_filepath}. Attempting to infer max_features from X_ml.")
            try:
                first_valid_xml_str = df['X_ml'].dropna().iloc[0] if not df['X_ml'].dropna().empty else None
                if first_valid_xml_str:
                    example_X_ml_str = first_valid_xml_str
                    example_X_ml_cleaned = clean_array_like_string_to_semicolon_separated(example_X_ml_str)
                    if example_X_ml_cleaned:
                        example_X_ml = [float(x) for x in example_X_ml_cleaned.split(';') if x.strip()]
                        max_features = len(example_X_ml)
                        print(f"  Inferred max_features as {max_features} from X_ml in {input_filepath}.")
                    else:
                        print(
                            f"  Warning: Could not parse example X_ml for inferring features in {input_filepath}. Skipping expansion.")
                        max_features = 0
                else:
                    print(f"  Warning: X_ml column is empty or all NaN in {input_filepath}. Skipping expansion.")
                    max_features = 0
            except Exception as e:
                print(
                    f"  Warning: Could not infer max_features from X_ml in {input_filepath} due to {e}. Skipping expansion.")
                max_features = 0

    if max_features > 0:
        for col_prefix in ['X_ml', 'X_ml_unscale']:
            if col_prefix not in df.columns:
                continue

            new_cols_list = []

            for index, row in df.iterrows():
                row_new_cols = {}
                raw_val_str = row[col_prefix]

                if pd.isna(raw_val_str):
                    cleaned_val_str = ""
                else:
                    cleaned_val_str = clean_array_like_string_to_semicolon_separated(str(raw_val_str))

                values = []
                if cleaned_val_str:
                    try:
                        values = [float(x) for x in cleaned_val_str.split(';') if x.strip()]
                    except ValueError:
                        print(
                            f"  Warning: Row {index}: Could not parse float array for {col_prefix} in {input_filepath}: '{cleaned_val_str}'. Using NaNs for this row's {col_prefix} features.")
                        num_feats_for_row = int(row['num_features']) if 'num_features' in row and pd.notna(
                            row['num_features']) else max_features
                        values = [np.nan] * num_feats_for_row

                num_actual_features_for_row = int(row['num_features']) if 'num_features' in row and pd.notna(
                    row['num_features']) else len(values)

                for i in range(max_features):
                    col_name_feat = f'{col_prefix}_{i}'
                    if i < len(values):
                        row_new_cols[col_name_feat] = values[i]
                    elif i < num_actual_features_for_row:
                        row_new_cols[col_name_feat] = np.nan
                    else:
                        row_new_cols[col_name_feat] = np.nan
                new_cols_list.append(row_new_cols)

            if new_cols_list:
                new_cols_df = pd.DataFrame(new_cols_list, index=df.index)
                df = pd.concat([df, new_cols_df], axis=1)
            df = df.drop(columns=[col_prefix])
    elif 'X_ml' in df.columns or 'X_ml_unscale' in df.columns:
        print(
            f"  'num_features' is 0 or could not be determined reliably for {input_filepath}. Skipping expansion of X_ml/X_ml_unscale.")

    try:
        # Ensure the output directory for this specific file exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        df.to_csv(output_filepath, index=False)
        print(f"  Successfully saved cleaned data to {output_filepath}")
    except Exception as e:
        print(f"  Error writing {output_filepath}: {e}")


# --- Main script execution to process all CSVs in a directory ---
if __name__ == "__main__":
    input_directory = "finbench_csv_data"
    output_directory = "../../processed/FinBench"  # Define the output directory

    if not os.path.isdir(input_directory):
        print(
            f"Error: Input directory '{input_directory}' not found. Please ensure your CSV files are in this directory.")
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        print(f"Starting CSV cleaning process for directory: {input_directory}")
        print(f"Cleaned files will be saved in: {output_directory}")

        processed_count = 0
        for filename in os.listdir(input_directory):
            if filename.endswith(".csv") and not filename.startswith(
                    "cleaned_"):  # Avoid processing original "cleaned_" files if any
                input_filepath = os.path.join(input_directory, filename)

                # Output filename will still be "cleaned_" + original, but in the new directory
                output_filename = "cleaned_" + filename
                output_filepath = os.path.join(output_directory, output_filename)

                process_single_csv_file(input_filepath, output_filepath)
                processed_count += 1
            elif filename.endswith(".csv") and filename.startswith("cleaned_"):
                print(
                    f"Skipping file that already appears to be cleaned (starts with 'cleaned_'): {os.path.join(input_directory, filename)}")

        if processed_count == 0:
            print(
                f"No new CSV files found to process in '{input_directory}'. Check if files end with '.csv' and don't start with 'cleaned_'.")
        else:
            print(f"\nFinished processing {processed_count} CSV file(s) from '{input_directory}'.")
            print(f"Cleaned files are located in '{output_directory}'.")