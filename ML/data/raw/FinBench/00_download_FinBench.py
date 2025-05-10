from datasets import load_dataset, get_dataset_config_names
import os

# Define the dataset name on Hugging Face
dataset_name = "yuweiyin/FinBench"

# Create a directory to save the CSV files
output_dir = "finbench_csv_data"
os.makedirs(output_dir, exist_ok=True)

# Get the list of available configurations for the dataset
try:
    config_names = get_dataset_config_names(dataset_name)
    print(f"Found configurations: {config_names}")
except Exception as e:
    # Fallback if get_dataset_config_names fails or for datasets without explicit configs
    # For FinBench, we know the configurations from its documentation
    config_names = ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
    print(f"Using predefined configurations: {config_names}")
    print(f"Note: If this list is outdated, please check the dataset card on Hugging Face: https://huggingface.co/datasets/{dataset_name}")


# Loop through each configuration
for config_name in config_names:
    print(f"\nProcessing configuration: {config_name}")
    try:
        # Load the specific configuration of the dataset
        dataset_config = load_dataset(dataset_name, name=config_name)

        # Loop through each split in the configuration (e.g., 'train', 'validation', 'test')
        for split_name in dataset_config.keys():
            print(f"  Processing split: {split_name}")
            dataset_split = dataset_config[split_name]

            # Define the output CSV file path
            csv_file_name = f"{dataset_name.replace('/', '_')}_{config_name}_{split_name}.csv"
            csv_file_path = os.path.join(output_dir, csv_file_name)

            # Save the split to a CSV file
            # The 'X_ml' field in FinBench is a list of floats.
            # Pandas handles this by default, but be aware of how it's represented in the CSV.
            try:
                dataset_split.to_csv(csv_file_path, index=False)
                print(f"    Successfully saved to {csv_file_path}")
            except Exception as e:
                print(f"    Error saving {split_name} of {config_name} to CSV: {e}")

    except Exception as e:
        print(f"  Error loading configuration {config_name}: {e}")
        print(f"  Skipping this configuration.")

print("\nFinished processing all configurations.")