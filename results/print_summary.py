# read_training_logs.py
import argparse
import json
import os
import pandas as pd
import fnmatch  # For pattern matching with os.walk
from collections import defaultdict


def get_metric(metrics_dict, key, fallback_keys=None):
    """Safely retrieves a metric, trying primary key then fallback keys."""
    if not isinstance(metrics_dict, dict):
        return "N/A"
    val = metrics_dict.get(key)
    if val is not None and isinstance(val, (float, int)) and val >= 0:  # Ensure it's a valid numeric metric
        return float(val)
    if fallback_keys:
        for fb_key in fallback_keys:
            val = metrics_dict.get(fb_key)
            if val is not None and isinstance(val, (float, int)) and val >= 0:
                return float(val)
    return "N/A"


def get_performance_score(log_data):
    """
    Calculates a performance score for a log entry to determine the "best" model.
    Higher score is better.
    Priority: Test PR-AUC > Test ROC-AUC > Test F1 > Best Val PR-AUC > Best Val F1
    """
    score = -1.0  # Default for models with no comparable metrics

    test_metrics = None
    if "test_set_evaluation_with_best_model" in log_data:
        test_metrics = log_data["test_set_evaluation_with_best_model"]
    elif "test_set_evaluation" in log_data:
        test_metrics = log_data["test_set_evaluation"]
    elif "evaluation_metrics" in log_data:  # For older SVM/XGB logs
        test_metrics = log_data["evaluation_metrics"]

    if test_metrics and isinstance(test_metrics, dict):
        metrics_source = test_metrics.get("metrics", test_metrics)  # Handle nested 'metrics' key
        if isinstance(metrics_source, dict):
            pr_auc = get_metric(metrics_source, "pr_auc")
            if isinstance(pr_auc, float) and pr_auc > 0: return pr_auc * 1000  # Prioritize PR-AUC highly

            roc_auc = get_metric(metrics_source, "roc_auc")
            if isinstance(roc_auc, float) and roc_auc > 0: return roc_auc * 100

            f1 = get_metric(metrics_source, "f1_score", fallback_keys=["f1"])
            if isinstance(f1, float): return f1

    # Fallback to validation metrics if test metrics are not conclusive
    best_val_metrics = log_data.get("best_epoch_validation_metrics")
    if best_val_metrics and isinstance(best_val_metrics, dict):
        val_pr_auc = get_metric(best_val_metrics, "pr_auc")
        if isinstance(val_pr_auc, float) and val_pr_auc > 0: return val_pr_auc * 10  # Lower weight than test

        val_f1 = get_metric(best_val_metrics, "f1_score", fallback_keys=["f1"])
        if isinstance(val_f1, float): return val_f1 * 0.1

    return score


def print_log_summary(log_file_path, log_data):
    """Prints a summary of a single training log."""
    print("=" * 80)
    print(f"BEST MODEL SUMMARY for Model Type: {log_data.get('model_type', 'N/A')}")
    print(f"Log File: {log_file_path}")
    print("-" * 80)

    metadata_source = log_data.get("metadata_source", "N/A")
    print(f"  Metadata Source: {metadata_source}")

    output_gcs_prefix = log_data.get("output_gcs_prefix", "N/A")
    print(f"  GCS Output Prefix: {output_gcs_prefix}")

    saved_model_path = log_data.get("saved_model_path", "N/A")
    print(f"  Saved Model Path: {saved_model_path}")

    training_duration = log_data.get("training_duration_seconds",
                                     log_data.get("training_total_duration_seconds"))
    if training_duration is not None:
        try:
            print(f"  Training Duration: {float(training_duration):.2f} seconds")
        except (ValueError, TypeError):
            print(f"  Training Duration: {training_duration}")

    script_args = log_data.get("script_args")
    if script_args and isinstance(script_args, dict):
        print("  Key Script Arguments:")
        key_args_to_print = ['learning_rate', 'epochs', 'batch_size', 'svm_c', 'svm_kernel',
                             'xgb_n_estimators', 'mlp_hidden_dims', 'ft_d_token', 'ft_n_blocks',
                             'early_stopping_patience']
        for key, value in script_args.items():
            if key in key_args_to_print:
                print(f"    {key}: {value}")

    # Print Test Set Metrics
    evaluation_section = None
    section_name = "N/A"
    if "test_set_evaluation_with_best_model" in log_data:
        evaluation_section = log_data["test_set_evaluation_with_best_model"]
        section_name = "Test Set Evaluation (with Best Model)"
    elif "test_set_evaluation" in log_data:
        evaluation_section = log_data["test_set_evaluation"]
        section_name = "Test Set Evaluation"
    elif "evaluation_metrics" in log_data:
        evaluation_section = log_data["evaluation_metrics"]
        section_name = "Test Set Evaluation"

    if evaluation_section and isinstance(evaluation_section, dict):
        print(f"\n  {section_name}:")
        metrics = evaluation_section.get("metrics", evaluation_section)
        if isinstance(metrics, dict):
            for key in ["accuracy", "precision", "recall", "f1_score", "f1", "roc_auc", "pr_auc"]:
                val = get_metric(metrics, key)
                if val != "N/A":  # Only print if metric was found and valid
                    # Standardize f1_score key for printing
                    print_key = "F1-Score" if key in ["f1_score", "f1"] else key.replace("_", " ").title()
                    print(f"    {print_key}: {val:.4f}" if isinstance(val, float) else f"    {print_key}: {val}")

            cm = metrics.get("confusion_matrix")
            if cm:
                print("    Confusion Matrix:")
                if isinstance(cm, list) and all(isinstance(row, list) for row in cm):
                    for row_idx, row_val in enumerate(cm): print(f"      {row_val}")
                else:
                    print(f"      {cm}")
        else:
            print("    Metrics section (test) not in expected dictionary format.")
    else:
        print("\n  No Test Set Evaluation Metrics found or section is malformed.")

    # Print Best Validation Epoch Metrics (if available)
    best_val_metrics_section = log_data.get("best_epoch_validation_metrics")
    if best_val_metrics_section and isinstance(best_val_metrics_section, dict):
        print("\n  Best Validation Epoch Metrics (during training):")
        # Note: best_epoch_validation_metrics is already the metrics dict itself
        for key in ["accuracy", "precision", "recall", "f1_score", "f1", "roc_auc", "pr_auc"]:
            val = get_metric(best_val_metrics_section, key)
            if val != "N/A":
                print_key = "F1-Score" if key in ["f1_score", "f1"] else key.replace("_", " ").title()
                print(f"    {print_key}: {val:.4f}" if isinstance(val, float) else f"    {print_key}: {val}")

        best_val_loss_val = log_data.get("best_validation_loss", "N/A")
        best_epoch_val = log_data.get("best_epoch_for_early_stopping", "N/A")
        print(f"    Best Validation Loss: {best_val_loss_val:.4f}" if isinstance(best_val_loss_val,
                                                                                 float) else f"    Best Validation Loss: {best_val_loss_val}")
        print(f"    Achieved at Epoch: {best_epoch_val}")

    shap_run = log_data.get("shap_analysis_run", False)  # Default to False if not present
    print(f"\n  SHAP Analysis Run: {shap_run}")
    if shap_run and "shap_top_features" in log_data and log_data["shap_top_features"]:
        print("  Top SHAP Features:")
        try:
            shap_features_data = log_data["shap_top_features"]
            if isinstance(shap_features_data, list) and all(isinstance(item, dict) for item in shap_features_data):
                df_shap = pd.DataFrame(shap_features_data)
                if not df_shap.empty:
                    print(df_shap.to_string(index=False, float_format="%.4f"))
                else:
                    print("    SHAP features data is empty.")
            else:
                print(f"    SHAP features data is not in the expected list of dictionaries format.")
        except Exception as e:
            print(f"    Could not parse or display SHAP features: {e}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Read and summarize training log JSON files from a directory and its subdirectories, showing only the best per model type.")
    parser.add_argument("log_directory", type=str, help="Directory containing the training log files.")
    parser.add_argument("--file-pattern", type=str, default="*_log.json",
                        help="Pattern to match log files (e.g., '*_log.json'). Default is '*_log.json'.")

    args = parser.parse_args()

    log_directory = args.log_directory
    file_pattern = args.file_pattern

    if not os.path.isdir(log_directory):
        print(f"Error: Directory not found: {log_directory}");
        return

    all_log_data = []
    for root, _, files in os.walk(log_directory):
        for filename in files:
            if fnmatch.fnmatch(filename, file_pattern):
                log_file_path = os.path.join(root, filename)
                try:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['__file_path__'] = log_file_path  # Store path for later reference
                        all_log_data.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {log_file_path}. Skipping.")
                except Exception as e:
                    print(f"Warning: Error reading {log_file_path}: {e}. Skipping.")

    if not all_log_data:
        print(f"No valid log files found in '{log_directory}' (and subdirectories) matching pattern '{file_pattern}'.")
        return

    # Group logs by model_type
    grouped_logs = defaultdict(list)
    for log_entry in all_log_data:
        model_type = log_entry.get("model_type", "Unknown_Model_Type")
        grouped_logs[model_type].append(log_entry)

    print(f"Found {len(all_log_data)} total log file(s), grouped into {len(grouped_logs)} model type(s).\n")

    for model_type, logs_for_type in grouped_logs.items():
        if not logs_for_type:
            continue

        best_log_for_type = None
        best_score = -float('inf')  # Initialize with a very small number

        for current_log in logs_for_type:
            current_score = get_performance_score(current_log)
            # Log the score for debugging if needed
            # print(f"Debug: Model {model_type}, File {current_log['__file_path__']}, Score: {current_score}")
            if current_score > best_score:
                best_score = current_score
                best_log_for_type = current_log

        if best_log_for_type:
            print_log_summary(best_log_for_type['__file_path__'], best_log_for_type)
        else:
            print(f"Could not determine best model for type '{model_type}' (no valid performance scores found).\n")


if __name__ == "__main__":
    main()
