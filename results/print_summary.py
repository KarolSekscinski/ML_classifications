# read_training_logs.py
import argparse
import json
import os
import pandas as pd
import fnmatch  # For pattern matching with os.walk
from collections import defaultdict
import numpy as np  # For checking np.nan


def get_metric(metrics_dict, key, fallback_keys=None):
    """
    Safely retrieves a metric from a dictionary.
    Tries the primary key, then any fallback keys provided.
    Returns the metric value as float if valid, otherwise "N/A".
    """
    if not isinstance(metrics_dict, dict):
        return "N/A"

    # Try primary key
    val = metrics_dict.get(key)
    if val is not None and isinstance(val, (float, int)) and not (
            isinstance(val, float) and np.isnan(val)) and val >= 0:
        return float(val)

    # Try fallback keys
    if fallback_keys:
        for fb_key in fallback_keys:
            val = metrics_dict.get(fb_key)
            if val is not None and isinstance(val, (float, int)) and not (
                    isinstance(val, float) and np.isnan(val)) and val >= 0:
                return float(val)
    return "N/A"


def get_performance_score(log_data):
    """
    Determines the performance score of a model based on PR AUC from the final test set evaluation.
    Higher PR AUC is better.
    """
    pr_auc_score = -1.0  # Default for models where PR AUC is not found or not applicable

    # Path 1: For tuned PyTorch models (MLP, FT-Transformer) from my modified scripts
    # final_model_test_set_evaluation: {"loss": ..., "metrics": {"pr_auc": ...}}
    eval_data = log_data.get("final_model_test_set_evaluation")
    if isinstance(eval_data, dict):
        metrics_data = eval_data.get("metrics", eval_data)  # Handles if "metrics" is nested or not
        if isinstance(metrics_data, dict):
            pr_auc = get_metric(metrics_data, "pr_auc")
            if isinstance(pr_auc, float):
                return pr_auc

    # Path 2: For tuned Scikit-learn/XGBoost models from my modified scripts
    # final_model_evaluation_metrics: {"pr_auc": ...}
    eval_data_sklearn = log_data.get("final_model_evaluation_metrics")
    if isinstance(eval_data_sklearn, dict):
        pr_auc = get_metric(eval_data_sklearn, "pr_auc")
        if isinstance(pr_auc, float):
            return pr_auc

    # Path 3: For original script structure (if still encountered)
    # evaluation_metrics: {"pr_auc": ...} or evaluation_metrics: {"metrics": {"pr_auc": ...}}
    eval_data_orig = log_data.get("evaluation_metrics")
    if isinstance(eval_data_orig, dict):
        metrics_data_orig = eval_data_orig.get("metrics", eval_data_orig)
        if isinstance(metrics_data_orig, dict):
            pr_auc = get_metric(metrics_data_orig, "pr_auc")
            if isinstance(pr_auc, float):
                return pr_auc

    return pr_auc_score


def print_log_summary(log_file_path, log_data):
    """Prints a summary of a single training log, adapted for tuned model outputs."""
    print("=" * 80)
    model_type_name = log_data.get('model_type', 'N/A')
    # Clean up model type name if it contains "Tuned" from example
    if "Tuned" in model_type_name and "FinBench" in model_type_name:  # Specific to example
        model_type_name = model_type_name.replace("FinBench_", "").replace("_Tuned", "")

    print(f"NAJLEPSZY MODEL (wg PR AUC na teście) dla typu: {model_type_name}")
    print(f"Plik logu: {log_file_path}")
    print("-" * 80)

    metadata_source = log_data.get("metadata_source", "N/A")
    print(f"  Źródło metadanych: {metadata_source}")

    output_gcs_prefix = log_data.get("output_gcs_prefix", "N/A")
    print(f"  Prefiks wyjściowy GCS: {output_gcs_prefix}")

    saved_model_path = log_data.get("saved_model_path", "N/A")
    print(f"  Ścieżka zapisanego modelu: {saved_model_path}")

    # Training Duration
    training_duration = log_data.get("final_model_training_total_duration_seconds",  # From example
                                     log_data.get("final_model_training_duration_seconds",  # From my scripts
                                                  log_data.get("training_duration_seconds")))  # Fallback
    if training_duration is not None:
        try:
            print(f"  Czas trenowania finalnego modelu: {float(training_duration):.2f} sekund")
        except (ValueError, TypeError):
            print(f"  Czas trenowania finalnego modelu: {training_duration}")

    # Tuning Information
    print("\n  Informacje o strojeniu (Optuna):")
    tuning_args = log_data.get("tuning_args", {})
    n_trials = tuning_args.get("n_trials", log_data.get("optuna_n_trials", "N/A"))
    optimized_metric_name = tuning_args.get("optimization_metric")

    best_tuned_value = log_data.get("best_tuned_metric_value")
    if best_tuned_value is None:  # Try example's specific key
        if "optuna_best_trial_metric_value_pr_auc" in log_data:
            best_tuned_value = log_data.get("optuna_best_trial_metric_value_pr_auc")
            if not optimized_metric_name: optimized_metric_name = "pr_auc"  # Infer from key
        elif "best_trial_metric_value" in log_data:  # Generic fallback
            best_tuned_value = log_data.get("best_trial_metric_value")

    print(f"    Liczba prób (trials): {n_trials}")
    if optimized_metric_name:
        print(f"    Optymalizowana metryka (podczas strojenia): {optimized_metric_name.upper()}")
    if best_tuned_value is not None:
        try:
            print(f"    Najlepsza wartość metryki (podczas strojenia): {float(best_tuned_value):.4f}")
        except (ValueError, TypeError):
            print(f"    Najlepsza wartość metryki (podczas strojenia): {best_tuned_value}")

    # Best Hyperparameters
    best_hyperparams = log_data.get("best_hyperparameters")
    if best_hyperparams and isinstance(best_hyperparams, dict):
        print("\n  Najlepsze znalezione hiperparametry:")
        for key, value in best_hyperparams.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")

    # Script Arguments (original call arguments, for context)
    script_args = log_data.get("script_args")
    if script_args and isinstance(script_args, dict):
        print("\n  Argumenty skryptu (konfiguracja uruchomienia):")
        # Print a few key non-tuned args or general config
        key_args_to_print = ['epochs', 'num_workers', 'early_stopping_patience',
                             'n_trials', 'optimization_metric', 'batch_size_default',
                             'svm_max_iter', 'xgb_early_stopping_rounds']  # Add more fixed args if needed
        printed_script_args = False
        for key, value in script_args.items():
            if key in key_args_to_print:  # Print only selected fixed/config args
                print(f"    {key}: {value}")
                printed_script_args = True
        if not printed_script_args:
            print("    (Brak wybranych argumentów do wyświetlenia lub wszystkie były dynamiczne)")

    # Final Model Test Set Metrics
    test_metrics_data = None
    section_name = "Ewaluacja finalnego modelu na zbiorze testowym"

    # Try paths for metrics from tuned scripts
    if "final_model_test_set_evaluation" in log_data:
        test_metrics_container = log_data["final_model_test_set_evaluation"]
        if isinstance(test_metrics_container, dict):
            test_metrics_data = test_metrics_container.get("metrics", test_metrics_container)
    elif "final_model_evaluation_metrics" in log_data:  # For tuned sklearn/XGB
        test_metrics_data = log_data["final_model_evaluation_metrics"]
    elif "evaluation_metrics" in log_data:  # Fallback to older structure
        test_metrics_container = log_data["evaluation_metrics"]
        if isinstance(test_metrics_container, dict):
            test_metrics_data = test_metrics_container.get("metrics", test_metrics_container)

    if isinstance(test_metrics_data, dict):
        print(f"\n  {section_name}:")
        # Define preferred order and names for display
        metric_keys_display = {
            "pr_auc": "PR AUC",
            "roc_auc": "ROC AUC",
            "f1": "F1-Score",
            "f1_score": "F1-Score",  # Fallback key
            "recall": "Recall (Czułość)",
            "precision": "Precision (Precyzja)",
            "accuracy": "Accuracy (Dokładność)",
        }
        for key, display_name in metric_keys_display.items():
            val = get_metric(test_metrics_data, key)  # Pass the actual metrics dict
            if val != "N/A":
                print(f"    {display_name}: {val:.4f}")

        cm = test_metrics_data.get("confusion_matrix")
        if cm:
            print("    Macierz Pomyłek (Confusion Matrix):")
            if isinstance(cm, list) and all(isinstance(row, list) for row in cm):
                for row_val in cm: print(f"      {row_val}")
            else:
                print(f"      {cm}")
    else:
        print(f"\n  {section_name}: Brak danych metryk lub niepoprawny format.")

    # Information about validation during FINAL model training (if present, like in example)
    best_val_metrics_final_train = log_data.get("final_model_best_epoch_validation_metrics")
    if isinstance(best_val_metrics_final_train, dict):
        print("\n  Najlepsze metryki walidacyjne (podczas treningu finalnego modelu):")
        best_epoch_final_train = log_data.get("final_model_best_epoch_for_early_stopping", "N/A")
        best_val_loss_final_train = log_data.get("final_model_best_validation_loss", "N/A")

        print(f"    Osiągnięte w epoce: {best_epoch_final_train}")
        if isinstance(best_val_loss_final_train, float):
            print(f"    Najlepsza strata walidacyjna: {best_val_loss_final_train:.4f}")
        else:
            print(f"    Najlepsza strata walidacyjna: {best_val_loss_final_train}")

        for key, display_name in metric_keys_display.items():  # Reuse display map
            val = get_metric(best_val_metrics_final_train, key)
            if val != "N/A":
                print(f"    {display_name} (walidacja): {val:.4f}")
        cm_val = best_val_metrics_final_train.get("confusion_matrix")
        if cm_val:
            print("    Macierz Pomyłek (walidacja):")
            for row_val in cm_val: print(f"      {row_val}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Czyta i podsumowuje pliki JSON z logami treningowymi, wyświetlając najlepszy model dla każdego typu wg PR AUC na zbiorze testowym.")
    parser.add_argument("log_directory", type=str, help="Katalog zawierający pliki logów treningowych.")
    parser.add_argument("--file-pattern", type=str, default="*_log.json",
                        help="Wzorzec do dopasowania plików logów (np. '*_log.json', '*_final_training_log.json'). Domyślnie: '*_log.json'.")

    args = parser.parse_args()

    log_directory = args.log_directory
    file_pattern = args.file_pattern

    if not os.path.isdir(log_directory):
        print(f"Błąd: Katalog nie znaleziony: {log_directory}");
        return

    all_log_data = []
    for root, _, files in os.walk(log_directory):
        for filename in files:
            if fnmatch.fnmatch(filename, file_pattern) or fnmatch.fnmatch(filename,
                                                                          "*_final_training_log.json"):  # Also match new specific names
                log_file_path = os.path.join(root, filename)
                try:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['__file_path__'] = log_file_path
                        all_log_data.append(data)
                except json.JSONDecodeError:
                    print(f"Ostrzeżenie: Nie można zdekodować JSON z {log_file_path}. Pomijanie.")
                except Exception as e:
                    print(f"Ostrzeżenie: Błąd podczas czytania {log_file_path}: {e}. Pomijanie.")

    if not all_log_data:
        print(
            f"Nie znaleziono pasujących plików logów w '{log_directory}' (i podkatalogach) dla wzorca '{file_pattern}'.")
        return

    grouped_logs = defaultdict(list)
    for log_entry in all_log_data:
        model_type = log_entry.get("model_type", "Unknown_Model_Type")
        # Normalize model type slightly for grouping
        if "Tuned" in model_type and "FinBench" in model_type:  # Specific to example
            model_type_key = model_type.replace("FinBench_", "").replace("_Tuned", "")
        elif "_Tuned" in model_type:
            model_type_key = model_type.replace("_Tuned", "")
        elif " (Best)" in model_type:
            model_type_key = model_type.replace(" (Best)", "")
        elif " (Calibrated)" in model_type:  # for SVM
            model_type_key = model_type.replace(" (Calibrated)", "")
        else:
            model_type_key = model_type
        grouped_logs[model_type_key].append(log_entry)

    print(f"Znaleziono {len(all_log_data)} plików logów, pogrupowanych w {len(grouped_logs)} typów modeli.\n")

    for model_type_key, logs_for_type in grouped_logs.items():
        if not logs_for_type:
            continue

        best_log_for_type = None
        # Initialize with a very small number, as PR AUC is >= 0
        best_pr_auc = -1.0

        for current_log in logs_for_type:
            current_pr_auc = get_performance_score(current_log)

            # Debugging line if needed:
            # print(f"Debug: Model {model_type_key}, Plik {current_log['__file_path__']}, PR AUC: {current_pr_auc}")

            if isinstance(current_pr_auc, float) and current_pr_auc > best_pr_auc:
                best_pr_auc = current_pr_auc
                best_log_for_type = current_log
            elif best_log_for_type is None and isinstance(current_pr_auc,
                                                          float) and current_pr_auc >= 0:  # Handle first valid log
                best_pr_auc = current_pr_auc
                best_log_for_type = current_log

        if best_log_for_type:
            print_log_summary(best_log_for_type['__file_path__'], best_log_for_type)
        else:
            print(
                f"Nie można było ustalić najlepszego modelu dla typu '{model_type_key}' (brak poprawnej metryki PR AUC na zbiorze testowym).\n")


if __name__ == "__main__":
    main()