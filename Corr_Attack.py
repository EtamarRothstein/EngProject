from evaluation_funcs import get_next_step_toward_original
import pandas as pd
import numpy as np
import random
from evaluation_funcs import evaluate_DT
from correlation_analysis import correlation_dict
import json

# with open('correlation_dict.json', 'r') as f:
#     correlation_dict = json.load(f)


number_of_evaluations = 0

# Load dataset of datapoints the model predicts correctly
df = pd.read_csv('Datasets/correct_predictions_DT.csv')

label_column = 'income'


# Select a single random data point
original_datapoint = df.sample(n=1).copy()
adversarial_datapoint = original_datapoint.copy()

true_label = original_datapoint[label_column].iloc[0]

# All features except the label
all_features = [f for f in df.columns if f != label_column]

# --- Adversarial perturbation phase (bounded by correlation list) ---
n = 0
max_attempts = 1000

while evaluate_DT(adversarial_datapoint) == true_label and n < max_attempts:
    n += 1
    number_of_evaluations += 1
    for feature in all_features:
        original_value = original_datapoint[feature].iloc[0]
        ordered_vals = list(correlation_dict[feature].keys())

        if original_value in ordered_vals:
            original_index = ordered_vals.index(original_value)
            lower_bound = max(0, original_index - 2)
            upper_bound = min(len(ordered_vals) - 1, original_index + 2)

            # Create a sublist of allowed values within the bounds
            bounded_vals = ordered_vals[lower_bound : upper_bound + 1]

            # Ensure we don't just pick the original value again
            if len(bounded_vals) > 1 and original_value in bounded_vals:
                bounded_vals.remove(original_value)

            if bounded_vals:
                adversarial_datapoint.at[adversarial_datapoint.index[0], feature] = random.choice(bounded_vals)
            else:
                # If the original value is the only one within the bounds (e.g., list is short),
                # we might not be able to perturb. In this case, we can either skip this feature
                # or implement a different strategy (e.g., wider bounds). For now, we skip.
                continue
        else:
            # If the original value is not in the correlation dictionary (which shouldn't happen
            # if the dictionary was built from the same data), we fall back to random choice.
            possible_vals = list(correlation_dict[feature].keys())
            adversarial_datapoint.at[adversarial_datapoint.index[0], feature] = random.choice(possible_vals)

# --- Refinement: move back toward original via correlation steps ---
if evaluate_DT(adversarial_datapoint) != true_label:
    print("\nInitial adversarial example found!")
    print("Original Data Point:\n", original_datapoint.T)
    print("\nInitial Adversarial Data Point (bounded perturbation):\n", adversarial_datapoint.T)

    at_boundary = {feature: False for feature in all_features}

    while not all(at_boundary.values()):
        feature_to_adjust = random.choice(all_features)
        if at_boundary[feature_to_adjust]:
            continue

        original_value = original_datapoint[feature_to_adjust].iloc[0]
        adversarial_value = adversarial_datapoint[feature_to_adjust].iloc[0]

        if original_value == adversarial_value:
            at_boundary[feature_to_adjust] = True
            continue

        ordered_vals = list(correlation_dict[feature_to_adjust].keys())
        new_val = get_next_step_toward_original(adversarial_value, original_value, ordered_vals)

        adversarial_temp = adversarial_datapoint.copy()
        adversarial_temp.at[adversarial_temp.index[0], feature_to_adjust] = new_val

        if evaluate_DT(adversarial_temp) != true_label:
            number_of_evaluations += 1
            adversarial_datapoint = adversarial_temp
        else:
            at_boundary[feature_to_adjust] = True

    print("\nMinimized adversarial example found!")
    print(adversarial_datapoint.T)

    # --- Print differing features and their values with correlations ---
    differing_features = {}
    for feature in all_features:
        original_val = original_datapoint[feature].iloc[0]
        adversarial_val = adversarial_datapoint[feature].iloc[0]
        if original_val != adversarial_val:
            original_corr = correlation_dict[feature].get(original_val, None)
            adversarial_corr = correlation_dict[feature].get(adversarial_val, None)
            differing_features[feature] = {
                'original': (original_val, f"{original_corr:.4f}" if original_corr is not None else "N/A"),
                'adversarial': (adversarial_val, f"{adversarial_corr:.4f}" if adversarial_corr is not None else "N/A")
            }

    if differing_features:
        print("\nDiffering features between original and minimized adversarial (with correlations):")
        for feature, values in differing_features.items():
            print(f"Feature: {feature}")
            print(f"  Original Value: {values['original'][0]} (Correlation: {values['original'][1]})")
            print(f"  Adversarial Value: {values['adversarial'][0]} (Correlation: {values['adversarial'][1]})")
    else:
        print("\nNo features differ between the original and minimized adversarial example.")
    # --- End of printing differing features with correlations ---

    print("\nNumber of attempts (initial search):", n)
    print("True Label:", true_label)
    print("Predicted Label (Minimized Adversarial):", evaluate_DT(adversarial_datapoint))
    number_of_evaluations += 1

    # Compute correlation score distance
    corr_distance = 0
    for feature in all_features:
        val_orig = original_datapoint[feature].iloc[0]
        val_adv = adversarial_datapoint[feature].iloc[0]
        corr_val_orig = correlation_dict[feature].get(val_orig, 0)
        corr_val_adv = correlation_dict[feature].get(val_adv, 0)
        corr_distance += abs(corr_val_orig - corr_val_adv)

    print("\nCorrelation Score Distance between original and minimized adversarial:", corr_distance)

else:
    print(f"\nFailed to find an initial adversarial example within {max_attempts} attempts.")

print("\nTotal Evaluations:", number_of_evaluations)