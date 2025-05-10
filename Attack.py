import pandas as pd
import numpy as np
from numpy.linalg import norm
from evaluation_funcs import evaluate_DT
import random

number_of_evaluations = 0

# Load dataset of datapoints the model predicts correctly
df = pd.read_csv('correct_predictions_DT.csv')

# Assume 'income' is the label column
label_column = 'income'
numerical_features = ['age', 'educational-num', 'hours-per-week']

# Initial values for perturbation bounds
initial_upper_bound = 5
initial_lower_bound = -initial_upper_bound

# Randomly choose a single data point
original_datapoint = df.sample(n=1).copy()
adversarial_datapoint = original_datapoint.copy()  # Start with a copy for modification

# Get the true label of the original data point
true_label = original_datapoint[label_column].iloc[0]

n = 0
max_attempts = 1000
max_perturbation = 100  # Example maximum perturbation range
upper_bound = initial_upper_bound  # Initialize upper_bound
lower_bound = initial_lower_bound  # Initialize lower_bound

# First, find an adversarial example
while evaluate_DT(adversarial_datapoint) == true_label and n < max_attempts and upper_bound <= max_perturbation:
    n += 1
    number_of_evaluations += 1
    upper_bound = 5 + n // 10
    lower_bound = -upper_bound
    array_size = len(numerical_features) # Perturb each feature independently

    # Generate array of random integers for perturbation
    random_array = np.random.randint(low=lower_bound, high=upper_bound + 1, size=array_size)

    # Apply the perturbation to the numerical features
    adversarial_datapoint[numerical_features] = adversarial_datapoint[numerical_features].values + random_array

# Attempt to move the adversarial example closer to the original
if evaluate_DT(adversarial_datapoint) != true_label:
    print("\nInitial adversarial example found!")
    print("Original Numerical Features:\n", original_datapoint[numerical_features])
    print("\nInitial Adversarial Numerical Features:\n", adversarial_datapoint[numerical_features])

    at_boundary = {feature: False for feature in numerical_features}

    while not all(at_boundary.values()):
        feature_to_adjust = random.choice(numerical_features)
        if not at_boundary[feature_to_adjust]:
            original_value = original_datapoint[feature_to_adjust].iloc[0]
            adversarial_value = adversarial_datapoint[feature_to_adjust].iloc[0]
            change = np.sign(adversarial_value - original_value)

            # If original and adversarial values are the same, consider it at the boundary
            if original_value == adversarial_value:
                at_boundary[feature_to_adjust] = True
                continue  # Move to the next feature

            # Try moving one step closer
            adversarial_datapoint_temp = adversarial_datapoint.copy()
            adversarial_datapoint_temp[feature_to_adjust] -= change

            if evaluate_DT(adversarial_datapoint_temp) != true_label:
                number_of_evaluations += 1
                adversarial_datapoint = adversarial_datapoint_temp
            else:
                at_boundary[feature_to_adjust] = True

    print("\nMinimized adversarial example found!")
    # print("Original Data Point:")
    # print(original_datapoint)
    # print((original_datapoint[numerical_features]))
    print("\nMinimized Adversarial Data Point:")
    # print(adversarial_datapoint)
    print((adversarial_datapoint[numerical_features]))
    print("\nNumber of attempts (initial search):", n)
    print("True Label:", true_label)
    print("Predicted Label (Minimized Adversarial):", evaluate_DT(adversarial_datapoint))
    number_of_evaluations += 1

    # Calculate L2 distance
    original_features = original_datapoint[numerical_features].values.flatten()
    adversarial_features = adversarial_datapoint[numerical_features].values.flatten()
    l2_distance = norm(original_features - adversarial_features)
    print("\nL2 distance between true data and minimized adversarial:", l2_distance)

else:
    print(f"Failed to find an initial adversarial example within {max_attempts} attempts (max perturbation: {max_perturbation}).")

print("\nTotal Evaluations:", number_of_evaluations)