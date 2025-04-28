import pandas as pd
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump
from sklearn.preprocessing import LabelEncoder

# Load data and model
try:
    model = joblib.load('decision_tree_model.joblib')
    df_experiment = pd.read_csv('experiment_data.csv')
    X_experiment_original = df_experiment.drop('income', axis=1).copy() # Keep original features
    y_experiment_original = df_experiment['income'].copy()

    # One-hot encode the data
    X_experiment_encoded = pd.get_dummies(X_experiment_original).values
    train_cols_encoded = joblib.load('train_cols.joblib')
    temp_df = pd.DataFrame(X_experiment_encoded, columns=pd.get_dummies(X_experiment_original).columns)
    X_experiment_encoded_aligned = temp_df.reindex(columns=train_cols_encoded, fill_value=0).values

    # Label encode the target variable for ART
    label_encoder = LabelEncoder()
    y_experiment_encoded = label_encoder.fit_transform(y_experiment_original)

    # Identify indices of the numerical features after one-hot encoding
    numerical_features = ['age', 'educational-num', 'hours-per-week']
    encoded_columns = list(train_cols_encoded)
    numerical_feature_indices = [i for i, col in enumerate(encoded_columns) if col in numerical_features]

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# Wrap the scikit-learn model with ART's SklearnClassifier
art_classifier = SklearnClassifier(model=model, clip_values=(0, 1)) # Adjust clip_values if your numerical features are scaled differently

# Instantiate the HopSkipJump attack
attack = HopSkipJump(classifier=art_classifier, norm=2, max_iter=50, max_eval=5000, verbose=True)  # Changed 1 to True

# Create a mask (1 for perturbable features, 0 for others)
mask = np.zeros(X_experiment_encoded_aligned.shape[1])
for index in numerical_feature_indices:
    if index < mask.shape[0]:
        mask[index] = 1.0

# Select an example to attack
target_index = 0
attack_example = X_experiment_encoded_aligned[target_index:target_index+1]
original_label_encoded = y_experiment_encoded[target_index]
target_label_art = np.array([1 - original_label_encoded])

# Generate the adversarial example with the mask
try:
    adversarial_example = attack.generate(x=attack_example, y=target_label_art, mask=mask)

    print("Original Example (Encoded):", attack_example)
    print("Original Prediction (Encoded):", model.predict(attack_example))
    print("Adversarial Example:", adversarial_example)
    print("Adversarial Prediction (Encoded):", model.predict(adversarial_example))

    # You might want to see the perturbed numerical features
    encoded_columns_array = np.array(encoded_columns)
    numerical_feature_names_encoded = encoded_columns_array[numerical_feature_indices]

    original_numerical_values = attack_example[:, numerical_feature_indices]
    adversarial_numerical_values = adversarial_example[:, numerical_feature_indices]

    print("\nNumerical Features (Encoded):", numerical_feature_names_encoded)
    print("Original Numerical Values:", original_numerical_values)
    print("Adversarial Numerical Values:", adversarial_numerical_values)

except Exception as e:
    print(f"Error during attack generation: {e}")