import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib  # Import joblib

# Load the saved DataFrames
df_train = pd.read_csv('train_data.csv')
df_validation = pd.read_csv('validation_data.csv')
df_experiment = pd.read_csv('experiment_data.csv')

# Separate features (X) and target (y)
X_train = df_train.drop('income', axis=1)
X_validation = df_validation.drop('income', axis=1)
X_experiment = df_experiment.drop('income', axis=1)

y_train = df_train['income']
y_validation = df_validation['income']
y_experiment = df_experiment['income']

print("First few rows of X_train BEFORE one-hot encoding:")
print(X_train.head())

# Handle categorical features (assuming they are still in string format)
X_train = pd.get_dummies(X_train)
X_validation = pd.get_dummies(X_validation)
X_experiment = pd.get_dummies(X_experiment)

print("First few rows of X_train AFTER one-hot encoding:")
print(X_train.head())

# Ensure consistent columns
train_cols = X_train.columns
X_validation = X_validation.reindex(columns=train_cols, fill_value=0)
X_experiment = X_experiment.reindex(columns=train_cols, fill_value=0)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# **Save the trained model**
joblib.dump(model, 'decision_tree_model.joblib')
print("Trained decision tree model saved as 'decision_tree_model.joblib'")

# **Save the training columns**
joblib.dump(train_cols, 'train_cols.joblib')
print("Training columns saved as 'train_cols.joblib'")

# Make predictions and evaluate on validation set
y_pred_val = model.predict(X_validation)
accuracy_val = accuracy_score(y_validation, y_pred_val)
print(f"Validation Accuracy: {accuracy_val:.4f}")

# Make predictions on the experiment set
y_pred_exp = model.predict(X_experiment)
accuracy_exp = accuracy_score(y_experiment, y_pred_exp)
print(f"Experiment Accuracy: {accuracy_exp:.4f}")

# Create pandas Series from the true and predicted labels
y_true_series = df_experiment['income'].reset_index(drop=True)
y_pred_series = pd.Series(y_pred_exp)

# Create a DataFrame to compare true and predicted labels
comparison_df = pd.DataFrame({'true_income': y_true_series, 'predicted_income': y_pred_series})

# Identify correctly predicted examples
correct_predictions_df = df_experiment[comparison_df['true_income'] == comparison_df['predicted_income']].copy()

# Identify incorrectly predicted examples
incorrect_predictions_df = df_experiment[comparison_df['true_income'] != comparison_df['predicted_income']].copy()

print("DataFrame of correctly predicted examples:")
print(correct_predictions_df.head())
print(f"\nShape of correctly predicted DataFrame: {correct_predictions_df.shape}")

print("\nDataFrame of incorrectly predicted examples:")
print(incorrect_predictions_df.head())
print(f"\nShape of incorrectly predicted DataFrame: {incorrect_predictions_df.shape}")