# evaluation_utils.py
import pandas as pd
import joblib

# Load the training columns (we'll load the model inside the function)
try:
    train_cols = joblib.load('train_cols.joblib')
except FileNotFoundError:
    print("Error: 'train_cols.joblib' not found. Make sure the model training script has been run.")
    train_cols = None

def evaluate_DT(data_point):
    """
    Evaluates the trained Decision Tree model's prediction on a single data point.
    The model is loaded from 'decision_tree_model.joblib'.

    Args:
        data_point: A pandas DataFrame containing a single row of the data point to evaluate.
                    The columns of this data point should match the features the model was trained on.

    Returns:
        The predicted class label for the data point, or None if the model or training columns are not loaded.
    """
    if train_cols is None:
        return None

    # Load the Decision Tree model
    try:
        model = joblib.load('decision_tree_model.joblib')
    except FileNotFoundError:
        print("Error: 'decision_tree_model.joblib' not found. Make sure the model training script has been run.")
        return None

    # Make a copy to avoid modifying the original data_point
    processed_data_point = data_point.copy()

    # Handle categorical features by performing one-hot encoding
    processed_data_point = pd.get_dummies(processed_data_point)

    # Ensure the data point has the same columns as the training data
    processed_data_point = processed_data_point.reindex(columns=train_cols, fill_value=0)

    # Make the prediction
    prediction = model.predict(processed_data_point)

    # Return the single prediction
    return prediction[0]