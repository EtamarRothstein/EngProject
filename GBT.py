import ydf
import pandas as pd

# Load the saved DataFrames
df_train = pd.read_csv('train_data.csv')
df_validation = pd.read_csv('validation_data.csv')
df_experiment = pd.read_csv('experiment_data.csv')

# Ensure the target column 'income' is treated as a string (for classification)
df_train["income"] = df_train["income"].astype(str)
df_validation["income"] = df_validation["income"].astype(str)
df_experiment["income"] = df_experiment["income"].astype(str)

# Train the Gradient Boosted Trees model
model = ydf.GradientBoostedTreesLearner(label="income").train(df_train)


# Look at a model (input features, training logs, structure, etc.)
model.describe()

# Evaluate a model (e.g. roc, accuracy, confusion matrix, confidence intervals)
model.evaluate(df_experiment)

# Generate predictions
model.predict(df_experiment)

# Analyse a model (e.g. partial dependence plot, variable importance)
model.analyze(df_experiment)

# Benchmark the inference speed of a model
model.benchmark(df_experiment)

