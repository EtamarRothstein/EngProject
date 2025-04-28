import pandas as pd
from sklearn.model_selection import train_test_split

df_original = pd.read_csv('adult.csv')

feature_list = df_original.columns.tolist()
print('List of features:')
print(feature_list)


unique_values_array = df_original['income'].unique()
print("List of possible values in 'income':")
print(unique_values_array)
print("Of type:")
print([type(value) for value in unique_values_array])

# Splitting the DataFrame
df_greater = df_original[df_original['income'] == '>50K']
df_less = df_original[df_original['income'] == '<=50K']

print(f"Number of examples with income greater than 50k: {df_greater.shape[0]}")
print(f"Number of examples with income less than 50k: {df_less.shape[0]}")

min_samples = min(df_greater.shape[0], df_less.shape[0])
print(f"\nNumber of samples to take from each group: {min_samples}")

# Randomly sample 'min_samples' from df_greater
df_greater_sampled = df_greater.sample(n=min_samples, random_state=42)
df_less_sampled = df_less.sample(n=min_samples, random_state=42)

print(df_less_sampled.shape[0])
print(df_greater_sampled.shape[0])

# Define the proportions for the splits
train_ratio = 0.5
validation_ratio = 0.1
experiment_ratio = 0.4

# Split df_greater_sampled
df_greater_train, temp_greater = train_test_split(df_greater_sampled, test_size=(validation_ratio + experiment_ratio), random_state=42)
df_greater_val, df_greater_exp = train_test_split(temp_greater, test_size=experiment_ratio / (validation_ratio + experiment_ratio), random_state=42)

# Split df_less_sampled
df_less_train, temp_less = train_test_split(df_less_sampled, test_size=(validation_ratio + experiment_ratio), random_state=42)
df_less_val, df_less_exp = train_test_split(temp_less, test_size=experiment_ratio / (validation_ratio + experiment_ratio), random_state=42)


# Concatenate the train sets
df_train = pd.concat([df_greater_train, df_less_train]).sample(frac=1, random_state=42).reset_index(drop=True)
df_validation = pd.concat([df_greater_val, df_less_val]).sample(frac=1, random_state=42).reset_index(drop=True)
df_experiment = pd.concat([df_greater_val, df_less_exp]).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nShape of the training dataset:", df_train.shape)
print("Value counts of 'income' in the training set:\n", df_train['income'].value_counts())

print("\nShape of the validation dataset:", df_validation.shape)
print("Value counts of 'income' in the validation set:\n", df_validation['income'].value_counts())

print("\nShape of the experimentation dataset:", df_experiment.shape)
print("Value counts of 'income' in the experimentation set:\n", df_experiment['income'].value_counts())