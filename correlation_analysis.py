import pandas as pd
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
import json

# ---- Step 1: Load and prepare the main dataset ----
df_main = pd.read_csv("experiment_data.csv")
df_main['income_binary'] = df_main['income'].apply(lambda x: 1 if '>50K' in x else 0)

target = df_main['income_binary']
features_main = df_main.drop(columns=['income', 'income_binary'])

# ---- Step 2: Compute correlation per value ----
correlation_dict = {}

for col in features_main.columns:
    value_corrs = {}
    if df_main[col].dtype == 'object':
        for val in df_main[col].unique():
            binary_vec = df_main[col].apply(lambda x: 1 if x == val else 0)
            corr, _ = pointbiserialr(binary_vec, target)
            value_corrs[val] = corr if pd.notnull(corr) else 0
    else:
        for val in sorted(df_main[col].unique()):
            binary_vec = df_main[col].apply(lambda x: 1 if x == val else 0)
            corr, _ = pointbiserialr(binary_vec, target)
            value_corrs[val] = corr if pd.notnull(corr) else 0
    correlation_dict[col] = dict(sorted(value_corrs.items(), key=lambda x: x[1], reverse=True))

# ---- Print sorted value-correlation lists ----
# for feature, val_corrs in correlation_dict.items():
#     print(f"\nFeature: {feature}")
#     for val, corr in val_corrs.items():
#         print(f"  {val}: {corr:.4f}")


# ---- Step 3: Function to compute row score ----
def compute_score(df, corr_dict):
    def row_score(row):
        score = 0
        for col in corr_dict:
            val = row.get(col)
            score += corr_dict[col].get(val, 0)
        return score
    return df.apply(row_score, axis=1)

# ---- Step 4: Apply to new datasets ----
for file in ["incorrect_predictions_DT.csv", "correct_predictions_DT.csv"]:
    df = pd.read_csv(file)
    df['correlation_score'] = compute_score(df, correlation_dict)
    avg_abs_score = df['correlation_score'].abs().mean()
    print(f"\nDataset: {file}")
    print(f"Average absolute correlation score: {avg_abs_score:.4f}")

# ---- Step 5: Plot correlation values for numerical features ----
# import matplotlib.pyplot as plt
#
# numerical_features = ['age', 'educational-num', 'hours-per-week']
#
# for feature in numerical_features:
#     corr_data = correlation_dict.get(feature, {})
#     x = list(corr_data.keys())
#     y = list(corr_data.values())
#
#     plt.figure(figsize=(8, 4))
#     plt.scatter(x, y, color='teal')
#     plt.title(f"Correlation of '{feature}' values with income")
#     plt.xlabel("Value")
#     plt.ylabel("Correlation")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()



# # Store it in a JSON file
# json_data = json.dumps(correlation_dict, indent=4)

# You can write this to a file if needed
# with open("correlation_dict_int.json", "w") as file:
#     file.write(json_data)
new
# print(json_data)



