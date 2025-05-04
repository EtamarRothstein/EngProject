import pandas as pd
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
from tqdm import tqdm  # <-- new

# Load the fine-tuned model and tokenizer
model_path = './fine_tuned_roberta_income'
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# Load the experiment data
experiment_df = pd.read_csv("experiment_data.csv")

# Define the mask token and its ID
mask_token = tokenizer.mask_token
mask_token_id = tokenizer.mask_token_id

# Define target token IDs for income
greater_than_id = 9312  # Token ID for " Greater"
less_than_id = 10862    # Token ID for " Less"

# Function to create masked input sequences for the experiment data
def create_masked_inputs_experiment(df):
    inputs = []
    for _, row in df.iterrows():
        sentence = (
            f"age: {row['age']}, workclass: {row['workclass']}, education: {row['education']}, "
            f"educational-num: {row['educational-num']}, marital-status: {row['marital-status']}, "
            f"occupation: {row['occupation']}, relationship: {row['relationship']}, race: {row['race']}, "
            f"gender: {row['gender']}, capital-gain: {row['capital-gain']}, capital-loss: {row['capital-loss']}, "
            f"hours-per-week: {row['hours-per-week']}, native-country: {row['native-country']}, "
            f"income:{mask_token} than 50k"
        )
        tokenized_input = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        inputs.append(tokenized_input)
    return inputs

# Create input tensors for the experiment data
experiment_inputs = create_masked_inputs_experiment(experiment_df)

# Make predictions with a progress bar
all_predicted_token_ids = []
with torch.no_grad():
    for input_tensor in tqdm(experiment_inputs, desc="Running inference"):  # <-- wrap with tqdm
        outputs = model(**input_tensor)
        logits = outputs.logits

        masked_index = (input_tensor['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
        if masked_index.numel() > 0:
            predicted_token_id = torch.argmax(logits, dim=-1)[0, masked_index[0]].item()
            all_predicted_token_ids.append(predicted_token_id)
        else:
            all_predicted_token_ids.append(None)

# Map the predicted token IDs back to income labels
predicted_incomes = []
for token_id in all_predicted_token_ids:
    if token_id == greater_than_id:
        predicted_incomes.append('>50K')
    elif token_id == less_than_id:
        predicted_incomes.append('<=50K')
    else:
        predicted_incomes.append('UNKNOWN')

# Add predicted incomes as a new column
experiment_df['predicted_income'] = predicted_incomes

# Calculate accuracy
correct_predictions = (experiment_df['income'].str.strip() == experiment_df['predicted_income']).sum()
accuracy = correct_predictions / len(experiment_df)
print(f"Accuracy on the experiment set: {accuracy:.4f}")

# Create DataFrames for correct and incorrect predictions
correct_df = experiment_df[experiment_df['income'].str.strip() == experiment_df['predicted_income']].copy()
incorrect_df = experiment_df[experiment_df['income'].str.strip() != experiment_df['predicted_income']].copy()

# Add the predicted token ID to the DataFrames
correct_df['predicted_token_id'] = [all_predicted_token_ids[i] for i in correct_df.index]
incorrect_df['predicted_token_id'] = [all_predicted_token_ids[i] for i in incorrect_df.index]

# Save the DataFrames to new CSV files
correct_df.to_csv("correct_predictions_LM.csv", index=False)
incorrect_df.to_csv("incorrect_predictions_LM.csv", index=False)

print("Correctly predicted data saved to correct_predictions_LM.csv")
print("Incorrectly predicted data saved to incorrect_predictions_LM.csv")
