import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForMaskedLM, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch

# Load your datasets
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("validation_data.csv")

# Define the RoBERTa model and tokenizer
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)

# Define the mask token
mask_token = tokenizer.mask_token
mask_token_id = tokenizer.mask_token_id

# Define target token IDs for income
greater_than_id = 9312  # Token ID for " Greater"
less_than_id = 10862

# Function to create input sentences and labels (MODIFIED)
def create_masked_inputs(df):
    inputs = []
    labels = []
    for index, row in df.iterrows():
        sentence = f"age: {row['age']}, workclass: {row['workclass']}, education: {row['education']}, educational-num: {row['educational-num']}, marital-status: {row['marital-status']}, occupation: {row['occupation']}, relationship: {row['relationship']}, race: {row['race']}, gender: {row['gender']}, capital-gain: {row['capital-gain']}, capital-loss: {row['capital-loss']}, hours-per-week: {row['hours-per-week']}, native-country: {row['native-country']}, income:{mask_token} than 50k"
        tokenized_input = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        inputs.append(tokenized_input)

        label_ids = torch.full_like(tokenized_input['input_ids'], -100)
        masked_index = (tokenized_input['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]

        if masked_index.numel() > 0:
            if row['income'].strip() == '>50K':
                label_ids[0, masked_index[0]] = greater_than_id
            elif row['income'].strip() == '<=50K':
                label_ids[0, masked_index[0]] = less_than_id

        labels.append(label_ids)

    return inputs, labels

# Create the datasets
class IncomePredictionDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(),
            'labels': self.labels[idx].squeeze()
        }

train_inputs, train_labels = create_masked_inputs(train_df)
val_inputs, val_labels = create_masked_inputs(val_df)

train_dataset = IncomePredictionDataset(train_inputs, train_labels)
val_dataset = IncomePredictionDataset(val_inputs, val_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./roberta_income_prediction',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss', # Masked language modeling loss
    logging_dir='./logs_roberta_income',
    logging_steps=100,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda data: {k: torch.stack([d[k] for d in data]) for k in data[0]}, # Simple collator
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the fine-tuned model
trainer.save_model('./fine_tuned_roberta_income')
tokenizer.save_pretrained('./fine_tuned_roberta_income')

print("Fine-tuning complete. Model and tokenizer saved to ./fine_tuned_roberta_income")