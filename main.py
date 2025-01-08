import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch

# Load the dataset
file_path = r"metaai\Riddles.csv"
data = pd.read_csv(file_path)

# Combine Riddle and Hint columns
data['Combined_Text'] = data['Riddle'] + " " + data['Hint']

# Features and target
X = data['Combined_Text']
y = data['Answer']

# Convert target to numerical labels
labels = {label: idx for idx, label in enumerate(y.unique())}
data['label'] = data['Answer'].map(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['Combined_Text'], data['label'], test_size=0.2, random_state=42)

# Define a custom dataset class
class RiddleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))

# Prepare datasets
train_dataset = RiddleDataset(X_train, y_train, tokenizer, max_len=128)
test_dataset = RiddleDataset(X_test, y_test, tokenizer, max_len=128)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model_save_path = r"D:\git\metaai\bert_riddle_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved at: {model_save_path}")

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)

# Example usage
def predict_riddle(riddle, hint):
    model.eval()
    combined_text = riddle + " " + hint
    encoding = tokenizer.encode_plus(
        combined_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return [key for key, value in labels.items() if value == predicted_label][0]

example_riddle = "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?"
example_hint = "Think about a sound that repeats itself in nature"
print("Predicted Answer:", predict_riddle(example_riddle, example_hint))
