import torch
import wandb  # Import Weights and Biases
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Experiment and Project names
exp_name = "summary_without_input"
project_name = "finetune_BART_generate_novel_summaries"

# Initialize wandb
wandb.init(project=project_name, name=exp_name)

# Initialize GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("kmfoda/booksum")
train_dataset = dataset['train']

# Initialize BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)

# Tokenize the 'summary_text' field
def tokenize_data(example):
    encoded_summary = tokenizer.encode(example['summary_text'], truncation=True, padding='max_length', max_length=1024)
    return {
        'labels': encoded_summary,
        'input_ids': [0],  # Dummy input_ids
        'attention_mask': [0]  # Dummy attention_mask
    }

tokenized_dataset = train_dataset.map(tokenize_data)

# Create DataLoader
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8)

# Training Arguments
training_args = TrainingArguments(
    output_dir=f"./results/{exp_name}",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir=f'./logs/{exp_name}',
    report_to="wandb"  # Enable wandb for logging
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model
model_path = f"./saved_models/BART_{exp_name}"
model.save_pretrained(model_path)
