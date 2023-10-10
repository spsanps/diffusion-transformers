import torch
import wandb
from torch.utils.data import DataLoader
from transformers import BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk

from custom_BARTs.noise_encoder_BART import BartForConditionalGeneration, BartConfig


# Experiment and Project names
exp_name = "summary_with_noise_working_p1_45_one_sentence_128"
project_name = "finetune_BART_with_noise_one_sentence_128"

# Initialize wandb
wandb.init(project=project_name, name=exp_name)

# Initialize GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_from_disk("data/booksum_one_sentence_dataset")
                       #load_dataset("kmfoda/booksum")
train_dataset = dataset['train']

# Initialize BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
config = BartConfig.from_pretrained('facebook/bart-base')
config.encoder_gaussian_ratio=0.45
config.encoder_gaussian_ratio_low=0.1
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', config=config).to(device)

# Tokenize the 'summary_text' field
def tokenize_data(example):
    encoded_summary = tokenizer.encode(example['one_sentence_summary'], truncation=True, padding='max_length', max_length=128)
    return {
        'labels': encoded_summary,
        'input_ids': encoded_summary,  # Dummy input_ids
        'attention_mask': [1] * len(encoded_summary)  # Dummy attention_mask
    }

tokenized_dataset = train_dataset.map(tokenize_data)

# Create DataLoader
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8)

# Training Arguments
training_args = TrainingArguments(
    output_dir=f"./results/{exp_name}",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_dir=f'./logs/{exp_name}',
    report_to="wandb",
    logging_steps=10  # Log every 10 steps

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
