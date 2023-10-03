import torch
import wandb
from torch.utils.data import DataLoader
from transformers import BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

from custom_BARTs.noise_encoder_BART import BartForConditionalGeneration, BartConfig


# Experiment and Project names
"""exp_name = "summary_with_noise_working_p4"
project_name = "finetune_BART_with_noise"

# Initialize wandb
wandb.init(project=project_name, name=exp_name)"""

# Initialize GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("kmfoda/booksum")
train_dataset = dataset['train']

# Initialize BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
config = BartConfig.from_pretrained('.saved_models/BART_summary_with_noise_working_p4')
config.encoder_gaussian_ratio=0.4
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', config=config).to(device)

# Tokenize the 'summary_text' field
def tokenize_data(example):
    encoded_summary = tokenizer.encode(example['summary_text'], truncation=True, padding='max_length', max_length=1024)
    return {
        'labels': encoded_summary,
        'input_ids': encoded_summary,  # Dummy input_ids
        'attention_mask': [1] * len(encoded_summary)  # Dummy attention_mask
    }

tokenized_dataset = train_dataset.map(tokenize_data)

# print one random summary
import random
#random.seed(42)
random_index = random.randint(0, len(train_dataset))
print("Random index: ", random_index)
print("Original summary: ", train_dataset[random_index]['summary_text'])

# inference
output = model.generate(tokenized_dataset[random_index]['input_ids'].unsqueeze(0).to(device))
print("Generated summary: ", tokenizer.decode(output[0], skip_special_tokens=True))