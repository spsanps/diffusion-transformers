from datasets import load_dataset
from transformers import BartTokenizer, BartModel
import torch
from tqdm import tqdm

# Initialize GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("kmfoda/booksum")
train_dataset = dataset['train']

# Initialize BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartModel.from_pretrained('facebook/bart-base', output_hidden_states=True).to(device)

# Batch size
batch_size = 16

# Open raw binary file
with open('final_layer_output.bin', 'wb') as bin_file:

    for i in tqdm(range(0, len(train_dataset), batch_size), desc='Processing batches'):
        
        batch_data = train_dataset[i: i + batch_size]
        batch_text = batch_data['summary_text']
        
        encoding = tokenizer(batch_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
        input_ids = encoding['input_ids'].to(device)

        # Assertion to check padding
        assert input_ids.shape[1] == 1024, f"Expected padding to 1024 but got {input_ids.shape[1]}"

        with torch.no_grad():
            output = model(input_ids)
            final_layer_output = output.encoder_hidden_states[-1].cpu().numpy()

        # Write the batch data to binary file
        bin_file.write(final_layer_output.tobytes())
