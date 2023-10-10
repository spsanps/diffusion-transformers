from datasets import load_dataset, load_from_disk
from transformers import BartTokenizer, BartModel
import torch
from tqdm import tqdm

# Initialize GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_from_disk("data/booksum_one_sentence_dataset")
dataset = dataset["train"]

# Initialize BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartModel.from_pretrained(
    "saved_models/BART_summary_with_noise_working_p1_45", output_hidden_states=True
).to(device)

# Batch size
batch_size = 16

# Open raw binary file
with open("final_layer_output_128.bin", "wb") as bin_file:
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch_data = dataset[i : i + batch_size]
        batch_text = batch_data["one_sentence_summary"]

        encoding = tokenizer(
            batch_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        input_ids = encoding["input_ids"].to(device)

        # Assertion to check padding
        assert (
            input_ids.shape[1] == 128
        ), f"Expected padding to 128 but got {input_ids.shape[1]}"

        with torch.no_grad():
            output = model(input_ids)
            final_layer_output = output.encoder_hidden_states[-1].cpu().numpy()

        # Write the batch data to binary file
        bin_file.write(final_layer_output.tobytes())
