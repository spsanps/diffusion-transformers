import json
import shutil
from openai_helpers.openai_helpers import prompt_request
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from tqdm import tqdm
import random

system_prompt = "You are summarizing bot. Summarize given plot in one sentence. Only provide the plot summary - no context is required."

batch_size = 100  # Adjust as needed

def get_summary(text):
    return prompt_request(
        prompt=text, system_prompt=system_prompt, temperature=0, output_max_tokens=128, log=False
    )

# Load last saved data
try:
    with open("booksum_one_sentence.json", "r") as f:
        data = json.load(f)
        processed_data = data["processed_data"]
except FileNotFoundError:
    processed_data = {"train": [], "validation": [], "test": []}

batch_size = 100  # Adjust as needed

# Load the dataset
dataset = load_dataset("kmfoda/booksum")

# Process and save periodically
for split in ["train", "validation", "test"]:
    print(f"Split: {split}")

    last_saved_index = len(processed_data[split])

    for i in tqdm(range(last_saved_index, len(dataset[split]["summary_text"]))):
        entry = dataset[split][i]
        entry["one_sentence_summary"] = get_summary(entry["summary_text"])
        processed_data[split].append(entry)

        if (i + 1 - last_saved_index) % batch_size == 0:
            # Save the data
            with open("booksum_one_sentence.json", "w") as f:
                json.dump({"processed_data": processed_data}, f)
            
            last_saved_index = i + 1  # Update last saved index

# Print a random summary
random.seed(42)
random_index = random.randint(0, len(dataset['train']) - 1)
print(f"Random index: {random_index}")
print(f"Original summary: {dataset['train'][random_index]['summary_text']}")
print(f"Generated summary: {processed_data['train'][random_index]['one_sentence_summary']}")


