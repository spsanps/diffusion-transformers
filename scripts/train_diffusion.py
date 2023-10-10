
from transformers import BartModel

# Load pre-trained BART model
pretrained_bart = BartModel.from_pretrained('facebook/bart-base')

from custom_BART_encoder_diffusion.encoder_diffusion import BartEncoder as CustomBartEncoder
from diffusion.diffusion import convert_to_diffusion
from binary_dataset.binary_dataset import BinaryFileDataset

DiffusionEncoder = convert_to_diffusion(CustomBartEncoder)


# Your custom encoder model (assuming it is instantiated and named 'custom_encoder')

# get config of the pre-trained BART model
bart_config = pretrained_bart.config

custom_encoder = DiffusionEncoder(bart_config)

# Loop through both state dictionaries
if True:
    for name_param_custom, param_custom in custom_encoder.named_parameters():
        for name_param_pretrained, param_pretrained in pretrained_bart.encoder.named_parameters():
            
            # Check if the names match and not fc_residual
            if name_param_custom == name_param_pretrained and 'fc_residual' not in name_param_custom:
                
                # Copy weights
                param_custom.data.copy_(param_pretrained.data)



    ## 

import numpy as np

# Define the shape of one sample and its data type (change as needed)
sample_shape = (128, 768)  # e.g., (1, 512)
dtype = np.float32  # or np.float64, depending on how you saved it

# Create a dataset from the binary file
dataset = BinaryFileDataset(
    "data/final_layer_output_128.bin", dtype=dtype, sample_shape=sample_shape
)

# Create a data loader
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# create wandb config from BartEncoder config
import wandb
project_name = "bart-encoder-diffusion-128-one-sentence"
run_name = "bart-encoder-diffusion-128-one-sentence_5"

custom_encoder.config.wandb_project = project_name
custom_encoder.config.wandb_run_name = run_name

wandb_cfg = {
    "batch_size": custom_encoder.config.batch_size,
    "lr": custom_encoder.config.lr,
    "epochs": custom_encoder.config.epochs,
    "wandb_project": project_name,
    "wandb_run_name": run_name,
}

save_path = f"saved_models/{run_name}"

# Train the model
custom_encoder.train_diffusion(dataset, save_path, wandb_cfg=wandb_cfg)

# generate embeddings

custom_encoder.generate_diffusion(n=8, save_path="data/generated_embeddings_128.bin")


