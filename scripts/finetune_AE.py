from custom_BARTs.vae_BART import EncoderAE
from binary_dataset.binary_dataset import BinaryFileDataset
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
project_name = "bart-vae-128-one-sentence"
run_name = "bart-vae-128-one-sentence_2048"

wandb_cfg = {
    "project_name": project_name,
    "run_name": run_name,
}

# load BART
from transformers import BartModel
pretrained_bart = BartModel.from_pretrained('facebook/bart-base')

# create model

vae = EncoderAE(pretrained_bart.config, latent_dim=2048)
vae.load_pretrained(pretrained_bart)
vae.run_train(data_loader, wandb_cfg=wandb_cfg, save_path=f"saved_models/{run_name}")

# encode dataset

# reset dataset
dataset = BinaryFileDataset(
    "data/final_layer_output_128.bin", dtype=dtype, sample_shape=sample_shape
)

data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

# encode dataset
vae.encode_data_loader(data_loader, save_path=f"data/vae-encoded_data.bin")

# save first decoded output
import torch
out = vae.forward(torch.from_numpy(dataset[0]).unsqueeze(0).cuda())
torch.save(out, f"data/vae-sample_output.pt")