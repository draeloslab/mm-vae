#from __future__ import print_function
import yaml 
import torch
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms
import os
import pandas as pd
from data_builder import DataBuilder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from train import*
from loss import*
from vanilla_vae import*
from utils import*

def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

latent_dim = config['model_params']['latent_dim']
random_seed = config['data']['random_seed']
epochs = config['training_params']['epochs']
batch_size = config['training_params']['batch_size']
learning_rate = config['training_params']['learning_rate']
log_interval = config['training_params']['log_interval']
beta = config['training_params']['beta']
output_folder =  config['output']['output_folder']
accelerator = config['runtime_config']['accelerator']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if accelerator else {}

beam_break_data = pd.read_csv('/home/rachel/Desktop/mm-vae_ext2/data/raw_beam_break_standardized_binned_dropna.csv')
train_set = DataBuilder(beam_break_data)
test_set = DataBuilder(beam_break_data)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True, collate_fn=custom_collate)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
output_dir = os.makedirs(output_folder, exist_ok=True)

if __name__ == "__main__":
    total_train_loss = []
    total_bce_loss = []
    total_kld_loss = []
    for epoch in range(1, epochs + 1):
        train_loss, bce_loss, kld_loss = train(epoch, model, train_loader, device, optimizer, log_interval, test_loader, latent_dim, output_folder, beta)
        total_train_loss.append(train_loss)
        total_bce_loss.append(bce_loss)
        total_kld_loss.append(kld_loss)

    fix, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(total_train_loss, color = 'blue', label='Training loss')
    ax[0].plot(total_bce_loss, color='red', label='Training MSE loss')
    ax[0].plot(np.array(total_kld_loss) * beta, color='green', label='Training KLD loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')
    ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].plot(total_bce_loss, color='red', label='Training MSE loss')
    ax[1].set_title('Training MSE plotted alone')
    ax[1].set_yscale('log')

    plt.suptitle('Training loss for the Vanilla VAE')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'train_test_loss.png'))