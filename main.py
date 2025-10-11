#from __future__ import print_function
import yaml 
import torch
import torch.utils.data 
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt 
import numpy as np
import random 

from cgmvae import*
from vae import*
from train import*
from latent_regularizer import*
from loss import*
from data_builder import DataBuilder

def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

random_seed = config['data']['random_seed']

# set seed 
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

model_type = config['model']['type']
latent_dim = config['model']['latent_dim']
dataset = config['data']['dataset']
data_path = config['data']['data_path']
epochs = config['training_params']['epochs']
batch_size = config['training_params']['batch_size']
log_interval = config['training_params']['log_interval']
output_folder =  config['output']['output_folder']
os.makedirs(output_folder, exist_ok=True)
accelerator = config['runtime_config']['accelerator']
df_gmm = pd.read_csv(config['training_params']['gmm_centers'])
learning_rate = config['training_params']['learning_rate']
gmm_centers = torch.tensor([df_gmm[col].values for col in df_gmm.columns]).float()
gmm_centers = gmm_centers[1:]
gmm_std = config['training_params']['gmm_std']
ks_weight, cv_weight, samples, components = estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, num_samples=128)
components = np.array(components)
components = components[:, np.newaxis]
final_columns = [f'Dim{i+1}' for i in range(latent_dim)] + ['Component']
priors = pd.DataFrame(np.concatenate((samples.numpy(), components), axis=1), columns=final_columns)
priors.to_csv(os.path.join(output_folder, 'prior_distribution.csv'), index=False)
data_loss_weight = [config['loss_params']['data_loss_weight']]
kl_weight = config['loss_params']['kl_weight']

if accelerator:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if accelerator else {}

# load in data 
nature_filtered = pd.read_csv(data_path)
nature_filtered_nonan = set_nans_extreme(nature_filtered)
train_set = DataBuilder(nature_filtered_nonan)
test_set = DataBuilder(nature_filtered_nonan)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=937, shuffle=True)

if model_type == 'VAE' or model_type == 'GMVAE':
    model = VAE().to(device)
elif model_type == 'CGMVAE':
    model = CGMVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

output_dir = os.makedirs(output_folder, exist_ok=True)

if __name__ == "__main__":
    total_train_loss = []
    total_bce_loss = []
    total_kld_loss = []
    total_test_loss = []
    total_bce_test_loss = []
    total_kld_test_loss = []
    for epoch in range(1, epochs + 1):
        train_loss, bce_loss, kld_loss = train(epoch, model, train_loader, device, optimizer, log_interval, ks_weight, cv_weight, kl_weight, gmm_centers, gmm_std, model_type, test_loader, output_folder, latent_dim)
        total_train_loss.append(train_loss)
        total_bce_loss.append(bce_loss)
        total_kld_loss.append(kld_loss)

    fix, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(total_train_loss, color = 'blue', label='Training loss')
    ax[0].plot(total_bce_loss, color='red', label='Training MSE loss')
    ax[0].plot(total_kld_loss, color='green', label='Training KLD loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')
    ax[0].legend()

    ax[1].plot(total_bce_loss, color='red', label='Training MSE loss')
    ax[1].set_title('Training MSE plotted alone')

    plt.suptitle('Training loss for the ' + dataset + ' VAE')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'train_test_loss.png'))


