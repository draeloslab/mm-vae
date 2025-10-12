import os
import yaml 
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from data_builder import DataBuilder
import pandas as pd 

from cgmvae import*
from vae import*
 
def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config

model_weights = '/Users/racheliritani/Desktop/AD Project/mm-vae/results_DIET/VanillaVAE_5dim/saved_model_epoch5000.pth'
data_path = '/Users/racheliritani/Desktop/AD Project/VAE-work/c-gmvae/nature_filtered_nonan_cfc.csv'
output_path = '/Users/racheliritani/Desktop/AD Project/mm-vae/cfc/vanilla_vae_5dim'
latent_dim = 5
os.makedirs(output_path, exist_ok=True)

accelerator = True 

if accelerator:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

all_data = pd.read_csv(data_path)
# mouse_ids = all_data['MouseID']
# mouse_ids.to_csv(os.path.join(output_path, 'mouse_ids.csv'))

test_set = DataBuilder(all_data)
# don't want to shuffle because we want it to be in the same order as the mouseids 
test_loader = DataLoader(test_set, batch_size=504)

# model = CGMVAE().to(device)
# model.load_state_dict(torch.load(model_weights, map_location=device))
model = torch.load(model_weights, map_location=device)
model.eval()

for batch_idx, (data, labels, df_layer1, mouse_ids) in enumerate(test_loader):
    data = data.to(device)
    labels = labels.to(device)
    # pass in labels if you are using C-GMVAE but no labels if you are using normal VAE 
    #recon_batch, mu, logvar, z = model(data, labels)
    recon_batch, mu, logvar, z = model(data)

    latent_vectors = z.detach().cpu().numpy()
    latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_dim)])
    latent_space['labels'] = labels.detach().cpu().numpy()
    len(mouse_ids)
    latent_space['mouse_ids'] = list(mouse_ids)
    latent_space.to_csv(os.path.join(output_path, f'latent_variables_1000epochs.csv'), index=False)
    print('Latent variables saved')

    recons = pd.DataFrame(recon_batch.detach().cpu().numpy())
    recons.to_csv(os.path.join(output_path, f'recons_epoch_1000epochs.csv'), index=False)
    print('Reconstruction features saved')
