import os
import torch
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from data_builder_no_labels import MLPDataBuilder

model_weights = '/Users/racheliritani/Desktop/AD Project/mm-vae/results_DIET/CGMVAE_cfc_cond_5bins_5dim/mlp_15000/saved_model_epoch15000.pth'
data_path = '/Users/racheliritani/Desktop/AD Project/mm-vae/cfc/no_cfc_latent_space/latent_variables_5000epochs.csv'
output_path = '/Users/racheliritani/Desktop/AD Project/mm-vae/cfc_pred'

os.makedirs(output_path, exist_ok=True)

accelerator = True 
if accelerator:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

data_no_cfc_lv = pd.read_csv(data_path)

latent_dim = 5
latent_columns = [f'LV{i+1}' for i in range(latent_dim)]

with open('/Users/racheliritani/Desktop/AD Project/mm-vae/results_DIET/CGMVAE_cfc_cond_5bins_5dim/mlp_15000/scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

with open('/Users/racheliritani/Desktop/AD Project/mm-vae/results_DIET/CGMVAE_cfc_cond_5bins_5dim/mlp_15000/Test/y_scaler.pkl', 'rb') as file:
    loaded_y_scaler = pickle.load(file)

data_no_cfc_lv.loc[:, latent_columns] = loaded_scaler.transform(data_no_cfc_lv.loc[:, latent_columns])
df = MLPDataBuilder(data_no_cfc_lv, latent_columns)
data_loader = DataLoader(df, batch_size = len(df), shuffle=False)

model = torch.load(model_weights, map_location=device)
model.eval()

for batch_idx, (data, mouse_ids) in enumerate(data_loader):
    data = data.to(device)
    output = model(data)

    original_scale_output = loaded_y_scaler.inverse_transform(output.cpu().detach().numpy())

    out = pd.DataFrame(original_scale_output)
    out['mouse_ids'] = mouse_ids
out.to_csv('/Users/racheliritani/Desktop/AD Project/mm-vae/cfc_pred/predicted_cfc.csv', index=False)
