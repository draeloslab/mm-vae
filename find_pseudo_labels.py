from vae import*
from utils import*
import pandas as pd
from data_builder_no_labels import DataBuilder
from torch.utils.data import DataLoader
import sys
import os 

trained_model_path = '/Users/racheliritani/Desktop/AD Project/mm-vae/results_DIET/CGMVAE_cfc_cond_5bins_5dim/saved_model_epoch5000.pth'
data_no_cfc_path = '/Users/racheliritani/Desktop/AD Project/VAE-work/c-gmvae/nature_filtered_nonan_nocfc.csv'
output_path = '/Users/racheliritani/Desktop/AD Project/mm-vae/cfc_pred/pseudo_labels_no_cfc.csv'
os.makedirs(output_path, exist_ok=True)

accelerator = True 

if accelerator:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
all_data = pd.read_csv(data_no_cfc_path)

test_set = DataBuilder(all_data)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
model = torch.load(trained_model_path, map_location=device)
model.eval()

for batch_idx, (data, df_layer1, mouse_ids) in enumerate(test_loader):
    data = data.to(device)
    if batch_idx < 5:
        print(f'batch (sample) {batch_idx}: ')
        for i in range(5):
            print(f'label {i}:')
            labels = torch.full((data.shape[0],), i, dtype=torch.int64)
            print(labels)
            labels = labels.to(device)
            mu, logvar = model.encode(data, labels)
            print('mu: ' + str(mu))
            #print('logvar: ' + str(logvar))



