from vae import*
import pandas as pd
import sys
from utils import*

trained_model_path = '/Users/racheliritani/Desktop/AD Project/mm-vae/results_DIET/CGMVAE_cfc_cond_5bins_5dim/saved_model_epoch5000.pth'
data_path = '/Users/racheliritani/Desktop/AD Project/VAE-work/c-gmvae/nature_filtered_nonan_cfc.csv'
output_path = '/Users/racheliritani/Desktop/AD Project/mm-vae/cfc/'
data_with_cfc = pd.read_csv(data_path)
input_data = data_with_cfc.iloc[:, 1:-1]

trained_weights = torch.load(trained_model_path)

vanillaVAE = VAE()

vanillaVAE_state_dict = vanillaVAE.state_dict()

vanillaVAE_state_dict['bn1.weight'] = trained_weights['bn1.weight']
vanillaVAE_state_dict['bn1.bias'] = trained_weights['bn1.bias']
vanillaVAE_state_dict['fc21.weight'] = trained_weights['fc21.weight']
vanillaVAE_state_dict['fc21.bias'] = trained_weights['fc21.bias']
vanillaVAE_state_dict['fc22.weight'] = trained_weights['fc22.weight']
vanillaVAE_state_dict['fc22.bias'] = trained_weights['fc22.bias']

fc1_trained_weights_labeled = trained_weights['fc1.weight']
fc1_trained_weights_nolabel = fc1_trained_weights_labeled[:, :36]
vanillaVAE_state_dict['fc1.weight'] = fc1_trained_weights_nolabel
vanillaVAE_state_dict['fc1.bias'] = trained_weights['fc1.bias']

vanillaVAE.load_state_dict(vanillaVAE_state_dict)

vanillaVAE.eval()
with torch.no_grad(): 
    mu, logvar = vanillaVAE(input_data)

