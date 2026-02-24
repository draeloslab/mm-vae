#from __future__ import print_function
import yaml 
import torch
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt 
import numpy as np
import time

from vae_svhn import*
from vae_mnist import*
from train_mmvae import*
from latent_regularizer import*
from moemmvae import*
from data_builder import*

# load config file 
def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config

#load all parameters 
config = load_config("config_mmvae.yaml")
random_seed = config['data']['random_seed']
latent_dim = config['model_params']['latent_dim']
epochs = config['training_params']['epochs']
batch_size = config['training_params']['batch_size']
log_interval = config['training_params']['log_interval']
mnist_scale = config['loss_params']['mnist_scale']
svhn_scale = config['loss_params']['svhn_scale']
num_samples = config['loss_params']['num_samples']
output_folder =  config['output']['output_folder']
accelerator = config['runtime_config']['accelerator']

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# set device to gpu if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if accelerator else {}

# initialize the Joint MMVAE model 
model = MMVAE(latent_dim, mnist_scale, svhn_scale)
model.to(device)

# data loaders initialize function get_paired_data_loaders 
train_loader, test_loader = getDataLoaders(batch_size, device=device)

# potentially remove ams_grad parameter 
optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

# add saving all individual loss components at some point
if __name__ == "__main__":
    total_loss = []
    total_test_loss = []
    test_epochs = []
    start_time = time.time()
    for epoch in range(1, epochs+1):
        loss = train(epoch, model, optimizer, train_loader, num_samples, device)
        total_loss.append(loss) 
        step = int(epoch / 5)
        if epoch == 1 or epoch == 2: 
        #if epoch in [step, step * 2, step * 3, step * 4, step * 5]:
            test_epochs.append(epoch)
            test_loss = test(epoch, model, optimizer, test_loader, num_samples, device, output_folder)
            total_test_loss.append(test_loss)
    end_time = time.time()
    print(str(end_time - start_time) + ' seconds for training/testing')

    plt.figure()
    plt.plot(total_loss, label='Training loss')                                                                          
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Train ELBO Loss')             
    plt.savefig(os.path.join(output_folder, 'train_loss.png'))


    plt.figure()
    plt.plot(test_epochs, total_test_loss, label='Testing loss')
    plt.title('Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Test ELBO Loss')
    plt.savefig(os.path.join(output_folder, 'test_loss.png'))
    print('Plotted loss curves')