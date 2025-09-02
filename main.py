#from __future__ import print_function
import yaml 
import torch
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

from moemmvae import*
from train import*
from latent_regularizer import*
from loss import*

def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

random_seed = config['data']['random_seed']
epochs = config['training_params']['epochs']
batch_size = config['training_params']['batch_size']
log_interval = config['training_params']['log_interval']
output_folder =  config['output']['output_folder']
accelerator = config['runtime_config']['accelerator']

if accelerator :
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if accelerator else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False, **kwargs)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
output_dir = os.makedirs(output_folder, exist_ok=True)

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, device, optimizer, log_interval)
        test(epoch, model, test_loader, device, batch_size, output_folder)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()  
            save_image(sample.view(64, 1, 28, 28),
                       output_folder + '/sample_' + str(epoch) + '.png')