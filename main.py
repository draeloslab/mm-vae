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

from vae_svhn import*
from vae_mnist import*
from train import*
from latent_regularizer import*
from loss_mse import*

def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

random_seed = config['data']['random_seed']
dataset = config['data']['dataset']
epochs = config['training_params']['epochs']
batch_size = config['training_params']['batch_size']
log_interval = config['training_params']['log_interval']
output_folder =  config['output']['output_folder']
accelerator = config['runtime_config']['accelerator']

# if accelerator :
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if accelerator else {}

# initialize MNIST normalization and standardization (z-score) transforms with pre-calculated mean and std values
mnist_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3801)
    ])

# load in MNIST train and test data
train_loader_MNIST = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=mnist_norm),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader_MNIST = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=mnist_norm),
    batch_size=batch_size, shuffle=False, **kwargs)

# initialize SVHN normalization and standardization (z-score) transforms pre-calculated mean and std values 
svhn_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])

# load in SVHN train and test data 
train_loader_SVHN = torch.utils.data.DataLoader(
    datasets.SVHN('./data', split='train', download=True,
                transform=svhn_norm), 
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader_SVHN = torch.utils.data.DataLoader(
    datasets.SVHN('./data', split='test', download=True, 
                transform=svhn_norm), 
    batch_size=batch_size, shuffle=False, **kwargs)

if dataset == 'MNIST':
    train_loader = train_loader_MNIST 
    test_loader = test_loader_MNIST 
    model = VAE_MNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
elif dataset == 'SVHN':
    train_loader = train_loader_SVHN
    test_loader = test_loader_SVHN
    model = VAE_SVHN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
elif dataset == 'BOTH':
    # load in MNIST dataset 
    train_loader1 = train_loader_MNIST 
    test_loader1 = test_loader_MNIST 
    model1 = VAE_MNIST().to(device)
    
    # load in SVHN dataset 
    train_loader2 = train_loader_SVHN 
    test_loader2 = test_loader_SVHN 
    model2 = VAE_SVHN().to(device)

    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

output_dir = os.makedirs(output_folder, exist_ok=True)

if __name__ == "__main__":
    total_train_loss = []
    total_bce_loss = []
    total_kld_loss = []
    total_test_loss = []
    total_bce_test_loss = []
    total_kld_test_loss = []
    for epoch in range(1, epochs + 1):
        # train(epoch, model, train_loader_MNIST, device, optimizer, log_interval)
        # test(epoch, model, test_loader_MNIST, device, batch_size, output_folder)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()  
        #     save_image(sample.view(64, 1, 28, 28),
        #                output_folder + '/sample_' + str(epoch) + '.png')
    # with torch.no_grad():
    #     sample = torch.randn(64, 20).to(device)
    #     sample = model.decode(sample).cpu()
    #     if dataset == 'MNIST':
    #         save_image(sample.view(64, 1, 28, 28),
    #             output_folder + '/sample_' + str(epoch) + '.png')
    #     elif dataset == 'SVHN':
    #         save_image(sample.view(64, 1, 32, 32), 
    #             output_folder + '/sample_' + str(epoch) + '.png')
        train_loss, bce_loss, kld_loss = train(epoch, model, train_loader, device, optimizer, log_interval)
        test_loss, bce_test_loss, kld_test_loss = test(epoch, model, test_loader, device, batch_size, output_folder)
        total_train_loss.append(train_loss)
        total_bce_loss.append(bce_loss)
        total_kld_loss.append(kld_loss)
        total_test_loss.append(test_loss)
        total_bce_test_loss.append(bce_test_loss)
        total_kld_test_loss.append(kld_test_loss)

    # plot loss figures 
    fix, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(total_train_loss, color = 'blue', label='Training loss')
    ax[0].plot(total_bce_loss, color='red', label='Training BCE loss')
    ax[0].plot(total_kld_loss, color='green', label='Training KLD loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')
    ax[0].legend()

    ax[1].plot(total_test_loss, color = 'blue', label='Testing loss')
    ax[1].plot(total_bce_test_loss, color='red', label='Testing BCE loss')
    ax[1].plot(total_kld_test_loss, color='green', label='Testing KLD loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Testing loss')
    ax[1].legend()

    plt.suptitle('Training and testing loss for the ' + dataset + ' VAE')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'train_test_loss.png'))

    # plot log loss figures 
    fix, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(np.log(total_train_loss), color = 'blue', label='Training loss')
    ax[0].plot(np.log(total_bce_loss), color='red', label='Training BCE loss')
    ax[0].plot(np.log(total_kld_loss), color='green', label='Training KLD loss')
    ax[0].set_xlabel('Epochs')    
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')
    ax[0].legend()

    ax[1].plot(np.log(total_test_loss), color = 'blue', label='Testing loss')
    ax[1].plot(np.log(total_bce_test_loss), color='red', label='Testing BCE loss')
    ax[1].plot(np.log(total_kld_test_loss), color='green', label='Testing KLD loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Testing loss')
    ax[1].legend()

    plt.suptitle('Training and testing loss on log scale for the ' + dataset + ' VAE')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'log_train_test_loss.png'))