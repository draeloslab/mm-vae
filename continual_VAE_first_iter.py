from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from vanillaVAE import VAE
from latent_regularizer import (mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting, mean_squared_covariance_gmm)
from loss import get_gmmvaeloss, estimate_loss_coefficients





#My goal is to create a VAE that inherits from the original VAE that implements slowed learning rates from https://arxiv.org/pdf/1612.00796
#generated images from https://arxiv.org/pdf/1906.03288, and a way to test two tasks A and B in 4 different ways
#   0. Default, no split
#   1. A = 0-4, B = 5-9
#   2. A=even, B=odd
#   3. A = 0-4, B = 0-5
#   4. A = 0-1, B = 2-3, C = 4-5...
# I also want to create a way to traverse the latent space by taking the means of two numbers and going in a straight line, 
#then also by implementing stepwise density based trajectory
#also want a way to visualize clusters
#using accuracy to judge

class default_VAE(VAE):
    def __init__(self, latent_size):
        super().__init__()

        self.fc1 = nn.Linear(784, 100)
        self.fc21 = nn.Linear(100, latent_size) #for means
        self.fc22 = nn.Linear(100, latent_size) #for stds
        self.fc3 = nn.Linear(latent_size, 100) #the output just needs one because you sample and reconstruct
        self.fc4 = nn.Linear(100, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) #returns tensor with random normal samples in the same shape of std
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784)) #view to flatten images
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, split = False):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') #reconstruction loss (difference between input and reconstruciton)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KLD difference from approximating posterior
        if split:
            return BCE, KLD
    
        return BCE+KLD

    def train_epoch(self, epoch, train_loader, device, optimizer, log_interval):
        self.train()
        recon_loss = 0
        dkl_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self(data) #pytorch runs forward automatically
            recon_loss_tmp,  dkl_loss_tmp= self.loss_function(recon_batch, data, mu, logvar, split = True)
            loss = recon_loss_tmp + dkl_loss_tmp
            loss.backward()
            recon_loss += recon_loss_tmp.item()
            dkl_loss+= dkl_loss_tmp.item()
            optimizer.step()
            
            
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t recon Loss: {:.6f} \t DKL Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    recon_loss_tmp.item() / len(data), dkl_loss_tmp.item() / len(data)))

        avg_recon = recon_loss / len(train_loader.dataset)
        avg_dkl = dkl_loss / len(train_loader.dataset)
        print('====> Epoch: {} Avg recon loss: {:.4f} \t Avg DKL loss {:.4f}'.format(epoch, avg_recon, avg_dkl))
        return avg_recon, avg_dkl


    def test_epoch(self, epoch, test_loader, device, batch_size, model_name, split_name, task_name, evaluated_name, vis):
        self.eval()
        recon_loss = 0
        dkl_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar = self(data)
                recon_loss_tmp, dkl_loss_tmp = self.loss_function(recon_batch, data, mu, logvar, split = True)
                recon_loss += recon_loss_tmp.item()
                dkl_loss+= dkl_loss_tmp.item()
                if i == 0:
                    if vis:
                        n = min(data.size(0), 8)
                        comparison = torch.cat([data[:n],
                                            recon_batch.view(batch_size, 1, 28, 28)[:n]])
                        
                        save_image(comparison.cpu(),
                                
                            'results/' + model_name + '/split'+ split_name +'/recreation_' + 'task_' + task_name + '_evaluatedOn_' + evaluated_name + '_epoch_' +str(epoch) + '.png', nrow=n)

        recon_loss /= len(test_loader.dataset)
        dkl_loss /= len(test_loader.dataset)
        print('====> Test set recon loss: {:.4f}; Test set DKL loss {:.4f}'.format(recon_loss, dkl_loss))

        return recon_loss, dkl_loss
    


class mult_guassian(VAE):
    def __init__(self,latent_size = 9, gmm_std = 2.0, data_loss_weight = 1.0, batch_size = 128, k = 3): 
        #just what they used for now, need to fix batch size to take in args later
        super().__init__()
        gmm_centers = torch.randn(k, latent_size) * 1 #made this my own because no clue where 1-30 comes from, just randomized them pretty close to zero
        print(gmm_centers)
        self.register_buffer("gmm_centers", gmm_centers)
        ks_weight, cv_weight = estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, num_samples=100)
        print("ks_weight, cv_weight")
        print(ks_weight)
        print(cv_weight)
        cv_weight = 0.0 #I dont care about the cv_weight anyways, so just set to zero
        
        self.fc1 = nn.Linear(784, 100) 

        #for now 3 guassians, should prob be a hyperparameter
        self.fc21 = nn.Linear(100, latent_size) #for means
        self.fc22 = nn.Linear(100, latent_size) #for stds

        self.fc3 = nn.Linear(latent_size, 100) #the output just needs one because you sample and reconstruct
        self.fc4 = nn.Linear(100, 784)

        # the VAE_model_architecture needs this john, I would be lying if I said I understood it
        # I just want to use the loss as is, and I thing this will play a minor roll
        self.cfm_head = nn.Linear(100, 1)

        #store stuff from earlier
        self.gmm_std = gmm_std
        self.ks_weight = ks_weight
        self.cv_weight = cv_weight
        self.data_loss_weight = data_loss_weight

        self.latent_dim = latent_size

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) #returns tensor with random normal samples in the same shape of std
        return mu + eps*std

     
    def decode(self, z): #just changed to add the cfm_head
        h3 = F.relu(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(h3))
        recon_cfm = torch.sigmoid(self.cfm_head(h3))
        return torch.cat([recon_x, recon_cfm], dim=1)
   
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784)) #view to flatten images
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

 
    def loss_function(self, recon_full, x, mu, logvar, z, split=False):
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        # need cfm for  each batch, just set to 0
        x_flat = x.view(batch_size, -1)
        true_cfm = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        true_full = torch.cat([x_flat, true_cfm], dim=1)

        # Call loss
        BCE, KLD = get_gmmvaeloss( #KLD isn't the same, just call it that for ease
            predicted_data=recon_full,
            latent_vectors=z,
            true_data=true_full,
            ks_weight=self.ks_weight,
            cv_weight=self.cv_weight,
            data_loss_weight=self.data_loss_weight,
            gmm_centers=self.gmm_centers,
            gmm_std=self.gmm_std,
        )

        #not sure if this is actually 1-1 with reconstruction and KL_loss, but it serves to track
        # recon_loss = (loss_count + loss_cfm)* batch_size #this is the recon from the data and the cfm (set to 0 for now)
        # kl_loss = loss_KL * batch_size #this is the KL loss

        if split:
            return BCE, KLD
        return BCE + KLD

    def train_epoch(self, epoch, train_loader, device, optimizer, log_interval):
        self.train()
        recon_loss = 0
        dkl_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, z = self(data) #pytorch runs forward automatically
            recon_loss_tmp,  dkl_loss_tmp= self.loss_function(recon_batch, data, mu, logvar, z, split = True)
            loss = recon_loss_tmp + dkl_loss_tmp
            loss.backward()
            recon_loss += recon_loss_tmp.item()
            dkl_loss+= dkl_loss_tmp.item()
            optimizer.step()
            
            
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t recon Loss: {:.6f} \t DKL Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    recon_loss_tmp.item() / len(data), dkl_loss_tmp.item() / len(data)))

        avg_recon = recon_loss / len(train_loader.dataset)
        avg_dkl = dkl_loss / len(train_loader.dataset)
        print('====> Epoch: {} Avg recon loss: {:.4f} \t Avg DKL loss {:.4f}'.format(epoch, avg_recon, avg_dkl))
        return avg_recon, avg_dkl


    def test_epoch(self, epoch, test_loader, device, batch_size, model_name, split_name, task_name, evaluated_name, vis):
        self.eval()
        recon_loss = 0
        dkl_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar, z = self(data)
                recon_loss_tmp, dkl_loss_tmp = self.loss_function(recon_batch, data, mu, logvar, z, split = True)
                recon_loss += recon_loss_tmp.item()
                dkl_loss+= dkl_loss_tmp.item()
                if i == 0:
                    if vis:
                        n = min(data.size(0), 8)
                        recon_img = recon_batch[:, :784]
                        comparison = torch.cat([
                            data[:n],
                            recon_img.view(-1, 1, 28, 28)[:n]
                        ])
                        
                        save_image(comparison.cpu(),
                                
                            'results/' + model_name + '/split'+ split_name +'/recreation_' + 'task_' + task_name + '_evaluatedOn_' + evaluated_name + '_epoch_' +str(epoch) + '.png', nrow=n)

        recon_loss /= len(test_loader.dataset)
        dkl_loss /= len(test_loader.dataset)
        print('====> Test set recon loss: {:.4f}; Test set DKL loss {:.4f}'.format(recon_loss, dkl_loss))

        return recon_loss, dkl_loss
