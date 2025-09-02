from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from vanillaVAE import VAE


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
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) #for means
        self.fc22 = nn.Linear(400, 20) #for stds
        self.fc3 = nn.Linear(20, 400) #the output just needs one because you sample and reconstruct
        self.fc4 = nn.Linear(400, 784)

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
    
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') #reconstruction loss (difference between input and reconstruciton)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KLD difference from approximating posterior

        return BCE + KLD

    def train_epoch(self, epoch, train_loader, device, optimizer, log_interval):
        self.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = self(data) #pytorch runs forward automatically
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))


    def test_epoch(self, epoch, test_loader, device, batch_size, model_name, split_name, task_name, evaluated_name):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar = self(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                            
                        'results/' + model_name + '/split'+ split_name +'/recreation_' + 'task_' + task_name + '_evaluatedOn_' + evaluated_name + '_epoch_' +str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
