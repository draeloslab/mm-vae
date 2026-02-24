# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist
from utils import Constants

class VAE_MNIST(nn.Module):
    def __init__(self, latent_dim, mnist_scale):
        super(VAE_MNIST, self).__init__()

        self.qz_x = dist.Laplace # initializing laplace posterior
        self.px_z = dist.Laplace # initializing laplace likelihood
        self.latent_dim = latent_dim
        self.mnist_scale = mnist_scale
        self.fc1 = nn.Linear(784, 400)
        #nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        self.fc21 = nn.Linear(400, self.latent_dim)
        self.fc22 = nn.Linear(400, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        logvar = self.fc22(h1)
        return self.fc21(h1), F.softmax(logvar, dim=-1) * logvar.size(-1) + Constants.eta

    # reparameterization done using rsample for laplacian 
    # # reparameterization reconstructs the sample process by sampling 
    # # from a standard normal dist (epsilon) and deterministically 
    # # transformming this random epsilon by z = mu + sigma * epsilon
    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps*std

    def decode(self, z):
        print(z.shape)
        h3 = F.relu(self.fc3(z))
        print(h3.shape)
        mean = torch.sigmoid(self.fc4(h3))
        print(mean.shape)
        exit()
        mean = mean.clamp(Constants.eta, 1 - Constants.eta)
        scale = torch.tensor(0.75).to(z.device)
        return mean, scale

    def forward(self, x, K):
        mu, logvar = self.encode(x.view(-1, 784))
        
        scaled = torch.exp(logvar)
        qz_x = self.qz_x(mu, scaled)

        zs = qz_x.rsample(torch.Size([K]))

        mean, scale = self.decode(zs)
        px_z = self.px_z(mean, scale)

        return qz_x, px_z, zs

