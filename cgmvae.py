# this file has the model architecture for the C-GMVAE model
# from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import sys

class CGMVAE(nn.Module):
    def __init__(self, num_gaussian=5, latent_dim=35):
        super(CGMVAE, self).__init__()

        self.num_gaussian = num_gaussian
        self.latent_dim = latent_dim
        self.label = nn.Embedding(num_gaussian, 1)

        self.fc1 = nn.Linear(35 + 1, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc21 = nn.Linear(16, num_gaussian*latent_dim)
        self.fc22 = nn.Linear(16, num_gaussian*latent_dim)
        self.fc3 = nn.Linear(latent_dim + 1, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 35)

    def encode(self, x, y):
        # adding the following lines for the conditional layer 
        y_emb = self.label(y)
        x = torch.cat((x, y_emb), dim=1)
        h1_linear = self.fc1(x)
        h1_bn = self.bn1(h1_linear)
        h1_act = F.tanh(h1_bn)
        mu = self.fc21(h1_act).view(-1, self.num_gaussian, self.latent_dim)
        logvar = self.fc22(h1_act).view(-1, self.num_gaussian, self.latent_dim)
        return mu, logvar

    # reparameterization reconstructs the sample process by sampling 
    # from a standard normal dist (epsilon) and deterministically 
    # transformming this random epsilon by z = mu + sigma * epsilon
    def reparameterize(self, mu, logvar, y): 
        batch_size = y.shape[0]
        mu_k = mu[torch.arange(batch_size), y]
        sigma_k = torch.exp(0.5 * logvar[torch.arange(batch_size), y])
        eps = torch.randn_like(sigma_k)
        return mu_k, sigma_k, mu_k + eps * sigma_k

    def decode(self, z, y):
        y_emb = self.label(y)
        z = torch.cat((z, y_emb), dim=1)
        h3_linear = self.fc3(z)
        h3_bn = self.bn3(h3_linear)
        h3_act = F.tanh(h3_bn)
        return self.fc4(h3_act)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        mu, logvar, z = self.reparameterize(mu, logvar, y)
        return self.decode(z, y), mu, logvar, z

