# this file will have the model architecture for a vanilla VAE with fully connected linear layers
# from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Latent dimension of 64
        self.fc1 = nn.Linear(5017, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc41 = nn.Linear(128, 64)
        self.fc42 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, 512)
        self.fc7 = nn.Linear(512, 2048)
        self.fc8 = nn.Linear(2048, 5017)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h1))
        h1 = F.relu(self.fc3(h1))
        return self.fc41(h1), self.fc42(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc5(z))
        h3 = F.relu(self.fc6(h3))
        h3 = F.relu(self.fc7(h3))
        # return torch.sigmoid(self.fc8(h3))
        # removing sigmoid
        return self.fc8(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 5017))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z