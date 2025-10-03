import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(35, 16)
        self.fc21 = nn.Linear(16, 2)
        self.fc22 = nn.Linear(16, 2)
        self.fc3 = nn.Linear(2, 16)
        self.fc4 = nn.Linear(16, 35)

    def encode(self, x):
        h1 = F.tanh(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    # reparameterization reconstructs the sample process by sampling 
    # from a standard normal dist (epsilon) and deterministically 
    # transforming this random epsilon by z = mu + sigma * epsilon
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.tanh(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 35))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z