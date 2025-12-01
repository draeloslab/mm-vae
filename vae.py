# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()

        latent_dim = 64

        self.conv1 = nn.Conv1d(1, 32, kernel_size=16, stride=4, padding=0)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=0)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=0)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=0)

        self.fc_mu = nn.Linear(627 * 256, latent_dim)
        self.fc_logvar = nn.Linear(627 * 256, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 627 * 256)

        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=0, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=0, output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2, padding=0, output_padding=1)
        self.deconv4 = nn.ConvTranspose1d(32, 1, kernel_size=16, stride=4, padding=0, output_padding=1)

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1, 627 * 256) 
        
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc_decode(z))

        h = h.view(-1, 256, 627)
        
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        
        return self.deconv4(h)

    def forward(self, x):
        x = x.view(-1, 1, 20068)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon.view(-1, 20068), mu, logvar