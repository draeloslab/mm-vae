# this file will have the model architecture for the MoE mmVAE model 
# from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist
from utils import Constants

class VAE(nn.Module):
    def __init__(self, input_dim, H1, H2, H3, latent_dim, scale):
        super(VAE, self).__init__()

        # self.qz_x = dist.Laplace # initializing laplace posterior
        # self.px_z = dist.Laplace # initializing laplace likelihood
        self.qz_x = dist.Normal # initializing normal posterior
        self.px_z = dist.Normal # initializing normal likelihood
        self.qz_x_params = None 
        self.latent_dim = latent_dim
        self.scale = scale
        # encoder
        self.fc1 = nn.Linear(input_dim, H1)
        self.bn1 = nn.BatchNorm1d(H1)
        self.fc2 = nn.Linear(H1, H2)
        self.bn2 = nn.BatchNorm1d(H2)
        self.fc3 = nn.Linear(H2, H3)
        self.bn3 = nn.BatchNorm1d(H3)
        self.fc41 = nn.Linear(H3, self.latent_dim)
        self.fc42 = nn.Linear(H3, self.latent_dim)
        self.bn41 = nn.BatchNorm1d(self.latent_dim)
        self.bn42 = nn.BatchNorm1d(self.latent_dim)

        # decoder
        self.fc5 = nn.Linear(self.latent_dim, H3)
        self.bn5 = nn.BatchNorm1d(H3)
        self.fc6 = nn.Linear(H3, H2)
        self.bn6 = nn.BatchNorm1d(H2)
        self.fc7 = nn.Linear(H2, H1)
        self.bn7 = nn.BatchNorm1d(H1)
        self.fc8 = nn.Linear(H1, input_dim)
        self.bn8 = nn.BatchNorm1d(input_dim)

    def encode(self, x):
        h1 = torch.sigmoid(self.bn1(self.fc1(x)))
        h2 = torch.sigmoid(self.bn2(self.fc2(h1)))
        h3 = torch.sigmoid(self.bn3(self.fc3(h2)))
        # h1 = torch.relu(self.fc1(x))
        # h2 = torch.relu(self.fc2(h1))
        # h3 = torch.relu(self.fc3(h2))
        logvar = self.fc42(h3)
        return self.bn41(self.fc41(h3)), F.softmax(logvar, dim=-1) * logvar.size(-1) + Constants.eta
        #return self.bn41(self.fc41(h3)), self.bn42(logvar)

    def decode(self, z):
        K, B, D = z.shape
        x = z.view(-1, D)
        h5 = torch.sigmoid(self.bn5(self.fc5(x))) 
        h6 = torch.sigmoid(self.bn6(self.fc6(h5)))
        h7 = torch.sigmoid(self.bn7(self.fc7(h6)))
        # h5 = torch.relu(self.fc5(z))
        # h6 = torch.relu(self.fc6(h5))
        # h7 = torch.relu(self.fc7(h6))
        mean = self.bn8(self.fc8(h7))
        #mean = mean.clamp(Constants.eta, 1 - Constants.eta)
        scale = torch.tensor(0.75).to(z.device)
        return mean.view(K, B, -1), scale

    def forward(self, x, K):
        mu, logvar = self.encode(x)
        #std = torch.exp(0.5 * logvar)
        self.qz_x_params = [mu, logvar]
        qz_x = self.qz_x(*self.qz_x_params)

        zs = qz_x.rsample(torch.Size([K]))
        mean, scale = self.decode(zs)
        px_z = self.px_z(mean, scale)

        return qz_x, px_z, zs