import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist
from utils import Constants

# vae with conditional layer 
class CGMVAE(nn.Module):
    def __init__(self, input_dim, H1, H2, H3, latent_dim, scale, num_classes):
        super(CGMVAE, self).__init__()

        self.num_classes = num_classes
        self.label = nn.Embedding(num_classes, 1)
        self.qz_x = dist.Normal # initializing normal posterior
        self.px_z = dist.Normal # initializing normal likelihood
        self.qz_x_params = None 
        self.latent_dim = latent_dim
        self.scale = scale
        # encoder
        self.fc1 = nn.Linear(input_dim + 1, H1)
        self.bn1 = nn.BatchNorm1d(H1)
        self.fc2 = nn.Linear(H1, H2)
        self.bn2 = nn.BatchNorm1d(H2)
        self.fc3 = nn.Linear(H2, H3)
        self.bn3 = nn.BatchNorm1d(H3)
        self.fc41 = nn.Linear(H3, self.latent_dim * self.num_classes)
        self.fc42 = nn.Linear(H3, self.latent_dim * self.num_classes)

        # decoder
        self.fc5 = nn.Linear(self.latent_dim + 1, H3)
        self.bn5 = nn.BatchNorm1d(H3)
        self.fc6 = nn.Linear(H3, H2)
        self.bn6 = nn.BatchNorm1d(H2)
        self.fc7 = nn.Linear(H2, H1)
        self.bn7 = nn.BatchNorm1d(H1)
        self.fc8 = nn.Linear(H1, input_dim)
        self.bn8 = nn.BatchNorm1d(input_dim)

    def encode(self, x, y):
        y_emb = self.label(y).view(-1, 1)
        x = torch.cat((x, y_emb), dim=1)
        h1 = torch.sigmoid(self.bn1(self.fc1(x)))
        h2 = torch.sigmoid(self.bn2(self.fc2(h1)))
        h3 = torch.sigmoid(self.bn3(self.fc3(h2)))
        logvar = self.fc42(h3)
        #logvar = torch.clamp(logvar, min=-10, max=10)
        #return self.fc41(h3), F.softmax(logvar, dim=-1) * logvar.size(-1) + Constants.eta
        return self.fc41(h3), logvar

    def gaussian_sampler(self, mu, logsigma, y):
        y = y.long().view(-1)
        batch_size = y.shape[0]
        mu_k = mu[torch.arange(batch_size), y]  
        sigma_k = torch.exp(0.5 * logsigma[torch.arange(batch_size), y])
        eps = torch.randn_like(sigma_k)
        z = mu_k + eps * sigma_k
        return mu_k, sigma_k, z

    def decode(self, z, y):
        y_exp = self.label(y)
        z = torch.cat((z, y_exp), dim=1)
        h5 = self.bn5(torch.sigmoid(self.fc5(z)))
        h6 = self.bn6(torch.sigmoid(self.fc6(h5)))
        h7 = self.bn7(torch.sigmoid(self.fc7(h6)))
        mean = self.bn8(torch.sigmoid(self.fc8(h7)))
        #mean = mean.clamp(Constants.eta, 1 - Constants.eta)
        scale = torch.tensor(1).to(z.device)
        return mean, scale

    def forward(self, x, y, K):
        mu, std = self.encode(x, y)
        mu, std, z = self.gaussian_sampler(mu.view(-1, self.num_classes, self.latent_dim), std.view(-1, self.num_classes, self.latent_dim), y)
        self.qz_x_params = [mu, torch.exp(0.5 * std)]
        qz_x = self.qz_x(*self.qz_x_params)
        mean, scale = self.decode(z, y)
        px_z = self.px_z(mean, scale)

        return qz_x, px_z, z, y