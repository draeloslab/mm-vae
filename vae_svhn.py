import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.distributions as dist
from utils import Constants

class VAE_SVHN(nn.Module):
    def __init__(self, latent_dim, svhn_scale):
        super(VAE_SVHN, self).__init__()

        self.qz_x = dist.Laplace # initializing laplace posterior
        self.px_z = dist.Laplace # initializing laplace likelihood
        self.latent_dim = latent_dim
        self.svhn_scale = svhn_scale
        # 3 is the number of channels in the image (3 because colored)
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1) # 32 x 16 x 16
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1) # 64 x 8 x 8
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1) # 128 x 4 x 4 
        self.conv4 = nn.Conv2d(128, latent_dim, 4, stride=2, padding=0)
        self.conv5 = nn.Conv2d(128, latent_dim, 4, stride=2, padding=0)

        self.deconv1 = nn.ConvTranspose2d(latent_dim, 128, 4, stride=1, padding=0) # 128 x 4 x 4 
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) # 64 x 8 x 8
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1) # 32 x 16 x 16
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1) # 3 x 32 x 32

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.conv4(x)
        #x = self.conv5(x)
        #print(self.conv4(x).shape) 
        #log_scale = torch.clamp(self.conv5(x).squeeze(), min=Constants.logfloorc, max=Constants.logceilc)
        logvar = self.conv5(x).squeeze()
        return self.conv4(x).squeeze(), F.softmax(logvar, dim=-1) * logvar.size(-1) + Constants.eta

    # do reparameterization with rsample for laplacian initializations 
    # reparameterization reconstructs the sample process by sampling 
    # from a standard normal dist (epsilon) and deterministically 
    # transformming this random epsilon by z = mu + sigma * epsilon
    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps*std

    def decode(self, z):
        K, B = z.shape[:2]
        z = z.view(K*B, self.latent_dim)
        z = z.unsqueeze(2).unsqueeze(3)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        mean = torch.sigmoid(self.deconv4(z))
        mean = mean.clamp(Constants.eta, 1 - Constants.eta)
        mean = mean.view(K, B, *mean.shape[1:])
        scale = torch.tensor(0.75).to(z.device)

        return mean, scale
    
    def forward(self, x, K):
        mu, logvar = self.encode(x)

        scaled = torch.exp(logvar)
        #print(logvar)
        #print(scaled)
        qz_x = self.qz_x(mu, scaled) # posterior distribution qz_x

        zs = qz_x.rsample(torch.Size([K]))

        mean, scale = self.decode(zs)
        px_z = self.px_z(mean, scale) # laplace likelihood 

        return qz_x, px_z, zs