import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class VAE_SVHN(nn.Module):
    def __init__(self):
        super(VAE_SVHN, self).__init__()

        # 3 is the number of channels in the image (3 because colored)
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 20, 4, stride=2, padding=0)
        self.conv5 = nn.Conv2d(128, 20, 4, stride=2, padding=0)

        # reshape? If not working 
        # need to calculate num_in_channels
        self.deconv1 = nn.ConvTranspose2d(20, 128, 4, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.conv4(x)
        #x = self.conv5(x)
        #print(self.conv4(x).shape)
        return self.conv4(x).squeeze(), self.conv5(x).squeeze()

    # reparameterization reconstructs the sample process by sampling 
    # from a standard normal dist (epsilon) and deterministically 
    # transformming this random epsilon by z = mu + sigma * epsilon
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = z.unsqueeze(2)
        z = z.unsqueeze(3)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        return torch.sigmoid(self.deconv4(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar