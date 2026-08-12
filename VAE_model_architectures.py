import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch import nn, optim
from torch.autograd import Variable

# adjusted to be modular -> can accomodate an arbitrary number of hidden layers
class Autoencoder_CGMVAE(nn.Module):
    #C-GMVAE VAE with GMM prior and conditional layer
    def __init__(self, D_in, D_out, hidden_dims=[128, 64, 32], dim_code=10, num_gaussian = 4):
        # initialize posterior 
        self.qz_x = dist.Normal
        self.px_z = dist.Normal 
        
        #Encoder
        super(Autoencoder_CGMVAE, self).__init__()
        self.dim_code = dim_code
        self.num_gaussian = num_gaussian
        self.label = nn.Embedding(num_gaussian, 1)

        encoder_dims = [D_in + 3] + hidden_dims
        self.encoder1 = self.build_mlp_encoder(encoder_dims)

        last_hidden = hidden_dims[-1] if hidden_dims else D_in + 3
        self.mu = nn.Linear(last_hidden, num_gaussian * dim_code)
        self.logsigma = nn.Linear(last_hidden, num_gaussian * dim_code)

        decoder_dims = [dim_code + 1] + hidden_dims[::-1] + [D_out + 2]
        self.decoder = self.build_mlp_decoder(decoder_dims)

    def build_mlp_encoder(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def build_mlp_decoder(self, dims):
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers.append(nn.Sigmoid())
                layers.append(nn.BatchNorm1d(dims[i+1]))
            return nn.Sequential(*layers)
        
    def gaussian_sampler(self, mu, logsigma, y):
        y = y.long().view(-1)
        batch_size = y.shape[0]
        mu_k = mu[torch.arange(batch_size), y]
        sigma_k = torch.exp(0.5 * logsigma[torch.arange(batch_size), y])
        eps = torch.randn_like(sigma_k)
        z = mu_k + eps * sigma_k
        return mu_k, sigma_k, z
        
    def encode(self, x, y, cfm, lifespan):
        cfm = cfm.view(cfm.size(0),1)
        lifespan = lifespan.view(lifespan.size(0), 1)
        if x.dtype != cfm.dtype:
            cfm = cfm.to(x.dtype)
            lifespan = lifespan.to(x.dtype)
        x = torch.cat((x, cfm, lifespan), dim=1)
        y_emb = self.label(y)
        y_emb = y_emb.view(y_emb.size(0),1)
        x = torch.cat((x, y_emb), dim=1)
        x = self.encoder1(x)
        mu, logsigma = self.mu(x).view(-1, self.num_gaussian, self.dim_code), self.logsigma(x).view(-1, self.num_gaussian, self.dim_code) 
        mu, logsigma, z = self.gaussian_sampler(mu, logsigma, y)
        return mu, logsigma, z, y, cfm
    
    def decode(self, x, y):
        y_emb = self.label(y)
        y_emb = y_emb.view(y_emb.size(0),1)
        x = torch.cat((x, y_emb), dim=1)
        reconstruction = self.decoder(x)
        scale = torch.tensor(0.75).to(x.device)
        return reconstruction, scale
    
    def forward(self, x, y, cfm, lifespan):
        cfm = cfm.view(cfm.size(0),1)
        lifespan = lifespan.view(lifespan.size(0), 1)
        if x.dtype != cfm.dtype:
            cfm = cfm.to(x.dtype)
            lifespan = lifespan.to(x.dtype)
        x = torch.cat((x, cfm, lifespan), dim=1)
        y_emb = self.label(y)
        y_emb = y_emb.view(y_emb.size(0),1)
        x = torch.cat((x, y_emb), dim=1)
        x = self.encoder1(x)
        mu, logsigma = self.mu(x).view(-1, self.num_gaussian, self.dim_code), self.logsigma(x).view(-1, self.num_gaussian, self.dim_code) 
        mu, logsigma, z = self.gaussian_sampler(mu, logsigma, y)
        qz_x = self.qz_x(mu, logsigma)
        u = torch.cat((z, y_emb), dim=1)
        reconstruction = self.decoder(u)
        scale = torch.tensor(0.75).to(u.device)
        px_z = self.px_z(reconstruction, scale)
        #return mu, logsigma, z, reconstruction
        return qz_x, px_z, z

# non-modular original version

# class Autoencoder_CGMVAE(nn.Module):
#     #C-GMVAE VAE with GMM prior and conditional layer
#     def __init__(self, D_in, D_out, H1=128, H2=64, H3=32, dim_code=10, num_gaussian = 4):
#         # initialize posterior 
#         self.qz_x = dist.Normal
#         self.px_z = dist.Normal 
        
#         #Encoder
#         super(Autoencoder_CGMVAE, self).__init__()
#         self.dim_code = dim_code
#         self.num_gaussian = num_gaussian
#         self.label = nn.Embedding(num_gaussian, 1)
#         self.encoder1 = nn.Sequential(
#             nn.Linear(D_in+3, H1),
#             nn.BatchNorm1d(H1),
#             nn.Sigmoid(),
#             nn.Linear(H1,H2),
#             nn.BatchNorm1d(H2),
#             nn.Sigmoid(),
#             nn.Linear(H2,H3),
#             nn.BatchNorm1d(H3),
#             nn.Sigmoid())
    
#         self.mu = nn.Sequential(nn.Linear(H3, out_features = num_gaussian*dim_code))
                            
#         self.logsigma = nn.Sequential(nn.Linear(H3, out_features = num_gaussian*dim_code))
        
#         self.decoder = nn.Sequential(
#             nn.Linear(dim_code+1, H3),
#             nn.Sigmoid(),
#             nn.BatchNorm1d(H3),
#             nn.Linear(H3,H2),
#             nn.Sigmoid(),
#             nn.BatchNorm1d(H2),
#             nn.Linear(H2,H1),
#             nn.Sigmoid(),
#             nn.BatchNorm1d(H1),
#             nn.Linear(H1,D_out+2),
#             nn.Sigmoid(),
#             nn.BatchNorm1d(D_out+2))
        
#     def gaussian_sampler(self, mu, logsigma, y):
#         y = y.long().view(-1)
#         batch_size = y.shape[0]
#         mu_k = mu[torch.arange(batch_size), y]
#         sigma_k = torch.exp(0.5 * logsigma[torch.arange(batch_size), y])
#         eps = torch.randn_like(sigma_k)
#         z = mu_k + eps * sigma_k
#         return mu_k, sigma_k, z
        
#     def encode(self, x, y, cfm, lifespan):
#         cfm = cfm.view(cfm.size(0),1)
#         lifespan = lifespan.view(lifespan.size(0), 1)
#         if x.dtype != cfm.dtype:
#             cfm = cfm.to(x.dtype)
#             lifespan = lifespan.to(x.dtype)
#         x = torch.cat((x, cfm, lifespan), dim=1)
#         y_emb = self.label(y)
#         y_emb = y_emb.view(y_emb.size(0),1)
#         x = torch.cat((x, y_emb), dim=1)
#         x = self.encoder1(x)
#         mu, logsigma = self.mu(x).view(-1, self.num_gaussian, self.dim_code), self.logsigma(x).view(-1, self.num_gaussian, self.dim_code) 
#         mu, logsigma, z = self.gaussian_sampler(mu, logsigma, y)
#         return mu, logsigma, z, y, cfm
    
#     def decode(self, x, y):
#         y_emb = self.label(y)
#         y_emb = y_emb.view(y_emb.size(0),1)
#         x = torch.cat((x, y_emb), dim=1)
#         reconstruction = self.decoder(x)
#         scale = torch.tensor(0.75).to(x.device)
#         return reconstruction, scale
    
#     def forward(self, x, y, cfm, lifespan):
#         cfm = cfm.view(cfm.size(0),1)
#         lifespan = lifespan.view(lifespan.size(0), 1)
#         if x.dtype != cfm.dtype:
#             cfm = cfm.to(x.dtype)
#             lifespan = lifespan.to(x.dtype)
#         x = torch.cat((x, cfm, lifespan), dim=1)
#         y_emb = self.label(y)
#         y_emb = y_emb.view(y_emb.size(0),1)
#         x = torch.cat((x, y_emb), dim=1)
#         x = self.encoder1(x)
#         mu, logsigma = self.mu(x).view(-1, self.num_gaussian, self.dim_code), self.logsigma(x).view(-1, self.num_gaussian, self.dim_code) 
#         mu, logsigma, z = self.gaussian_sampler(mu, logsigma, y)
#         qz_x = self.qz_x(mu, logsigma)
#         u = torch.cat((z, y_emb), dim=1)
#         reconstruction = self.decoder(u)
#         scale = torch.tensor(0.75).to(u.device)
#         px_z = self.px_z(reconstruction, scale)
#         #return mu, logsigma, z, reconstruction
#         return qz_x, px_z, z
    
class Autoencoder_GMVAE(nn.Module):
    #VAE with GMM prior, with no conditional layer
    def __init__(self, D_in, D_out, H1=128, H2=64, H3=32, dim_code=10, num_gaussian = 1):
        #Encoder
        super(Autoencoder_GMVAE, self).__init__()
        self.dim_code = dim_code
        self.num_gaussian = num_gaussian
        self.label = nn.Embedding(num_gaussian, 1)
        self.encoder1 = nn.Sequential(
            nn.Linear(D_in+1, H1),
            nn.BatchNorm1d(H1),
            nn.Sigmoid(),
            nn.Linear(H1,H2),
            nn.BatchNorm1d(H2),
            nn.Sigmoid(),
            nn.Linear(H2,H3),
            nn.BatchNorm1d(H3),
            nn.Sigmoid())
    
        self.mu = nn.Sequential(nn.Linear(H3, out_features = dim_code))
                            
        self.logsigma = nn.Sequential(nn.Linear(H3, out_features = dim_code))
        
        self.decoder = nn.Sequential(
            nn.Linear(dim_code, H3),
            nn.Sigmoid(),
            nn.BatchNorm1d(H3),
            nn.Linear(H3,H2),
            nn.Sigmoid(),
            nn.BatchNorm1d(H2),
            nn.Linear(H2,H1),
            nn.Sigmoid(),
            nn.BatchNorm1d(H1),
            nn.Linear(H1,D_out+1),
            nn.Sigmoid(),
            nn.BatchNorm1d(D_out+1))
        
    def gaussian_sampler(self, mu, logsigma, y):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logsigma, z
        
    def encode(self, x, y, cfm):
        cfm = cfm.view(cfm.size(0),1)
        if x.dtype != cfm.dtype:
            cfm = cfm.to(x.dtype)
        x = torch.cat((x, cfm), dim=1)
        x = self.encoder1(x)
        mu, logsigma = self.mu(x), self.logsigma(x)
        mu, logsigma, z = self.gaussian_sampler(mu, logsigma, y)
        return mu, logsigma, z
    
    def decode(self, x, y):
        reconstruction = self.decoder(x)
        return reconstruction
    
    def forward(self, x, y, cfm):
        cfm = cfm.view(cfm.size(0),1)
        if x.dtype != cfm.dtype:
            cfm = cfm.to(x.dtype)
        x = torch.cat((x, cfm), dim=1)
        
        x = self.encoder1(x)
        mu, logsigma = self.mu(x), self.logsigma(x)
        mu, logsigma, z = self.gaussian_sampler(mu, logsigma, y)
        reconstruction = self.decoder(z)
        return mu, logsigma, z, reconstruction
    
class Autoencoder_VAE(nn.Module):
    #standard VAE
    def __init__(self, D_in, D_out, H1=128, H2=64, H3=32, dim_code=10):
        super(Autoencoder_VAE, self).__init__()
        self.dim_code = dim_code
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(D_in + 1, H1),  # + 1 for cfm 
            nn.BatchNorm1d(H1),
            nn.Sigmoid(),
            nn.Linear(H1, H2),
            nn.BatchNorm1d(H2),
            nn.Sigmoid(),
            nn.Linear(H2, H3),
            nn.BatchNorm1d(H3),
            nn.Sigmoid())
        
        self.mu = nn.Sequential(nn.Linear(H3, dim_code), nn.BatchNorm1d(dim_code)) # BatchNorm done to stablize training
        self.logvar = nn.Sequential(nn.Linear(H3, dim_code), nn.BatchNorm1d(dim_code))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(dim_code, H3),
            nn.Sigmoid(),
            nn.BatchNorm1d(H3),
            nn.Linear(H3, H2),
            nn.Sigmoid(),
            nn.BatchNorm1d(H2),
            nn.Linear(H2, H1),
            nn.Sigmoid(),
            nn.BatchNorm1d(H1),
            nn.Linear(H1, D_out + 1),  # Reconstructing x + cfm
            nn.Sigmoid(),
            nn.BatchNorm1d(D_out + 1))

    def gaussian_sampler(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, y, cfm):
        cfm = cfm.view(cfm.size(0),1)
        if x.dtype != cfm.dtype:
            cfm = cfm.to(x.dtype)
        x = torch.cat((x, cfm), dim=1)
        x = self.encoder(x)
        mu, logvar = self.mu(x),self.logvar(x)
        z = self.gaussian_sampler(mu, logvar)
        return mu, logvar, z

    def decode(self, x, y):
        return self.decoder(x)

    def forward(self, x, y, cfm):
        cfm = cfm.view(cfm.size(0),1)
        if x.dtype != cfm.dtype:
            cfm = cfm.to(x.dtype)
        x = torch.cat((x, cfm), dim=1)
        x = self.encoder(x)
        mu, logvar = self.mu(x),self.logvar(x)
        z = self.gaussian_sampler(mu, logvar)
        recon = self.decode(z, y)
        return mu, logvar, z, recon
    
class Autoencoder_CVAE(nn.Module):
    #standard VAE with conditional layer
    def __init__(self, D_in, D_out, H1=128, H2=64, H3=32, dim_code=10, num_gaussian=4):
        super(Autoencoder_CVAE, self).__init__()
        self.dim_code = dim_code
        self.label = nn.Embedding(num_gaussian, 1) 
        self.num_classes = num_classes

        self.encoder = nn.Sequential(nn.Linear(D_in + 2, H1), 
            nn.BatchNorm1d(H1),
            nn.Sigmoid(),
            nn.Linear(H1, H2),
            nn.BatchNorm1d(H2),
            nn.Sigmoid(),
            nn.Linear(H2, H3),
            nn.BatchNorm1d(H3),
            nn.Sigmoid())

        self.mu = nn.Sequential(nn.Linear(H3, dim_code), nn.BatchNorm1d(dim_code))
        self.logvar = nn.Sequential(nn.Linear(H3, dim_code), nn.BatchNorm1d(dim_code))

        self.decoder = nn.Sequential(
            nn.Linear(dim_code + 1, H3), 
            nn.Sigmoid(),
            nn.BatchNorm1d(H3),
            nn.Linear(H3, H2),
            nn.Sigmoid(),
            nn.BatchNorm1d(H2),
            nn.Linear(H2, H1),
            nn.Sigmoid(),
            nn.BatchNorm1d(H1),
            nn.Linear(H1, D_out + 1),  
            nn.Sigmoid(),
            nn.BatchNorm1d(D_out + 1))

    def gaussian_sampler(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self,  x, y, cfm):
        cfm = cfm.view(cfm.size(0),1) 
        if x.dtype != cfm.dtype:
            cfm = cfm.to(x.dtype)
        y_emb = self.label(y).view(-1, 1)
        x = torch.cat((x, cfm, y_emb), dim=1)  
        x = self.encoder(x)
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.gaussian_sampler(mu, logvar)
        return mu, logvar, z

    def decode(self, x, y):
        y_emb = self.label(y).view(-1, 1)
        u = torch.cat((x, y_emb), dim=1)
        recon = self.decoder(u)
        return recon

    def forward(self,  x, y, cfm):
        cfm = cfm.view(cfm.size(0),1)
        if x.dtype != cfm.dtype:
            cfm = cfm.to(x.dtype)
        y_emb = self.label(y).view(-1, 1)
        x = torch.cat((x, cfm, y_emb), dim=1)
        x = self.encoder(x)
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.gaussian_sampler(mu, logvar)
        u = torch.cat((z, y_emb), dim=1)
        recon = self.decoder(u)
        return mu, logvar, z, recon