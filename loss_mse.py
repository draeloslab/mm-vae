# this file will calculate the overall loss for the MoE mmvae 
import torch
from torch.nn import functional as F
import torch.nn as nn

def loss_function(recon_x, x, mu, logvar):
    criterion_mse_x = nn.MSELoss()
    MSE = criterion_mse_x(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD, MSE, KLD




