# this file will calculate the overall loss for the MoE mmvae 
import torch
from torch.nn import functional as F

def loss_function_bce(recon_x, x, mu, logvar, beta):
    if x.shape[-1] == 28:
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    else:
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD, BCE, KLD