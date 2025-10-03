import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from latent_regularizer import*
import sys
import math 

def estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, num_samples=100):
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses = [], []
    z_list = []
    components = []
    for i in range(num_samples):
        z, comp  = draw_gmm_samples(
            batch_size, gmm_centers, gmm_std)
        ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(
             embedding_matrix=z, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_loss = ks_loss.cpu().detach().numpy()
        cv_loss = mean_squared_covariance_gmm(
            embedding_matrix=z, gmm_centers=gmm_centers, gmm_std=gmm_std)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ks_loss)
        cv_losses.append(cv_loss)

        z_list.append(z)
        components.extend(comp)

    samples = torch.vstack(z_list)

    ks_weight = 1 / np.mean(ks_losses)
    cv_weight = 1 / np.mean(cv_losses)

    return ks_weight, cv_weight, samples, components

def get_gmmvaeloss(predicted_data, latent_vectors, true_data, ks_weight, cv_weight, kl_weight, gmm_centers,
                gmm_std):
    #calculate GMM VAE loss
    criterion_mse = nn.MSELoss(reduction='none')
    rec_data_loss = criterion_mse(predicted_data, true_data)
    ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(latent_vectors, gmm_centers, gmm_std)
    cs_loss = mean_squared_covariance_gmm(latent_vectors, gmm_centers, gmm_std)
    weighted_ksloss = ks_weight * ks_loss
    weighted_cov_loss = cv_weight * cs_loss
    loss_recons = rec_data_loss.mean()
    loss_KL = weighted_ksloss + weighted_cov_loss
    losses =  loss_recons + loss_KL * kl_weight

    if math.isnan(losses):
        print('ks_loss: ' + str(ks_loss))
        print('cs_loss: ' + str(cs_loss))
        print('recon_loss' + str(rec_data_loss))
        print('weighted_ksloss: '+ str(weighted_ksloss))
        print('weighted_cov_loss' + str(weighted_cov_loss))
        print('loss_recons: ' + str(loss_recons))
        sys.exit()
    
    return losses, loss_recons, loss_KL
    #return losses, loss_recons, loss_KL, sorted_embeddings, gmm_centers_new
    #return losses, loss_recons, loss_KL

## don't need this for now
def get_vaeloss(recon_data, true_data, mu, logvar):
    #calculate loss of VAE with single gaussian
    criterion_mse_x = nn.MSELoss()
    criterion_mse_cfm = nn.MSELoss()
    
    # Separate output reconstructions
    recon_x = recon_data[:, :-1] 
    recon_cfm = recon_data[:, -1].view(-1, 1) 

    # Separate targets
    true_x = true_data[:, :-1]
    true_cfm = true_data[:, -1].view(-1, 1)

    # Reconstruction losses
    rec_x_loss = criterion_mse_x(recon_x, true_x)
    rec_cfm_loss = criterion_mse_cfm(recon_cfm, true_cfm)
    rec_total = rec_x_loss + rec_cfm_loss

    prior_mean = torch.tensor([ 6.4791529, 14.62007832, 4.0321147, -2.03427168, 0.64734837, -8.34248918, 13.05364219,
                               -2.34621337, 7.08688432, 10.69098765], device=mu.device, dtype=mu.dtype)
    prior_var = 36.0 # prior std = 6.0
    var = torch.exp(logvar) 

    # Standard, closed-form KL loss for Vanilla VAE where prior is N(prior_mean, prior_var)
    kl = 0.5 * torch.sum(var / prior_var + (mu - prior_mean)**2 / prior_var - 1 + torch.log(prior_var / var), dim=1)  # Sum over latent dimensions
    kl_loss = kl.mean() # Averaged over batch_size

    # Standard, closed-form KL loss for Vanilla VAE where prior is N(0, I)
    # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0) # Same as .mean()
    
    # Total loss
    total_loss = rec_total + kl_loss

    return total_loss, rec_x_loss, rec_cfm_loss, kl_loss