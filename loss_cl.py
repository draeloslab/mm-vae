import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from latent_regularizer import*

def estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, num_samples=100):
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses = [], []
    for i in range(num_samples):
        z, _ = draw_gmm_samples(
            batch_size, gmm_centers, gmm_std)
        ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(
            embedding_matrix=z, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_loss = ks_loss.cpu().detach().numpy()
        cv_loss = mean_squared_covariance_gmm(
            embedding_matrix=z, gmm_centers=gmm_centers, gmm_std=gmm_std)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ks_loss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, cv_weight

def osd_loss_from_latents(latent_vectors, bin_labels, strain_labels, residual_labels, latent_dim, temperature=0.1):
    
    coords = latent_vectors[:, :latent_dim]
    coords = torch.nan_to_num(coords, nan=0.0)

    bin_labels = bin_labels.float().to(coords.device)

    if torch.is_tensor(residual_labels):
        residual_labels_np = residual_labels.detach().cpu().numpy()
    else:
        residual_labels_np = np.asarray(residual_labels)
    strain_labels_np = np.asarray(strain_labels)
    
    sorted_idx = np.argsort(residual_labels_np)

    unique_strains = []
    for s in strain_labels_np[sorted_idx]:
        if s not in unique_strains:
            unique_strains.append(s)

    strain_sus_label = unique_strains[0]
    strain_res_label = unique_strains[-1]

    mask_sus = torch.tensor(strain_labels_np == strain_sus_label, dtype=torch.bool, device=coords.device)

    mask_res = torch.tensor(strain_labels_np == strain_res_label, dtype=torch.bool, device=coords.device)

    sus_anchor = coords[mask_sus].mean(dim=0)
    res_anchor = coords[mask_res].mean(dim=0)

    line_vec = res_anchor - sus_anchor
    line_len = torch.sqrt(torch.sum(line_vec ** 2) + 1e-8)

    projected_scores = torch.sum((coords - sus_anchor) * line_vec, dim=1) / line_len

    unique_bins = torch.unique(bin_labels).sort()[0]

    # differentiable Spearman-like part using bin means - same as mmvae OSD loss
    bin_means = torch.stack([
        projected_scores[bin_labels == b].mean()
        for b in unique_bins
    ])

    vx = bin_means - bin_means.mean()
    vy = unique_bins - unique_bins.mean()

    rho_est = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)

    # differentiable Somers' D-like part - different, mmvae loss uses tanh
    # potentially can edit this to make it strain-wise
    soft_d_list = []

    for i in range(len(unique_bins) - 1):
        low_vals = projected_scores[bin_labels == unique_bins[i]]
        high_vals = projected_scores[bin_labels == unique_bins[i + 1]]

        pairwise_diff = high_vals[:, None] - low_vals[None, :]
        soft_auc = torch.sigmoid(pairwise_diff / temperature).mean()
        soft_d = 2.0 * soft_auc - 1.0

        soft_d_list.append(soft_d)

    somers_est = torch.stack(soft_d_list).mean()

    osd_value = rho_est * somers_est
    osd_loss = 1.0 - osd_value

    return osd_loss, osd_value, rho_est, somers_est


def SupCon_loss(p: torch.Tensor, cfm: torch.Tensor, lambda_cl: float = 1.0, k: int = 8, T: float = 0.1, eps: float = 1e-8, symmetric_knn: bool = False) -> torch.Tensor:
    """
    Supervised Contrastive-style loss using kNN positives defined w.r.t. CFM,
    and similarities computed on projection-head embeddings p.

    Returns:
        weighted_cl_loss, loss_cl
    """

    device = p.device
    B = p.size(0)

    if B < 2:
        loss_cl = p.sum() * 0.0
        weighted_cl_loss = lambda_cl * loss_cl
        return weighted_cl_loss, loss_cl

    if cfm.dim() > 1:
        cfm = cfm.view(-1)
    else:
        cfm = cfm.flatten()

    if cfm.size(0) != B:
        raise ValueError(f"cfm must have same batch size as p. Got {cfm.size(0)} vs {B}")

    # Normalize projection embeddings for cosine similarity
    p = F.normalize(p, p=2, dim=1, eps=eps)

    # Pairwise logits
    logits = torch.matmul(p, p.T) / T

    # Mask self-comparisons
    self_mask = torch.eye(B, device=device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, float("-inf"))

    # kNN in CFM space
    cfm_dist = torch.abs(cfm.unsqueeze(1) - cfm.unsqueeze(0))
    cfm_dist = cfm_dist.masked_fill(self_mask, float("inf"))

    k_eff = min(k, B - 1)
    knn_idx = torch.topk(cfm_dist, k=k_eff, dim=1, largest=False).indices

    pos_mask = torch.zeros((B, B), device=device, dtype=torch.bool)
    row_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, k_eff)
    pos_mask[row_idx, knn_idx] = True

    if symmetric_knn:
        pos_mask = pos_mask | pos_mask.T

    pos_mask = pos_mask & (~self_mask)

    # Log-probabilities over all non-self samples
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0

    if valid.sum() == 0:
        loss_cl = p.sum() * 0.0
        weighted_cl_loss = lambda_cl * loss_cl
        return weighted_cl_loss, loss_cl

    # Avoid 0 * (-inf) = nan
    masked_log_prob = log_prob.masked_fill(~pos_mask, 0.0)

    mean_log_prob_pos = masked_log_prob.sum(dim=1) / (pos_counts.float() + eps)
    loss_cl = -mean_log_prob_pos[valid].mean()
    weighted_cl_loss = lambda_cl * loss_cl

    return weighted_cl_loss, loss_cl
    

def get_gmmvaeloss(predicted_data, latent_vectors, true_data, ks_weight, cv_weight, data_loss_weight, gmm_centers, gmm_std):
    #calculate GMM VAE loss
    criterion_mse = nn.MSELoss(reduction='none')
    outputs_first_n = predicted_data[:, :]
    y_first_n = true_data[:, :]
    rec_data_loss = criterion_mse(outputs_first_n, y_first_n)
    loss_cfm = rec_data_loss.mean(dim=0)[-1]
    loss_count = rec_data_loss.mean(dim=0)[0:-1].mean()
    ks_loss = mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(latent_vectors, gmm_centers, gmm_std)
    cs_loss = mean_squared_covariance_gmm(latent_vectors, gmm_centers, gmm_std)
    weighted_ksloss = ks_weight * ks_loss
    weighted_cov_loss = cv_weight * cs_loss
    loss_recons = data_loss_weight * (loss_count + loss_cfm)
    loss_KL = weighted_ksloss + weighted_cov_loss
    losses =  loss_recons + loss_KL
    
    return losses, loss_count, loss_cfm, loss_KL, weighted_ksloss, weighted_cov_loss, loss_recons

def get_cvaeloss(predicted_data, true_data, mu, logvar, data_loss_weight=1.0, kl_loss_weight=1.0):
    criterion_mse = nn.MSELoss(reduction='none')

    rec_data_loss = criterion_mse(predicted_data, true_data)
    loss_cfm = rec_data_loss.mean(dim=0)[-1]
    loss_count = rec_data_loss.mean(dim=0)[:-1].mean()

    loss_recons = data_loss_weight * (loss_count + loss_cfm)
    
    # Standard, closed-form KL loss for Vanilla VAE where prior is N(0, I)
    loss_KL = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)).mean()

    losses = loss_recons + kl_loss_weight * loss_KL

    return losses, loss_count, loss_cfm, loss_KL

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