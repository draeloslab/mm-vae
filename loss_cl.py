import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import math
from latent_regularizer import*

# Edited get_gmmvaeloss to also include a regularization constant for KL loss (lambda_kl) - on September 14, 2026

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

def pearson_corr_torch(x, y, eps=1e-9):
    """
    Differentiable Pearson correlation for 1D tensors.
    """
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)) + eps
    return torch.sum(x * y) / denom


def projected_scores_from_strain_anchors(coords, strain_labels, residual_labels, eps=1e-9):
    """
    Project latent coordinates onto susceptible -> resilient strain axis.
    coords: latent variables [B, latent_dim]
    """

    device = coords.device
    if torch.is_tensor(residual_labels):
        residual_np = residual_labels.detach().cpu().numpy()
    else:
        residual_np = np.asarray(residual_labels)
    if torch.is_tensor(strain_labels):
        strain_np = strain_labels.detach().cpu().numpy()
    else:
        strain_np = np.asarray(strain_labels)
    sorted_idx = np.argsort(residual_np)
    unique_strains = []
    for s in strain_np[sorted_idx]:
        if s not in unique_strains:
            unique_strains.append(s)
    # Determine endpoint strains based on their residual values
    strain_sus_label = unique_strains[0]
    strain_res_label = unique_strains[-1]
    mask_sus = torch.tensor(strain_np == strain_sus_label, dtype=torch.bool, device=device)
    mask_res = torch.tensor(strain_np == strain_res_label, dtype=torch.bool, device=device)
    # Create anchor points (z) by taking mean
    sus_anchor = coords[mask_sus].mean(dim=0)
    res_anchor = coords[mask_res].mean(dim=0)
    # Create unit direction vector from sus -> res
    line_vec = res_anchor - sus_anchor
    line_len = torch.sqrt(torch.sum(line_vec ** 2) + eps) # Same as np.linalgnorm (L2 norm) while handling for zero division
    projected_scores = torch.sum((coords - sus_anchor) * line_vec, dim=1) / line_len # (X - X_sus) * u_hate
    return projected_scores, sus_anchor, res_anchor

def distance_matching_loss_intrabin(latent_vectors, phenotype_labels, group_labels, latent_dim, min_group_size=4, eps=1e-9):
    """
    Within-Gaussian latent-CFM distance matching loss - to quantify local smoothness within each bin with respect to CFM using a distance correlation matching loss.
    For each Gaussian/bin group, computes the Pearson correlation between pairwise Euclidean latent distances and the corresponding pairwise absolute CFM differences, encouraging cells with similar CFM to lie closer together and cells with more different CFM to lie farther apart.:
    Corr(pairwise latent distance, pairwise phenotype distance)
    """

    coords = latent_vectors[:, :latent_dim] # Latent points: [B, latent_dim]
    coords = torch.nan_to_num(coords, nan=0.0)
    device = coords.device
    phenotype_labels = phenotype_labels.float().to(device) # CFM: [B]
    group_labels = group_labels.to(device) # Bins
    unique_groups = torch.unique(group_labels).sort()[0]
    losses = []
    corrs = []
    for g in unique_groups:
        mask = group_labels == g
        if mask.sum() < min_group_size: # Only for 4 bins
            continue
        z_g = coords[mask]
        y_g = phenotype_labels[mask]
        Dz = torch.cdist(z_g, z_g, p=2) # Computes the batched p-norm distance between each pair of the two collections of row vectors (p = 2 makes it L2)
        Dy = torch.abs(y_g[:, None] - y_g[None, :]) # Absolute differences between CFM samples
        # Keep only the upper triangle of the pairwise distance matrix (i < j) excluding self-pairs and avoiding counting each cell pair twice.
        tri_mask = torch.triu(torch.ones_like(Dz, dtype=torch.bool), diagonal=1)
        Dz_vec = Dz[tri_mask]
        Dy_vec = Dy[tri_mask]
        # Skip bins where either the latent pairwise distances or CFM pairwise differences have essentially no variation, since Pearson correlation is undefined when one variable has near-zero standard deviation.
        if torch.std(Dz_vec) < eps or torch.std(Dy_vec) < eps:
            continue
        corr = pearson_corr_torch(Dz_vec, Dy_vec, eps=eps)
        corrs.append(corr)
        losses.append(1.0 - corr)
    if len(losses) == 0:
        zero = coords.sum() * 0.0
        return zero, zero
    # Average the phenotype-smoothness loss equally across valid bins
    smooth_loss = torch.stack(losses).mean()
    mean_corr = torch.stack(corrs).mean()
    return smooth_loss, mean_corr

def distance_matching_loss_interbin(latent_vectors, phenotype_labels, group_labels, latent_dim, eps=1e-9):
    """
    Global latent-CFM distance matching loss - Computes the Pearson correlation between pairwise Euclidean latent distances and corresponding pairwise absolute CFM differences across all cells in the batch and across all bins. 
    This encourages cells with similar CFM to lie closer together and cells with more different CFM to lie farther apart.
    Corr(pairwise latent distance, pairwise absolute CFM difference)
    """
    coords = latent_vectors[:, :latent_dim] # Latent points: [B, latent_dim]
    coords = torch.nan_to_num(coords, nan=0.0)
    device = coords.device
    phenotype_labels = phenotype_labels.float().to(device) # CFM: [B]
    # Global pairwise latent differences (Euclidean distances)
    Dz = torch.cdist(coords, coords, p=2)
    # Global pairwise CFM differences (Euclidean distances)
    Dy = torch.abs(phenotype_labels[:, None] - phenotype_labels[None, :]) # [B, B]
    # To keep only unique pairs of cells:
    # Upper triangle: i < j
    tri_mask = torch.triu(torch.ones_like(Dz, dtype=torch.bool), diagonal=1)
    Dz_vec = Dz[tri_mask]
    Dy_vec = Dy[tri_mask]
    # Skip bins where either the latent pairwise distances or CFM pairwise differences have essentially no variation, since Pearson correlation is undefined when one variable has near-zero standard deviation.
    if (torch.std(Dz_vec) < eps or torch.std(Dy_vec) < eps):
        zero = coords.sum() * 0.0
        return zero, zero
    # Global latent-CFM difference
    corr = pearson_corr_torch(Dz_vec, Dy_vec, eps=eps)
    smooth_loss = 1.0 - corr
    return smooth_loss, corr

def gaussian_ordering_loss(latent_vectors, bin_labels, strain_labels, residual_labels, latent_dim, eps=1e-9):
    """
    Between-Gaussian/bin ordering loss - this is a Spearman's-like metric (similar to metric from C-GMVAE paper) to quantify ordering.
    Projects all latent points onto susceptible -> resilient strain axis
    Then computes Pearson correlation between projected bin means and bin order.
    """

    coords = latent_vectors[:, :latent_dim]
    coords = torch.nan_to_num(coords, nan=0.0)
    device = coords.device
    bin_labels = bin_labels.float().to(device)
    projected_scores, _, _ = projected_scores_from_strain_anchors(coords=coords, strain_labels=strain_labels, residual_labels=residual_labels, eps=eps)
    unique_bins = torch.unique(bin_labels).sort()[0]
    bin_means = []
    for b in unique_bins:
        mask = bin_labels == b
        if mask.sum() == 0:
            continue
        bin_means.append(projected_scores[mask].mean())
    bin_means = torch.stack(bin_means)
    used_bins = unique_bins[: len(bin_means)].float().to(device)
    rho = pearson_corr_torch(bin_means, used_bins, eps=eps)
    order_loss = 1.0 - rho
    return order_loss, rho

def gaussian_overlap_loss_sigmoid(latent_vectors, bin_labels, strain_labels, residual_labels, latent_dim, temperature=0.1, target_soft_d=0.5, eps=1e-9):
    """
    Adjacent-Gaussian overlap/separation loss - this is a Somer's D-like (similar to metric from C-GMVAE paper) soft-AUD metric to quantify separation
    Projects latent points onto susceptible -> resilient strain axis
    This loss function uses sigmoid activation function.
    For each adjacent bin pair, computes:
        soft_auc = mean sigmoid((s_high - s_low) / T)
        soft_d = 2 * soft_auc - 1
    Instead of maximizing soft_d toward 1 (complete separation), this version optimizes toward a desired target_soft_d
        Raw loss = 1 - mean soft_d
        Loss = (mean_soft_d - target_soft_d)^2
    """
    coords = latent_vectors[:, :latent_dim]
    coords = torch.nan_to_num(coords, nan=0.0)
    device = coords.device
    bin_labels = bin_labels.float().to(device)
    projected_scores, _, _ = projected_scores_from_strain_anchors(coords=coords, strain_labels=strain_labels, residual_labels=residual_labels, eps=eps)
    unique_bins = torch.unique(bin_labels).sort()[0]
    soft_d_list = []
    for i in range(len(unique_bins) - 1):
        low_bin = unique_bins[i]
        high_bin = unique_bins[i + 1]
        low_vals = projected_scores[bin_labels == low_bin]
        high_vals = projected_scores[bin_labels == high_bin]
        if low_vals.numel() == 0 or high_vals.numel() == 0:
            continue
        pairwise_diff = high_vals[:, None] - low_vals[None, :]
        soft_auc = torch.sigmoid(pairwise_diff / temperature).mean()
        soft_d = 2.0 * soft_auc - 1.0
        soft_d_list.append(soft_d)
    if len(soft_d_list) == 0:
        zero = coords.sum() * 0.0
        return zero, zero, zero
    mean_soft_d = torch.stack(soft_d_list).mean()
    raw_overlap_loss = 1.0 - mean_soft_d
    # Actual optimization objective - keep adjacent bins at a desired amount of separation/overlap (chosen from lookup table)
    target = torch.as_tensor(target_soft_d, dtype=mean_soft_d.dtype, device=device)
    # Square the deviation so values above or below the target are both penalized, making the target the minimum of the loss
    overlap_loss = (mean_soft_d - target) ** 2
    return overlap_loss, mean_soft_d, raw_overlap_loss

def gaussian_overlap_loss_tanh(latent_vectors, bin_labels, strain_labels, residual_labels, latent_dim, temperature=0.1, target_soft_d=0.5, eps=1e-9):
    """
    Adjacent-Gaussian overlap/separation loss - this is a Somer's D-like (similar to metric from C-GMVAE paper) soft-AUD metric to quantify separation
    Projects latent points onto susceptible -> resilient strain axis
    This loss function uses tanh activation function.
    For each adjacent bin pair, computes:
        diff = mean_high - mean_low
        soft_d = tanh(diff)
    The loss directly minimizes:
        Loss = 1 - mean soft_d
    """
    coords = latent_vectors[:, :latent_dim]
    coords = torch.nan_to_num(coords, nan=0.0)
    device = coords.device
    bin_labels = bin_labels.float().to(device)
    projected_scores, _, _ = projected_scores_from_strain_anchors(coords=coords, strain_labels=strain_labels, residual_labels=residual_labels, eps=eps)
    unique_bins = torch.unique(bin_labels).sort()[0]
    bin_means = []
    for b in unique_bins:
        vals = projected_scores[bin_labels == b]
        if vals.numel() > 0:
            bin_means.append(vals.mean())
    if len(bin_means) < 2:
        zero = coords.sum() * 0.0
        return zero, zero, zero
    bin_means = torch.stack(bin_means) 
    diff = bin_means[1:] - bin_means[:-1]
    mean_soft_d = torch.tanh(diff).mean()
    raw_overlap_loss = 1.0 - mean_soft_d
    overlap_loss = raw_overlap_loss
    return overlap_loss, mean_soft_d, raw_overlap_loss

def SupCon_loss(p: torch.Tensor, cfm: torch.Tensor, lambda_cl: float = 1.0, k: int = 8, T: float = 0.1, eps: float = 1e-9, symmetric_knn: bool = False) -> torch.Tensor:
    """
    Supervised Contrastive-style loss using kNN-based positives defined w.r.t. CFM,
    and similarities computed on projection-head embeddings p.
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
        return weighted_cl_loss, loss_c
    # Avoid 0 * (-inf) = nan
    masked_log_prob = log_prob.masked_fill(~pos_mask, 0.0)
    mean_log_prob_pos = masked_log_prob.sum(dim=1) / (pos_counts.float() + eps)
    loss_cl = -mean_log_prob_pos[valid].mean()
    weighted_cl_loss = lambda_cl * loss_cl
    return weighted_cl_loss, loss_cl
    
def weighted_nt_xent(mu_k: torch.Tensor, cfm: torch.Tensor, lambda_cl = 1.0, T: float = 0.1, s: float = 0.5, eps: float = 1e-9, row_normalize: bool = True) -> torch.Tensor:
    """
    Weighted NT-Xent with soft positives based on CFM distance.
    Uses mu_k (latent means) and cosine similarity. Weighted using a Gaussian kernel.
    row_normalize - if True, normalizes weights W_i,* to sum to 1 (excluding i=i)
    """
    B = mu_k.shape[0]
    cfm = cfm.view(-1)
    z = F.normalize(mu_k, p=2, dim=1, eps=eps) 
    S = (z @ z.T) / T
    diag = torch.eye(B, device=mu_k.device, dtype=torch.bool)
    S = S.masked_fill(diag, -1e9)
    delta = (cfm.view(B, 1) - cfm.view(1, B)).abs() 
    W = torch.exp(-(delta ** 2) / (2.0 * (s ** 2) + eps)) # Gaussian Kernel
    W = W.masked_fill(diag, 0.0)
    if row_normalize:
        W = W / (W.sum(dim=1, keepdim=True) + eps)
    # denom_i = sum_{k != i} exp(S_ik)
    log_denom = torch.logsumexp(S, dim=1) 
    # numer_i = sum_{j != i} W_ij * exp(S_ij)
    logW = torch.log(W + eps)
    log_numer = torch.logsumexp(S + logW, dim=1)
    loss_cl = -(log_numer - log_denom).mean()
    weighted_cl_loss = lambda_cl * loss_cl
    return weighted_cl_loss, loss_cl

def get_gmmvaeloss(predicted_data, latent_vectors, true_data, ks_weight, cv_weight, data_loss_weight, lambda_kl, gmm_centers, gmm_std):
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
    loss_KL = lambda_kl * (weighted_ksloss + weighted_cov_loss)
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