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

def osd_loss_from_latents(
    latent_vectors,
    bin_labels,
    strain_labels,
    residual_labels,
    latent_dim,
    temperature=0.1,
):
    """
    OSD loss that enforces proper gaussian ordering
    Use version 1 for benchmarking
    """
    coords = latent_vectors[:, :latent_dim]
    coords = torch.nan_to_num(coords, nan=0.0)
    # Version 0
    # projected scores with anchors from lowest bin -> highest bin - different, mmvae osd loss uses strains
    #bin_labels = bin_labels.float()
    #unique_bins = torch.unique(bin_labels).sort()[0]
    #low_bin = unique_bins[0]
    #high_bin = unique_bins[-1]
    #sus_anchor = coords[bin_labels == low_bin].mean(dim=0)
    #res_anchor = coords[bin_labels == high_bin].mean(dim=0)
    #line_vec = res_anchor - sus_anchor
    #line_len = torch.sqrt(torch.sum(line_vec ** 2) + 1e-8)
    #projected_scores = torch.sum((coords - sus_anchor) * line_vec, dim=1) / line_len
    
    # Version 1
    # projected scores - more like mmvae osd loss - calculated using strain anchors - didn't work on improving latent space
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
    mask_sus = torch.tensor(
        strain_labels_np == strain_sus_label,
        dtype=torch.bool,
        device=coords.device,
    )
    mask_res = torch.tensor(
        strain_labels_np == strain_res_label,
        dtype=torch.bool,
        device=coords.device,
    )
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
    rho_est = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
    )
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


def SupCon_loss(
    p: torch.Tensor,
    cfm: torch.Tensor,
    lambda_cl: float = 1.0,
    k: int = 8,
    T: float = 0.1,
    eps: float = 1e-8,
    symmetric_knn: bool = False,
) -> torch.Tensor:
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
    

def weighted_nt_xent(mu_k: torch.Tensor,
                         cfm: torch.Tensor,
                         lambda_cl = 1.0,
                         T: float = 0.1,
                         s: float = 0.5,
                         eps: float = 1e-8,
                         row_normalize: bool = True) -> torch.Tensor:
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

def weighted_nt_xent_qrt(mu: torch.Tensor,
                     qrt: torch.Tensor,
                     T: float = 0.1,
                     s: float = 0.5,
                     eps: float = 1e-12,
                     row_normalize: bool = True) -> torch.Tensor:
    """
    Weighted NT-Xent with soft positives based on QRT distance.
    row_normalize - if True, normalizes weights W_i,* to sum to 1 (excluding i=i)
    """
    B = mu.shape[0]
    qrt = qrt.view(-1)
    # normalize latent means for cosine similarity
    z = F.normalize(mu, p=2, dim=1) # (B, d)
    # cosine similarity / temperature
    S = (z @ z.T) / T  # (B, B)
    # mask diagonal
    diag = torch.eye(B, device=mu.device, dtype=torch.bool)
    S = S.masked_fill(diag, float("-inf"))
    # soft positive weights from qrt distance
    delta = (qrt.view(B, 1) - qrt.view(1, B)).abs()  # (B, B)
    W = torch.exp(-(delta ** 2) / (2.0 * (s ** 2) + eps))
    W = W.masked_fill(diag, 0.0)
    if row_normalize:
        W = W / (W.sum(dim=1, keepdim=True) + eps)
    # denominator: all non-self samples
    log_denom = torch.logsumexp(S, dim=1)  # (B,)
    # numerator: weighted soft positives
    log_numer = torch.logsumexp(S + torch.log(W + eps), dim=1)  # (B,)
    return -(log_numer - log_denom).mean()

def weighted_nt_xent_topk(mu_k: torch.Tensor,
                         cfm: torch.Tensor,
                         T: float = 0.2,
                         k: int = 8,
                         eps: float = 1e-12,
                         row_normalize: bool = True) -> torch.Tensor:
    B = mu_k.shape[0]
    cfm = cfm.view(-1)
    # Normalize embeddings
    z = torch.nn.functional.normalize(mu_k, dim=1)
    # Cosine similarity with temperature
    S = (z @ z.T) / T
    # Mask diagonal
    diag = torch.eye(B, device=mu_k.device, dtype=torch.bool)
    S = S.masked_fill(diag, float("-inf"))
    # Top-k positives based on CFM 
    delta = (cfm.view(B, 1) - cfm.view(1, B)).abs()
    delta_masked = delta.masked_fill(diag, float("inf"))
    idx = torch.topk(-delta_masked, k=k, dim=1).indices  # (B, k)
    W = torch.zeros_like(delta)
    # Soft weights within top-k 
    W_topk = torch.exp(-(delta.gather(1, idx) ** 2) / (2.0 * (0.2 ** 2) + eps))
    W.scatter_(1, idx, W_topk)
    if row_normalize:
        W = W / (W.sum(dim=1, keepdim=True) + eps)
    # InfoNCE 
    log_denom = torch.logsumexp(S, dim=1)
    logW = torch.log(W + eps)
    log_numer = torch.logsumexp(S + logW, dim=1)
    return -(log_numer - log_denom).mean()

def triplet_loss(
    mu_k: torch.Tensor,
    cfm: torch.Tensor,
    T: float = 1.0,
    margin: float = 0.1,
    K_near: int = 16,
    K_far: int = 64,
    P: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Triplet ranking loss 
    For each anchor i, we sample:
      - a near/positive sample j whose phenotype distance |cfm_i - cfm_j| is small
      - a far/negativw sample k whose phenotype distance |cfm_i - cfm_k| is large
    """
    B = mu_k.shape[0]
    cfm = cfm.view(-1)
    # Normalize embeddings for cosine similarity
    z = F.normalize(mu_k, dim=1)  # (B, d)
    # Similarity matrix (cosine), temperature-scaled
    S = (z @ z.T) / max(T, eps)   # (B, B)
    # Phenotype distance matrix
    delta = (cfm.view(B, 1) - cfm.view(1, B)).abs()  # (B, B)
    # Exclude self from candidate pools
    diag = torch.eye(B, device=mu_k.device, dtype=torch.bool)
    delta = delta.masked_fill(diag, float("inf"))
    S = S.masked_fill(diag, float("-inf"))
    # Near candidates: K_near smallest delta per row (positives)
    K_near_eff = min(K_near, B - 1)
    near_idx = torch.topk(-delta, k=K_near_eff, dim=1).indices  # (B, K_near_eff)
    # Far candidates: K_far largest delta per row (negatives)
    K_far_eff = min(K_far, B - 1)
    far_idx = torch.topk(delta, k=K_far_eff, dim=1).indices     # (B, K_far_eff)
    # Sample P positives and P negatives per anchor
    # (uniform over candidate sets)
    losses = []
    for _ in range(P):
        # Randomly pick one near and one far for each anchor
        j_pos = near_idx[torch.arange(B, device=mu_k.device),
                         torch.randint(0, K_near_eff, (B,), device=mu_k.device)]
        k_neg = far_idx[torch.arange(B, device=mu_k.device),
                        torch.randint(0, K_far_eff, (B,), device=mu_k.device)]
        s_pos = S[torch.arange(B, device=mu_k.device), j_pos]
        s_neg = S[torch.arange(B, device=mu_k.device), k_neg]
        # Hinge ranking loss: max(0, margin + s_neg - s_pos)
        losses.append(F.relu(margin + s_neg - s_pos))
    loss = torch.stack(losses, dim=0).mean()
    return loss
    

def get_gmmvaeloss(predicted_data, latent_vectors, true_data, ks_weight, cv_weight, data_loss_weight, gmm_centers,
                gmm_std):
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