import os
import math
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from latent_regularizer import*
from scipy.stats import spearmanr
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity


# constants defined in the mmvae paper (for clamping values)
class Constants(object):
    logceilc = 88
    logfloorc = -104
    eta = 1e-6

def unnormalize(x, mean, std):
    if (isinstance(mean, float)):
        new_mean = -mean/std
        new_std = 1/std
    else:
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
    unnorm = transforms.Normalize(new_mean, new_std)
    #return unnorm(x) * 255
    return unnorm(x)

def log_mean_exp(value, dim=0, keepdim=False):
    # calculate log(mean(exp(value)))
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

def resize_img(img, refsize):
    # adjusts size of generated images to match reference size (which is the size of the svhn data)
    batch_size = img.size(0)
    img = img.view(batch_size, 1, 28, 28)
    img_padded = F.pad(img, (2, 2, 2, 2))
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)

def save_latent_space(model, epoch, data_loader, device, output_path, num_modalities):
    model.eval()

    latent_mus = []

    for _, data in enumerate(data_loader):
        #print(data.shape)
        x_data = [data[m][0].to(device) for m in range(num_modalities)]
        labels = data[0][1].cpu().numpy()
        
        for m in range(num_modalities):
            
            mu = model.get_latent_space(x_data, encoder=m)
            mu = mu.cpu().numpy()

            latent_dim = mu.shape[1]

            latent_space = pd.DataFrame(mu, columns=[f'LV{i+1}' for i in range(latent_dim)])

            latent_space['labels'] = labels
            latent_space['encoder'] = m

            latent_mus.append(latent_space)

    full_latent_df = pd.concat(latent_mus, ignore_index=True)
    full_latent_df.to_csv(os.path.join(output_path, 'latent_space_epoch' + str(epoch) + '.csv'), index=False)
    print('Latent space at epoch '  + str(epoch) + ' saved.')

# calculate ks and cv weights 
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

def standardizer(input_array):
    mean = np.mean(input_array)
    std = np.std(input_array)
    return (input_array - mean)/std

def numpyToTensor(x):
    return torch.from_numpy(x)

# functions to compute phenotypic projection 
def line_coordinates_euclidean(A, B, P):
    A = np.array(A)
    B = np.array(B)
    P = np.array(P)

    u_hat = (B - A) / np.linalg.norm(B - A)  # Unit direction vector from A to B
    s = np.dot(P - A, u_hat)                # Signed distance from A
    P_proj = A + s * u_hat                  # Projected point on the line

    return P_proj, s

def project_row(row, coord_cols, sus, res):
    coords = row[coord_cols].values
    proj, dist = line_coordinates_euclidean(sus, res, coords)
    return pd.Series({'Projected': proj,'Signed_Distance': dist})

# functions for calculating OSD 
def compute_kde_peak(values, bandwidth=0.1, grid_size=1000):
    """Estimate KDE peak (mode) location."""
    values = values[:, None]
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)
    grid = np.linspace(values.min(), values.max(), grid_size)[:, None]
    log_dens = kde.score_samples(grid)
    peak = grid[np.argmax(log_dens)][0]
    return peak

def compute_osd(df, projection_col='PP', bin_col='bin', bandwidth=0.1):
    # Step 1: Compute KDE peaks for each bin
    bins_sorted = sorted(df[bin_col].unique())
    peak_locs = []
    for b in bins_sorted:
        values = df[df[bin_col] == b][projection_col].values
        if len(values) > 1:
            peak = compute_kde_peak(values, bandwidth=bandwidth)
        else:
            peak = np.nan
        peak_locs.append(peak)

    # Step 2: Spearman correlation
    rho, _ = spearmanr(bins_sorted, peak_locs)

    # Step 3: Pairwise adjacent AUCs → Somers' D
    D_list = []
    for i in range(len(bins_sorted) - 1):
        b_low, b_high = bins_sorted[i], bins_sorted[i + 1]
        df_pair = df[df[bin_col].isin([b_low, b_high])].copy()
        df_pair['label'] = (df_pair[bin_col] == b_high).astype(int)
        auc = roc_auc_score(df_pair['label'], df_pair[projection_col])
        D = 2 * auc - 1
        D_list.append(D)

    D_avg = np.mean(D_list)
    osd = rho * D_avg

    return {
        'rho_spearman': rho,
        'somers_d_adjacent': D_avg,
        'osd': osd,
        'peak_locations': dict(zip(bins_sorted, peak_locs)),
        'pairwise_D': D_list,
    }


# writing a function that takes in LV and returns the OSD values
def get_osd_from_latents(zss, label_col, residual_col, strain_col, latent_dim): 
    lv1 = zss[0].detach().cpu().numpy()
    df1 = pd.DataFrame(lv1, columns=[f'LV{i+1}' for i in range(lv1.shape[1])])
    coord_cols = df1.columns[:latent_dim]
    df1['strain'] = strain_col
    df1['14-6-residual'] = residual_col
    df1['bin'] = label_col
    df1[coord_cols] = df1[coord_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    lv2 = zss[1].detach().cpu().numpy()
    df2 = pd.DataFrame(lv2, columns=[f'LV{i+1}' for i in range(lv2.shape[1])])
    df2['strain'] = strain_col
    df2['14-6-residual'] = residual_col
    df2['bin'] = label_col
    df2[coord_cols] = df2[coord_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    df = pd.concat([df1, df2])
    df = df.reset_index(drop=True)

    strain_order = (df.sort_values('14-6-residual', ascending=True)['strain'].unique())
    strain_sus = strain_order[0]
    strain_res = strain_order[-1]

    sus = (df[df['strain'] == strain_sus][coord_cols].apply(np.mean, axis=0).values)
    res = (df[df['strain'] == strain_res][coord_cols].apply(np.mean, axis=0).values)

    proj = df.apply(lambda x: project_row(x, coord_cols, sus, res), axis=1)
    df = pd.concat([
        df.reset_index(drop=True),
        pd.DataFrame(proj['Projected'].tolist(), columns=[f'Proj_{i+1}' for i in range(10)]),
        proj[['Signed_Distance']]], axis=1)

    sus_cen = (df[df['bin'] == 0][coord_cols].apply(np.mean, axis=0).values)
    res_cen = (df[df['bin'] == 3][coord_cols].apply(np.mean, axis=0).values)

    _, s_sus_cen = line_coordinates_euclidean(sus, res, sus_cen)
    _, s_res_cen = line_coordinates_euclidean(sus, res, res_cen)

    _, sus_strain = line_coordinates_euclidean(sus, res, sus)
    _, res_strain = line_coordinates_euclidean(sus, res, res)

    df.rename(columns={df.columns[-1]: 'PP'}, inplace=True)

    osd = compute_osd(df)

    return osd['osd']


def get_osd_from_latents_torch(zss, label_col, residual_col, strain_col, latent_dim):
    df_latents = torch.cat(zss, dim=0)

    coords = df_latents[:, :latent_dim]
    coords = torch.nan_to_num(coords, nan=0.0)

    num_repeats = len(zss)

    strain_array = np.tile(strain_col, num_repeats)
    residual_array = np.tile(residual_col, num_repeats)

    unique_strains = []
    sorted_idx = np.argsort(residual_array)
    for s in strain_array[sorted_idx]:
        if s not in unique_strains:
            unique_strains.append(s)

    strain_sus_label = unique_strains[0]
    strain_res_label = unique_strains[-1]

    mask_sus = torch.tensor(strain_array == strain_sus_label, device=coords.device)
    mask_res = torch.tensor(strain_array == strain_res_label, device=coords.device)

    sus_anchor = coords[mask_sus].mean(dim=0)
    res_anchor = coords[mask_res].mean(dim=0)

    line_vec = res_anchor - sus_anchor
    line_len_sq = torch.sum(line_vec ** 2) + 1e-8

    relative_pos = coords - sus_anchor
    dot_prod = torch.sum(relative_pos * line_vec, dim=1)
    projected_scores = dot_prod / torch.sqrt(line_len_sq)

    bin_tensor = torch.as_tensor(label_col, dtype=torch.float32, device=coords.device)
    bin_values = bin_tensor.repeat(num_repeats)

    # replaces compute_osd function
    unique_bins = torch.unique(bin_values).sort()[0]
    bin_means = torch.stack([projected_scores[bin_values == b].mean() for b in unique_bins])

    vx = bin_means - torch.mean(bin_means)
    vy = unique_bins - torch.mean(unique_bins)

    rho_est = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)

    # Calculate differentiable version of Somers' D
    diff = bin_means[1:] - bin_means[:-1]
    somers_est = torch.tanh(diff).mean()

    osd_val = rho_est * somers_est

    return torch.stack([projected_scores, bin_values], dim=1), osd_val

def calc_osd_diff(projected_scores, label_col):
    bin_values = torch.as_tensor(label_col, dtype=torch.float32)

    # replaces compute_osd function
    unique_bins = torch.unique(bin_values).sort()[0]
    bin_means = torch.stack([projected_scores[bin_values == b].mean() for b in unique_bins])

    vx = bin_means - torch.mean(bin_means)
    vy = unique_bins - torch.mean(unique_bins)

    rho_est = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)

    # Calculate differentiable version of Somers' D
    diff = bin_means[1:] - bin_means[:-1]
    somers_est = torch.tanh(diff).mean()

    osd_val = rho_est * somers_est

    return osd_val

def get_osd_hard_proof(zss, label_col, residual_col, strain_col, latent_dim):

    coords_all = torch.cat(zss, dim=0)
    coords = coords_all[:, :latent_dim]

    num_repeats = len(zss)
    strain_array = np.tile(strain_col, num_repeats)
    residual_array = np.tile(residual_col, num_repeats)
    bin_values = torch.tensor(label_col, dtype=torch.float32, device=coords.device).repeat(num_repeats)

    orig_len = len(residual_col)
    sorted_idx = np.argsort(residual_array[:orig_len])
    unique_strains = []
    for s in np.array(strain_col)[sorted_idx]:
        if s not in unique_strains: unique_strains.append(s)

    label_sus, label_res = unique_strains[0], unique_strains[-1]

    sus_anchor = coords[torch.tensor(strain_array == label_sus, device=coords.device)].mean(dim=0)
    res_anchor = coords[torch.tensor(strain_array == label_res, device=coords.device)].mean(dim=0)

    line_vec = res_anchor - sus_anchor
    relative_pos = coords - sus_anchor
    projected_scores = torch.sum(relative_pos * line_vec, dim=1) / (torch.norm(line_vec) + 1e-8)

    rank_x = torch.argsort(torch.argsort(projected_scores)).float()
    rank_y = torch.argsort(torch.argsort(bin_values)).float()

    vx = rank_x - torch.mean(rank_x)
    vy = rank_y - torch.mean(rank_y)
    rho_hard = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)

    unique_bins = torch.unique(bin_values).sort()[0]
    d_list = []

    for i in range(len(unique_bins) - 1):
        b_low, b_high = unique_bins[i], unique_bins[i+1]

        scores_low = projected_scores[bin_values == b_low]
        scores_high = projected_scores[bin_values == b_high]

        pairs = (scores_high.unsqueeze(1) > scores_low.unsqueeze(0)).float()
        auc = torch.mean(pairs)
        d_list.append(2 * auc - 1)

    somers_hard = torch.stack(d_list).mean()

    osd_hard = rho_hard * somers_hard
    return osd_hard

