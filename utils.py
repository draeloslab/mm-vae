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