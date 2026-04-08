import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import psutil
from itertools import zip_longest
import time

device = torch.device('cpu')

from VAE_model_architectures import *
from latent_regularizer import *
from loss import *


# ----------------------------- #
#  Helper: reproducible seeds   #
# ----------------------------- #
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ----------------------------- #
#  Data utilities (unchanged)   #
# ----------------------------- #
dim_count = 4963  # number of RP / count features


def standardizer(input_array):
    mean = np.mean(input_array)
    std = np.std(input_array)
    return (input_array - mean) / std


def load_data(df):
    x = df.iloc[:, :dim_count].values.astype('float32')
    R_S_label = df['14-6-Bin'].values
    cfm = df['CFM_14_5_snRNA'].values
    df_layer1 = df.iloc[:, :dim_count].copy()
    df_layer1['cfm'] = cfm
    df_layer1 = df_layer1.values.astype('float32')
    return df_layer1, x, R_S_label, cfm


def numpyToTensor(x):
    return torch.from_numpy(x).to(device)


class DataBuilder(Dataset):
    def __init__(self, data):
        self.layer1, self.x, self.R_S_label, self.cfm = load_data(data)
        self.layer1 = numpyToTensor(self.layer1)
        self.x = numpyToTensor(self.x)
        self.R_S_label = numpyToTensor(self.R_S_label)
        self.cfm = numpyToTensor(self.cfm)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return (
            self.layer1[index],
            self.x[index],
            self.R_S_label[index],
            self.cfm[index],
        )

    def __len__(self):
        return self.len


# ----------------------------- #
#  Projection helpers           #
# ----------------------------- #
def line_coordinates_euclidean(A, B, P):
    A = np.array(A)
    B = np.array(B)
    P = np.array(P)

    u_hat = (B - A) / np.linalg.norm(B - A)  # Unit direction vector from A to B
    s = np.dot(P - A, u_hat)                # Signed distance from A
    P_proj = A + s * u_hat                  # Projected point on the line

    return P_proj, s


def compute_kde_peak(values, bandwidth=0.1, grid_size=1000):
    """Estimate KDE peak (mode) location."""
    values = values[:, None]
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)
    grid = np.linspace(values.min(), values.max(), grid_size)[:, None]
    log_dens = kde.score_samples(grid)
    peak = grid[np.argmax(log_dens)][0]
    return peak


def compute_osd(df, projection_col='PP', bin_col='14-6-Bin', bandwidth=0.1):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

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
        from sklearn.metrics import roc_auc_score
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


# ========================================================= #
#   MAIN ENTRY POINT FOR EACH SEED: run_one_seed(df, SEED)  #
# ========================================================= #
def run_one_seed(df_input: pd.DataFrame, SEED: int):
    """
    Run the full model training and phenotypic projection for one SEED.

    Parameters
    ----------
    df_input : pd.DataFrame
        Original input dataframe (genes/RP features + metadata columns).
    SEED : int
        Random seed to use for this run.

    Returns
    -------
    df_latent : pd.DataFrame or None
        Latent space + metadata + projection columns (PP).
    loss_frame : pd.DataFrame or None
        Training loss history.
    osd_result : dict or None
        OSD metric summary.
    """
    print(f"\n--- Starting run_one_seed with SEED = {SEED} ---")

    # -----------------------------
    # 0. Set seeds and environment
    # -----------------------------
    set_seed(SEED)

    # -----------------------------
    # 1. Prepare dataframe
    # -----------------------------
    df = df_input.copy()

    # CFM subset (used only for ordering / plotting)
    df_cfm = df[['Strain', 'CFM_14_5_snRNA', '14-6-residual', '14-6-Bin']].copy()
    df_cfm = df_cfm.sort_values(by='CFM_14_5_snRNA', ascending=True)
    hue_order = df_cfm['Strain'].unique()  # not strictly used, but kept

    # Standardize CFM and shuffle rows (same as your original code)
    df['CFM_14_5_snRNA'] = standardizer(df['CFM_14_5_snRNA'].values)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df_train = df

    num_epoch = 100
    gmm_std = 2
    latent_dim = 10
    latent_noise_scale = 0
    num_gaussian = 4
    batch_size = 511
    model_type = 'C-GMVAE'

    # -----------------------------
    # 2. GMM centers & loss weights
    # -----------------------------
    df_gmm = pd.read_csv('gmm_centers_4gaussian_14mon_fromQRT.csv')
    gmm_centers = torch.tensor(
        [df_gmm[f'center{i}'].values for i in range(1, 5)]
    ).float()
    ks_weight, cv_weight = estimate_loss_coefficients(
        batch_size, gmm_centers, gmm_std, num_samples=100
    )

    # -----------------------------
    # 3. Datasets & loaders
    # -----------------------------
    train_set = DataBuilder(df_train)
    test_set = DataBuilder(df_train)

    def _seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    testloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    # -----------------------------
    # 4. Model setup
    # -----------------------------
    D_in = train_set.x.shape[1]
    D_out = D_in
    H1, H2, H3 = 128, 64, 32
    data_loss_weight = 1

    if model_type == 'C-GMVAE':
        model = Autoencoder_CGMVAE(D_in, D_out, H1, H2, H3, latent_dim).to(device)
    elif model_type == 'GMVAE':
        model = Autoencoder_GMVAE(D_in, D_out, H1, H2, H3, latent_dim).to(device)
    elif model_type == 'VAE':
        model = Autoencoder_VAE(D_in, D_out, H1, H2, H3, latent_dim).to(device)
    elif model_type == 'C-VAE':
        model = Autoencoder_CVAE(D_in, D_out, H1, H2, H3, latent_dim).to(device)
    else:
        raise ValueError("Invalid model_type")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    default_path = (
        f"L:/yidingca/snRNA_HC_PFC_VAE/trained_models/{model_type}_SEED_{SEED}"
    )

    # Loss tracking
    train_losses, train_recons, train_cfms, train_KLs = [], [], [], []

    # -----------------------------
    # 5. Training function (per epoch)
    # -----------------------------
    def train_epoch(epoch):
        nonlocal train_losses, train_recons, train_cfms, train_KLs

        model.train()
        batch_losses, batch_recon, batch_cfm, batch_kl = [], [], [], []

        for batch_idx, (df_layer1, data, R_S, cfm) in enumerate(trainloader):
            df_layer1 = df_layer1.to(device)
            data = data.to(device)
            R_S = R_S.to(device)
            cfm = cfm.to(device)

            optimizer.zero_grad()
            mu, logvar, latent_vectors, recon_batch = model(data, R_S, cfm)

            if model_type in ['C-GMVAE', 'GMVAE']:
                losses, loss_recons, loss_cfm, loss_KL = get_gmmvaeloss(
                    recon_batch,
                    latent_vectors,
                    df_layer1,
                    ks_weight,
                    cv_weight,
                    data_loss_weight,
                    gmm_centers,
                    gmm_std,
                )
            else:
                losses, loss_recons, loss_cfm, loss_KL = get_vaeloss(
                    recon_batch, df_layer1, mu, logvar
                )

            batch_losses.append(losses.item())
            batch_recon.append(loss_recons.item())
            batch_cfm.append(loss_cfm.item())
            batch_kl.append(loss_KL.item())

            losses.backward()
            optimizer.step()

        # Epoch summary
        mean_loss = np.mean(batch_losses)
        mean_recon = np.mean(batch_recon)
        mean_cfm = np.mean(batch_cfm)
        mean_kl = np.mean(batch_kl)

        train_losses.append(mean_loss)
        train_recons.append(mean_recon)
        train_cfms.append(mean_cfm)
        train_KLs.append(mean_kl)

        print(f"====> Epoch: {epoch} Train loss: {mean_loss:.4f}")
        print(f"             {epoch} Rec countmat loss: {mean_recon:.10f}")
        print(f"             {epoch} Rec cfm loss: {mean_cfm:.10f}")
        print(f"             {epoch} KL loss: {mean_kl:.10f}")

        latent_space = None
        loss_frame = None

        # Save / extract at specific epoch (100 in your original code)
        if epoch == 100:
            try:
                print(f"Saving outputs for epoch {epoch}...")
                model.eval()
                with torch.no_grad():
                    mu, log, z = model.encode(
                        test_set.x, test_set.R_S_label, test_set.cfm
                    )
                latent_vectors = mu.detach().cpu().numpy()
                latent_space = pd.DataFrame(
                    latent_vectors,
                    columns=[f'LV{i+1}' for i in range(latent_dim)],
                )

                # Add metadata columns from df_train
                meta_cols = df_train.columns[dim_count:]
                latent_space[meta_cols] = df_train[meta_cols].reset_index(drop=True)

                loss_frame = pd.DataFrame(
                    list(
                        zip_longest(
                            train_losses,
                            train_recons,
                            train_cfms,
                            train_KLs,
                            fillvalue=np.nan,
                        )
                    ),
                    columns=['train_losses', 'train_recons', 'train_cfms', 'train_KLs'],
                )

            except Exception as e:
                print(f"Error during saving or plotting at epoch {epoch}: {e}")

        return latent_space, loss_frame

    # -----------------------------
    # 6. Full training loop
    # -----------------------------
    latent_space_final = None
    loss_frame_final = None

    for epoch in range(1, num_epoch + 1):
        latent_space, loss_frame = train_epoch(epoch)
        if latent_space is not None:
            latent_space_final = latent_space
            loss_frame_final = loss_frame

    if latent_space_final is None:
        print("Warning: latent space was not generated (no epoch == 100?).")
        return None, None, None

    df_latent = latent_space_final.copy()

    # -----------------------------
    # 7. Phenotypic projection
    # -----------------------------
    coord_cols = df_latent.columns[:10]  # First 10 columns are coordinates
    df_latent[coord_cols] = df_latent[coord_cols].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)

    # Determine most susceptible / most resilient strains by residual
    strain_order = (
        df_latent.sort_values('14-6-residual', ascending=True)['Strain'].unique()
    )
    strain_sus = strain_order[0]
    strain_res = strain_order[-1]

    sus = (
        df_latent[df_latent['Strain'] == strain_sus][coord_cols]
        .apply(np.mean, axis=0)
        .values
    )
    res = (
        df_latent[df_latent['Strain'] == strain_res][coord_cols]
        .apply(np.mean, axis=0)
        .values
    )

    def project_row(row):
        coords = row[coord_cols].values
        proj, dist = line_coordinates_euclidean(sus, res, coords)
        return pd.Series({'Projected': proj, 'Signed_Distance': dist})

    projections = df_latent.apply(project_row, axis=1)

    df_latent = pd.concat(
        [
            df_latent.reset_index(drop=True),
            pd.DataFrame(
                projections['Projected'].tolist(),
                columns=[f'Proj_{i+1}' for i in range(10)],
            ),
            projections[['Signed_Distance']],
        ],
        axis=1,
    )

    # Recompute some centers along the line
    coord_cols = df_latent.columns[:10]
    df_latent[coord_cols] = df_latent[coord_cols].apply(
        pd.to_numeric, errors='coerce'
    ).fillna(0)

    strain_order = (
        df_latent.sort_values('14-6-residual', ascending=True)['Strain'].unique()
    )
    strain_sus = strain_order[0]
    strain_res = strain_order[-1]

    sus = (
        df_latent[df_latent['Strain'] == strain_sus][coord_cols]
        .apply(np.mean, axis=0)
        .values
    )
    res = (
        df_latent[df_latent['Strain'] == strain_res][coord_cols]
        .apply(np.mean, axis=0)
        .values
    )
    sus_cen = (
        df_latent[df_latent['14-6-Bin'] == 0][coord_cols]
        .apply(np.mean, axis=0)
        .values
    )
    res_cen = (
        df_latent[df_latent['14-6-Bin'] == 3][coord_cols]
        .apply(np.mean, axis=0)
        .values
    )

    _, s_sus_cen = line_coordinates_euclidean(sus, res, sus_cen)
    _, s_res_cen = line_coordinates_euclidean(sus, res, res_cen)
    print("Signed distance (sus_cen, res_cen):", s_sus_cen, s_res_cen)

    _, sus_strain = line_coordinates_euclidean(sus, res, sus)
    _, res_strain = line_coordinates_euclidean(sus, res, res)
    print("Signed distance (sus_strain, res_strain):", sus_strain, res_strain)

    # Rename last column as PP (Phenotypic Projection)
    df_latent.rename(columns={df_latent.columns[-1]: 'PP'}, inplace=True)

    # -----------------------------
    # 8. KDE plot of PP by 14-6-Bin
    # -----------------------------
    sns.set(style="white")
    import matplotlib.cm as cm

    cmap1 = cm.get_cmap('gist_heat')
    cmap2 = cm.get_cmap('Blues')
    custom_palette = [cmap1(0.5), cmap1(0.85), cmap2(0.4), cmap2(0.99)]

    #temp = df_latent

    # -----------------------------
    # 9. OSD metrics
    # -----------------------------
    osd_result = compute_osd(df_latent)
    print("Spearman ρ:", osd_result['rho_spearman'])
    print("Mean Somers' D:", osd_result['somers_d_adjacent'])
    print("OSD:", osd_result['osd'])

    print(f"--- Finished run_one_seed with SEED = {SEED} ---\n")

    return df_latent, loss_frame_final, osd_result


# Optional: allow running this file directly for quick test
if __name__ == "__main__":
    path = 'L:/yidingca/snRNA_HC_PFC_VAE/processed_count_matrix/hc_pfc_combined_rp_nor.csv'
    df_main = pd.read_csv(path, index_col=0, dtype={'Strain': str})
    run_one_seed(df_main, SEED=13)
