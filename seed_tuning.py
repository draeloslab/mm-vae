import os
import time
import yaml
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as datetime 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, default_collate

from utils import*
from mm_cgmvae import *
from train_mmvae import*
from VAE_model_architectures import*

# initialize device 
# set device to gpu if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from mm_cgmvae import *
from train_mmvae import*
from VAE_model_architectures import*

# set seed
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

class PairedDataset(Dataset):
    def __init__(self, paired_df, input_dims, label_col, cfm_col, dataset, residual_col, strain_col):
        self.paired_df = paired_df
        self.dataset = dataset
        if dataset == 'AD':
            self.df1_layer1, self.x1, self.label1, self.cfm1, self.df2_layer1, self.x2, self.label2, self.cfm2, self.residuals, self.strain = load_pairedregion_data(paired_df, input_dims[0], input_dims[1], label_col, cfm_col, residual_col, strain_col)
            self.residuals = numpyToTensor(self.residuals)
            self.strain = self.strain
        elif dataset == 'DO': 
            self.df1_layer1, self.x1, self.label1, self.cfm1, self.df2_layer1, self.x2, self.label2, self.cfm2 = load_pairedmodal_data(paired_df, input_dims[0], label_col, cfm_col)
        self.df1_layer1 = numpyToTensor(self.df1_layer1)
        self.x1 = numpyToTensor(self.x1)
        self.cfm1 = numpyToTensor(self.cfm1)
        self.label1 = numpyToTensor(self.label1)

        self.df2_layer1 = numpyToTensor(self.df2_layer1)
        self.x2 = numpyToTensor(self.x2)
        self.cfm2 = numpyToTensor(self.cfm2)
        self.label2 = numpyToTensor(self.label2)

    def __len__(self):
        return len(self.paired_df)
    def __getitem__(self, index):
        if self.dataset == 'AD':
            return (self.df1_layer1[index], self.x1[index], self.label1[index], self.cfm1[index]), (self.df2_layer1[index], self.x2[index], self.label2[index], self.cfm2[index]), (self.residuals[index]), (self.strain[index])
    
        else:
            return (self.df1_layer1[index], self.x1[index], self.label1[index], self.cfm1[index]), (self.df2_layer1[index], self.x2[index], self.label2[index], self.cfm2[index])
    
def load_pairedregion_data(df, input_dim1, input_dim2, label_col, cfm_col, residual_col, strain_col):
    x1 = df[list(df.columns[0:input_dim1])].values.astype('float32')
    label1 = df[label_col].values
    cfm1 = df[cfm_col].values
    df1_layer1 = df[list(df.columns[0:input_dim1])].copy()
    df1_layer1['cfm'] = cfm1
    df1_layer1 = df1_layer1.values.astype('float32')

    x2 = df[list(df.columns[4852:4852+input_dim2])].values.astype('float32') # CHANGE back to df.columns[4852:4851+input_dim1]
    label2 = df[label_col].values
    cfm2 = df[cfm_col].values
    df2_layer1 = df[list(df.columns[4852:4852+input_dim2])].copy() # CHANGE BACK to df.columns[4852:4851+input_dim1]
    df2_layer1['cfm'] = cfm2
    df2_layer1 = df2_layer1.values.astype('float32')

    residuals = df[residual_col].values
    strain = df[strain_col].values

    return df1_layer1, x1, label1, cfm1, df2_layer1, x2, label2, cfm2, residuals, strain

def math_osd(df):
    coord_cols = df.columns[:10]
    df[coord_cols] = df[coord_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # find most susceptible/resilience strains
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
    #print("Signed distance (sus_cen, res_cen):", s_sus_cen, s_res_cen)

    _, sus_strain = line_coordinates_euclidean(sus, res, sus)
    _, res_strain = line_coordinates_euclidean(sus, res, res)
    #print("Signed distance (sus_strain, res_strain):", sus_strain, res_strain)

    # Rename last column as PP (Phenotypic Projection)
    df.rename(columns={df.columns[-1]: 'PP'}, inplace=True)

    osd = compute_osd(df)

    return osd

def math_osd_diff(df):
    return 0

# full run for a single seed
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

    # initialize parameters 
    input_dim = [4842, 4850] # 4841, 4841 CHANGE BACK
    H1 = [128, 128]
    H2 = [64, 64]
    H3 = [32, 32]
    latent_dim = 10
    batch_size = 271
    scales = [1, 1]
    dataset = 'AD'
    dataset_abbrev = ['HC', 'PFC'] # Change for different datasets ['HC1', 'HC2']
    num_classes = 4
    num_kl_samples = 100
    learning_rate = 1e-3
    num_epoch = 100

    # -----------------------------
    # 1. Prepare dataframe
    # -----------------------------
    df = df_input.copy()

    label_col = '14-6-Bin_HC' # '14-6-Bin_HC1' # CHANGE BACK
    cfm_col = 'CFM_14_5_snRNA_HC' # 'CFM_14_5_snRNA_HC1' # CHANGE BACK
    df[cfm_col] = standardizer(df[cfm_col])
    meta = {
        'strain': df['strain'],
        '14-6-residual': df['14-6-residual_HC'] # CHANGE BACK
    }

    df_train = df.sample(frac=1, random_state=SEED)
    train_dataset = PairedDataset(df_train, input_dim, label_col, cfm_col, dataset, '14-6-residual_HC', 'strain')
    test_dataset = PairedDataset(df, input_dim, label_col, cfm_col, dataset, '14-6-residual_HC', 'strain')

    def _seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=_seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0, worker_init_fn=_seed_worker, generator=g)

    first_iter = next(iter(train_loader))
    #print(first_iter[3][0])

    #initialize gmm centers and compute weights 
    # gmm_centers = pd.read_csv('/home/rachel/Desktop/mm-vae data/multiregion_adbxd_cr/gmm_centers_4gaussian_14mon_fromQRT.csv')
    # gmm_centers = gmm_centers.iloc[:, 1:].T
    # gmm_std = 2
    # ks_weight, cv_weight, samples, components = estimate_loss_coefficients(batch_size, torch.tensor([gmm_centers.values]).float().squeeze(), gmm_std, num_kl_samples)
    # ks_weights = [ks_weight, ks_weight]
    # cv_weights = [cv_weight, cv_weight]
    # gmm_centers = [torch.tensor([gmm_centers.values]).float().squeeze(), torch.tensor([gmm_centers.values]).float().squeeze()]

    gmm_std = 2
    gmm_centers = []
    ks_weights = []
    cv_weights = []
    for dataset in dataset_abbrev:
        gmm_path = f"/home/rachel/Desktop/mm-vae data/multiregion_adbxd_cr/{dataset}_priors.csv"
        #gmm_path = '/home/rachel/Desktop/mm-vae data/multiregion_adbxd_cr/gmm_centers_4gaussian_14mon_fromQRT.csv'
        gmm = pd.read_csv(gmm_path)
        gmm = gmm.iloc[:, 1:].T
        ks_weight, cv_weight, samples, components = estimate_loss_coefficients(batch_size, torch.tensor([gmm.values]).float().squeeze(), gmm_std, num_kl_samples)
        gmm_centers.append(torch.tensor([gmm.values]).float().squeeze())
        ks_weights.append(ks_weight)
        cv_weights.append(cv_weight)

    # initialize model and optimizer 
    model = MM_CGMVAE(input_dim, H1, H2, H3, latent_dim, scales, num_classes, gmm_centers, gmm_std, ks_weights, cv_weights)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    # initialize training function
    def train(epoch, model, optimizer, train_loader, test_loader, device):
        model.train()
        total_loss = 0
        for i, data in enumerate(train_loader):
            
            # unpack/prepare data
            df1_layer1 = data[0][0].to(device)
            df2_layer1 = data[1][0].to(device)
            layer1_data = [df1_layer1, df2_layer1]
            x_1 = data[0][1].to(device)
            x_2 = data[1][1].to(device)
            x_data = [x_1, x_2]
            y = data[0][2].to(device)
            cfm = data[0][3].to(device)
            optimizer.zero_grad()

            residual_col = data[2] 
            strain_col = data[3]

            loss, lpx_zs, kls, lpxz_ind, mse_loss, mse_cfm_loss, osd_enc, bin_means = model.moe_elbo_cgmvae_loss(x_data, y, cfm, layer1_data, strain_col, residual_col)
            
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        #print(f'====> Epoch: {epoch:03d} Train loss: {total_loss / len(train_loader.dataset):.4f}')
        print(f'====> Epoch: {epoch:03d} Train loss: {total_loss:.4f}')

        latent_space = None
        loss_frame = None

        if epoch == 100: 
            try: 
                print(f"Saving outputs for epoch {epoch}...")
                model.eval()
                with torch.no_grad():
                    for i, test_data in enumerate(test_loader):
                        optimizer.zero_grad()
                        latent_space = model.get_latent_space(test_data, dataset_abbrev, meta)

            except Exception as e:
                print(f"Error during saving or plotting at epoch {epoch}: {e}")

        return latent_space 

    # train the model 
    latent_space_final = None 

    for epoch in range(1, num_epoch + 1):
        latent_space = train(epoch, model, optimizer, train_loader, test_loader, device)
        if latent_space is not None:
            latent_space_final = latent_space

    df = latent_space_final.copy()

    # calculate osd value 
    # calculate phenotypic projection
    osd = math_osd(df)

    return osd

if __name__ == "__main__":
    #path = '/home/rachel/Desktop/mm-vae data/multiregion_adbxd_cr/sameregionhc_paired.csv'
    path = '/home/rachel/Desktop/mm-vae data/multiregion_adbxd_cr/hc_pfc_adbxd_paired_30perc.csv'
    df_main = pd.read_csv(path, low_memory=False)
    seeds = []
    rho = []
    som = []
    osds = []
    for seed in range(50, 300):
        osd = run_one_seed(df_main, seed)
        seeds.append(seed)
        rho.append(osd['rho_spearman'])
        som.append(osd['somers_d_adjacent'])
        osds.append(osd['osd'])
        print('Seed: ' + str(seed))
        print(osd['rho_spearman'], osd['somers_d_adjacent'], osd['osd'])
    seed_results = {
        'Seed': seeds, 
        'Rho spearman': rho,
        'Somers_D': som, 
        'OSD': osds 
    }
    pd.DataFrame(seed_results).to_csv("/home/rachel/Desktop/mm-vae results/multiregion_adbxd_cr_final/p2_multiregion_offset50-300.csv", index=False)