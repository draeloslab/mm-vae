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
import scanpy as sc
import yaml
from pathlib import Path
import hashlib

from VAE_model_architectures import*
from latent_regularizer import*
from loss import*

CONFIG_PATH = "config.yml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

device = torch.device('cpu')

DATA_PATH = Path(config["data_path"])
HASH_MANIFEST_PATH = Path(config["hash_manifest_path"])
GMM_PATH = Path(config["gmm_path"])
output_dir = config["output_dir"]

num_epoch = config["num_epoch"]
gmm_std = config["gmm_std"]
latent_dim = config["latent_dim"]
latent_noise_scale = config["latent_noise_scale"]
num_gaussian = config["num_gaussian"]
batch_size = config["batch_size"]
model_type = config["model_type"]
SEED = config["seed"]

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
set_seed(SEED)

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return SHA-256 hex digest of a file using chunked reading."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

if not DATA_PATH.exists():
    raise FileNotFoundError(f"AnnData file not found:\n{DATA_PATH}")

if not HASH_MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Hash manifest not found:\n{HASH_MANIFEST_PATH}")

hash_manifest = pd.read_csv(HASH_MANIFEST_PATH)

if "sha256" not in hash_manifest.columns:
    raise KeyError(
        f"'sha256' column not found in hash manifest. "
        f"Available columns: {list(hash_manifest.columns)}"
    )

if len(hash_manifest) == 1:
    expected_sha256 = str(hash_manifest["sha256"].iloc[0]).strip().lower()

else:
    filename_col = None
    for candidate in ["filename", "file", "path", "name"]:
        if candidate in hash_manifest.columns:
            filename_col = candidate
            break
    if filename_col is None:
        raise ValueError(
            "Hash manifest contains multiple rows, but no filename/path "
            "column was found to identify the H5AD file."
        )
    matching_rows = hash_manifest[
        hash_manifest[filename_col].astype(str).apply(
            lambda x: Path(x).name == DATA_PATH.name
        )
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            f"Expected exactly one hash entry for {DATA_PATH.name}, "
            f"but found {len(matching_rows)}."
        )
    expected_sha256 = str(
        matching_rows["sha256"].iloc[0]
    ).strip().lower()

actual_sha256 = sha256_file(DATA_PATH)

if actual_sha256 != expected_sha256:
    raise AssertionError(
        f"Hash mismatch for {DATA_PATH.name}.\n"
        f"Expected: {expected_sha256}\n"
        f"Got:      {actual_sha256}\n"
        "Refusing to proceed — wrong file/version."
    )

print(f"SHA-256 verified: {DATA_PATH.name}")


adata = sc.read_h5ad(DATA_PATH, backed="r")
print(f"Loaded AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
print("Random projection shape:", adata.obsm["random_projection"].shape)
random_projection = np.asarray(adata.obsm["random_projection"]).astype("float32")
dim_count = random_projection.shape[1]

rp_columns = [f"RP{i+1}" for i in range(dim_count)]
df = pd.DataFrame(random_projection, index=adata.obs_names, columns=rp_columns)

metadata_cols = ["CFM_14_5_snRNA", "cell_barcode", "subclass_name", "cell type", "SCBID", "MouseID", "Age", "Strain", "Genotype", "14-6-Bin"]

missing_cols = [col for col in metadata_cols if col not in adata.obs.columns]

if missing_cols:
    raise KeyError(f"Missing required AnnData obs columns: {missing_cols}")

for col in metadata_cols:
    df[col] = adata.obs[col].values

df["Strain"] = df["Strain"].astype(str)

print("Combined dataframe shape:", df.shape)
print("RP dimensions:", dim_count)

def standardizer(input_array):
    mean = np.mean(input_array)
    std = np.std(input_array)
    return (input_array - mean)/std

df_cfm = df[["CFM_14_5_snRNA", "cell_barcode", "subclass_name", "cell type", "SCBID", "MouseID", "Age", "Strain", "Genotype", "14-6-Bin"]]
df_cfm = df_cfm.sort_values(by='CFM_14_5_snRNA', ascending=True)
hue_order = df_cfm['Strain'].unique()
df['CFM_14_5_snRNA'] = standardizer(df['CFM_14_5_snRNA'].values)
df = df.sample(frac=1, random_state=42)

df_gmm = pd.read_csv(GMM_PATH)
gmm_centers = torch.tensor([df_gmm[f'center{i}'].values for i in range(1, 5)]).float()
ks_weight, cv_weight = estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, num_samples=100)

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
        return self.layer1[index], self.x[index], self.R_S_label[index], self.cfm[index]
    def __len__(self):
        return self.len
    
train_set = DataBuilder(df)
test_set = DataBuilder(df)
def _seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                         num_workers=0, worker_init_fn=_seed_worker, generator=g)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                         num_workers=0, worker_init_fn=_seed_worker, generator=g)

def train(epoch, default_path):
    train_loss, train_recon, train_cfm, train_KL = [], [], [], []
    model.train()
    for batch_idx, (df_layer1, data, R_S, cfm) in enumerate(trainloader):
        df_layer1, data, R_S, cfm = df_layer1.to(device), data.to(device), R_S.to(device), cfm.to(device)
        optimizer.zero_grad()

        if model_type in ['C-GMVAE', 'GMVAE']:
            mu, logvar, latent_vectors, recon_batch = model(data, R_S, cfm)
            losses, loss_recons, loss_cfm, loss_KL = get_gmmvaeloss(
                recon_batch, latent_vectors, df_layer1, ks_weight, cv_weight, data_loss_weight, gmm_centers, gmm_std)
        else:
            mu, logvar, latent_vectors, recon_batch = model(data, cfm)
            losses, loss_recons, loss_cfm, loss_KL = get_vaeloss(
                recon_batch, df_layer1, mu, logvar)

        train_loss.append(losses.item())
        train_recon.append(loss_recons.item())
        train_cfm.append(loss_cfm.item())
        train_KL.append(loss_KL.item())

        losses.backward()
        optimizer.step()

    model.eval()
    if epoch % 1 == 0:
        print(f'====> Epoch: {epoch} Train loss: {np.mean(train_loss):.4f}')
        print(f'             {epoch} Rec countmat loss: {np.mean(train_recon):.10f}')
        print(f'             {epoch} Rec cfm loss: {np.mean(train_cfm):.10f}')
        print(f'             {epoch} KL loss: {np.mean(train_KL):.10f}')
        train_losses.append(np.mean(train_loss))
        train_recons.append(np.mean(train_recon))
        train_cfms.append(np.mean(train_cfm))
        train_KLs.append(np.mean(train_KL))

    if epoch in [100, 1000, 3000, 5000, 7000, 10000, 15000, 17000, 20000]:
        try:
            print(f"Saving outputs for epoch {epoch}...")
            mu, log, z, recons = model.forward(test_set.x, test_set.R_S_label, test_set.cfm)
            latent_vectors = z.detach().numpy()
            latent_space = pd.DataFrame(latent_vectors, columns=[f'LV{i+1}' for i in range(latent_dim)])
            latent_space[df.iloc[:, dim_count:].columns] = df[df.iloc[:, dim_count:].columns].values
            latent_space.to_csv(os.path.join(default_path, f'latent_variables_epoch{epoch}.csv'))
            print('Latent variables saved')

            recons = pd.DataFrame(recons.detach().numpy())
            recons[df.iloc[:, dim_count:].columns] = df[df.iloc[:, dim_count:].columns].values
            recons.to_csv(os.path.join(default_path, f'recons_epoch{epoch}.csv'))
            print('Reconstructed features saved')

            torch.save(model, os.path.join(default_path, f'saved_model_epoch{epoch}.pth'))
            print('Model saved')

            loss_frame = pd.DataFrame(list(zip_longest(train_losses, train_recons, train_cfms, train_KLs, fillvalue=np.nan)),
                                      columns=['train_losses', 'train_recons', 'train_cfms', 'train_KLs'])
            loss_frame.to_csv(os.path.join(default_path, f'loss_over_epochs.csv'))
            print('Loss saved')

        except Exception as e:
            print(f"Error during saving or plotting at epoch {epoch}: {e}")

    return

# Model setup
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

os.makedirs(output_dir, exist_ok=True)

train_losses, train_recons, train_cfms, train_KLs = [], [], [], []

for epoch in range(1, num_epoch + 1):
    train(epoch, output_dir)
