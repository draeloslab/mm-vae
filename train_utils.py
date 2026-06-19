# ~~~~ Imports ~~~~
# OSD loss may not have all strains across all batches - may be noisy
# May be simpler to pass strain_sus_label and strain_res_label into the training loop
import os
import sys
import time
import random
import warnings
from datetime import datetime  
from itertools import zip_longest
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

warnings.filterwarnings("ignore")

from VAE_model_architectures import *
from latent_regularizer import *
from loss_cl import *

# ~~~~ Functions and Classes ~~~~
@dataclass
class Config:
    # Change to local paths for benchmarking
    gmm_path: str = "/nfs/turbo/umms-kaczoro/u19-shared/vae-mouse-hc-updated/VAE_code_SEED_experiment/gmm_centers_4gaussian_14mon_new.csv" 
    data_path: str = "/nfs/turbo/umms-kaczoro/u19-shared/vae-mouse-hc-updated/14_mon_HC_ADBXD_subset/all_cells_adbxd_14mon_rp_nor.csv"
    output_dir: str = "/home/varnika/draelos_lab/proj_VAE/results/benchmarks/15000ep_06152026"
    # Benchmark with my OSD loss version. Keep configs as is.
    seed: int = 29 # 29 works, 245 is a test
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dim_count: int = 4842
    num_epoch: int = 15000
    gmm_std: float = 2.0
    latent_dim: int = 10
    latent_noise_scale: float = 0.0
    num_gaussian: int = 4
    batch_size: int = 512
    model_type: str = "C-GMVAE"

    # C-GMVAE
    H1: int = 128
    H2: int = 64
    H3: int = 32

    # Projection head
    proj_hidden_dim: int = 16
    proj_out_dim: int = 32
    data_loss_weight: float = 1.0

    # Contrastive loss
    use_contrastive: bool = True
    use_supcon_knn: bool = True
    use_weighted_nt_xent: bool = False
    lambda_cl: float = 1.0
    T_cl: float = 0.2
    
    # Supcon loss
    knn_k: int = 16
    symmetric_knn: bool = False

    # Weighted NT-Xent loss
    nt_xent_s: float = 0.2
    nt_xent_row_normalize: bool = True

    use_equal_start_weighting: bool = True
    init_alpha_cgmvae: float = 1.0
    init_beta_cl: float = 1.0
    init_gamma_osd: float = 1.0

    use_adamw: bool = True
    base_lr: float = 1e-4
    projection_head_lr: float = 2e-4
    weight_decay: float = 1e-4
    use_separate_projection_lr: bool = False

    use_lr_scheduler: bool = False
    use_linear_warmup: bool = False
    warmup_epochs: int = 5
    use_cosine_decay: bool = False

    # CL scheduling
    use_cl_scheduler: bool = False
    cl_start_epoch: int = 2000
    cl_ramp_epochs: int = 500
    cl_start_scale: float = 0.0
    cl_max_scale: float = 1.0
    cl_schedule_type: str = "cosine"   # "linear" or "cosine"

    # OSD loss
    train_osd: bool = True
    osd_temperature: float = 0.1

    save_epochs: tuple = (100, 1000, 3000, 5000, 10000, 15000, 17000, 20000) # for shorter runs, also include 2000/7000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def standardizer(input_array):
    mean = np.mean(input_array)
    print(f"Mean CFM_14_5_snRNA before standardization: {mean}")
    std = np.std(input_array)
    print(f"Std of CFM_14_5_snRNA before standardization: {std}")
    return (input_array - mean) / std

# Modified dataloader to return strain and resiudal info for OSD loss during training
def load_data(df):
    x = df.iloc[:, :cfg.dim_count].values.astype("float32")
    R_S_label = df["14-6-Bin"].values
    cfm = df["CFM_14_5_snRNA"].values.astype("float32")
    strain = df["Strain"].values.astype(str) 
    residual = df["14-6-residual"].values.astype("float32") 
    df_layer1 = df.iloc[:, :cfg.dim_count].copy()
    df_layer1["cfm"] = cfm
    df_layer1 = df_layer1.values.astype("float32")
    return df_layer1, x, R_S_label, cfm, strain, residual


def numpyToTensor(x):
    return torch.from_numpy(x).to(device)

class DataBuilder(Dataset):
    def __init__(self, data):
        self.layer1, self.x, self.R_S_label, self.cfm, self.strain, self.residual = load_data(data)
        self.layer1 = numpyToTensor(self.layer1)
        self.x = numpyToTensor(self.x)
        self.R_S_label = numpyToTensor(self.R_S_label)
        self.cfm = numpyToTensor(self.cfm)
        self.residual = numpyToTensor(self.residual)
        self.len = self.x.shape[0]
        
    def __getitem__(self, index):
        return self.layer1[index], self.x[index], self.R_S_label[index], self.cfm[index], self.strain[index], self.residual[index]

    def __len__(self):
        return self.len

def _seed_worker(worker_id):
    worker_seed = cfg.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def grad_norm_from_grads(grads):
    sqsum = 0.0
    for g in grads:
        if g is not None:
            sqsum += g.detach().pow(2).sum().item()
    return sqsum ** 0.5


def get_grad_norm(module):
    sqsum = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sqsum += p.grad.detach().pow(2).sum().item()
    return sqsum ** 0.5


def flatten_grads(grads):
    flat = [g.detach().reshape(-1) for g in grads if g is not None]
    if len(flat) == 0:
        return None
    return torch.cat(flat)


def grad_cosine_similarity(grads1, grads2, eps=1e-12):
    g1 = flatten_grads(grads1)
    g2 = flatten_grads(grads2)
    if g1 is None or g2 is None:
        return np.nan
    return torch.dot(g1, g2).item() / (g1.norm().item() * g2.norm().item() + eps)


def build_optimizer(model, cfg: Config):
    proj_params = []
    if hasattr(model, "projection_head"):
        proj_params = list(model.projection_head.parameters())

    proj_param_ids = {id(p) for p in proj_params}
    base_params = [p for p in model.parameters() if id(p) not in proj_param_ids]

    optimizer_cls = AdamW if cfg.use_adamw else Adam

    if cfg.use_separate_projection_lr and len(proj_params) > 0:
        param_groups = [
            {"params": base_params, "lr": cfg.base_lr, "weight_decay": cfg.weight_decay},
            {"params": proj_params, "lr": cfg.projection_head_lr, "weight_decay": cfg.weight_decay},
        ]
    else:
        param_groups = [
            {"params": model.parameters(), "lr": cfg.base_lr, "weight_decay": cfg.weight_decay}
        ]

    optimizer = optimizer_cls(param_groups)
    return optimizer


def build_scheduler(optimizer, cfg: Config):
    if not cfg.use_lr_scheduler:
        return None

    if cfg.use_linear_warmup and cfg.use_cosine_decay:
        warmup_epochs = max(1, cfg.warmup_epochs)
        cosine_epochs = max(1, cfg.num_epoch - warmup_epochs)

        warmup = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: float(epoch + 1) / float(warmup_epochs)
        )
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )
        return scheduler

    if cfg.use_linear_warmup:
        warmup_epochs = max(1, cfg.warmup_epochs)
        return LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, float(epoch + 1) / float(warmup_epochs))
        )

    if cfg.use_cosine_decay:
        return CosineAnnealingLR(optimizer, T_max=cfg.num_epoch)

    return None

def get_cl_scale(epoch, cfg: Config):
    if not cfg.use_cl_scheduler:
        return 1.0

    if epoch < cfg.cl_start_epoch:
        return cfg.cl_start_scale

    progress = (epoch - cfg.cl_start_epoch) / max(1, cfg.cl_ramp_epochs)
    progress = min(max(progress, 0.0), 1.0)

    if cfg.cl_schedule_type == "linear":
        scale = cfg.cl_start_scale + (cfg.cl_max_scale - cfg.cl_start_scale) * progress
    elif cfg.cl_schedule_type == "cosine":
        cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
        scale = cfg.cl_start_scale + (cfg.cl_max_scale - cfg.cl_start_scale) * cosine_progress
    else:
        raise ValueError(f"Unknown cl_schedule_type.")

    return scale

def train(epoch, default_path):
    global equal_start_initialized, alpha_cgmvae, beta_cl, gamma_osd
    
    train_loss, train_recon, train_cfm, train_KL = [], [], [], []
    train_CL, train_weighted_ksloss = [], []
    train_weighted_cov_loss, train_weighted_loss_recons = [], []
    train_osd_loss, train_osd_value = [], []
    train_cl_grad_norms, train_total_grad_norms = [], []
    train_mu_grad_norms, train_cl_to_total_ratios = [], []

    train_cgmvae_grad_norms, train_grad_cos_sims = [], []
    train_cgmvae_to_total_ratios = []
    train_balanced_cgmvae_losses, train_balanced_cl_losses, train_effective_cl_losses = [], [], []

    model.train()

    for batch_idx, (df_layer1, data, R_S, cfm, strain, residual) in enumerate(trainloader):
        df_layer1 = df_layer1.to(device)
        data = data.to(device)
        R_S = R_S.to(device)
        cfm = cfm.to(device)
        residual = residual.to(device)

        optimizer.zero_grad()

        if cfg.model_type in ["C-GMVAE", "GMVAE"]:
            mu, logvar, latent_vectors, recon_batch = model(data, R_S, cfm)

            losses, loss_recons, loss_cfm, loss_KL, weighted_ksloss, weighted_cov_loss, weighted_loss_recons = get_gmmvaeloss(
                recon_batch,
                latent_vectors,
                df_layer1,
                ks_weight,
                cv_weight,
                cfg.data_loss_weight,
                gmm_centers,
                cfg.gmm_std,
            )

            cgmvae_loss = losses
            
            if cfg.train_osd:
                raw_osd_loss, osd_value, osd_rho, osd_somers = osd_loss_from_latents(
                    latent_vectors=latent_vectors,
                    bin_labels=R_S,
                    strain_labels=strain,
                    residual_labels=residual,
                    latent_dim=cfg.latent_dim,
                    temperature=cfg.osd_temperature,
                )
            else:
                raw_osd_loss = torch.tensor(0.0, device=device)
                osd_value = torch.tensor(0.0, device=device)

            if cfg.use_contrastive:
                if cfg.use_supcon_knn and cfg.use_weighted_nt_xent:
                    raise ValueError("Error.")
                p = model.projection_head(mu)
    
                if cfg.use_supcon_knn:
                    weighted_cl_loss, loss_cl = SupCon_loss(
                        p=p,
                        cfm=cfm,
                        lambda_cl=cfg.lambda_cl,
                        k=cfg.knn_k,
                        T=cfg.T_cl,
                        symmetric_knn=cfg.symmetric_knn,
                    )
            
                elif cfg.use_weighted_nt_xent:
                    weighted_cl_loss, loss_cl = weighted_nt_xent(
                        mu_k=p,
                        cfm=cfm,
                        lambda_cl=cfg.lambda_cl,
                        T=cfg.T_cl,
                        s=cfg.nt_xent_s,
                        row_normalize=cfg.nt_xent_row_normalize,
                    )
            
                else:
                    loss_cl = torch.tensor(0.0, device=device)
                    weighted_cl_loss = torch.tensor(0.0, device=device)
            
            else:
                loss_cl = torch.tensor(0.0, device=device)
                weighted_cl_loss = torch.tensor(0.0, device=device)

            if cfg.use_equal_start_weighting:
                if not equal_start_initialized:
                    alpha_cgmvae = 1.0 / (cgmvae_loss.detach().item() + 1e-8)
                    beta_cl = 1.0 / (weighted_cl_loss.detach().item() + 1e-8) if cfg.use_contrastive else 1.0
                    gamma_osd = 1.0 / (raw_osd_loss.detach().item() + 1e-8) if cfg.train_osd else 1.0
            
                    equal_start_initialized = True
            
                    print("Equal-start initialized:")
                    print(f"alpha_cgmvae = {alpha_cgmvae:.8f}")
                    print(f"beta_cl      = {beta_cl:.8f}")
                    print(f"gamma_osd    = {gamma_osd:.8f}")
                    print(f"Initial CGMVAE loss = {cgmvae_loss.detach().item():.8f}")
                    print(f"Initial CL loss     = {weighted_cl_loss.detach().item():.8f}")
                    print(f"Initial OSD loss    = {raw_osd_loss.detach().item():.8f}")
            
                balanced_cgmvae_loss = alpha_cgmvae * cgmvae_loss
                balanced_cl_loss = beta_cl * weighted_cl_loss
                balanced_osd_loss = gamma_osd * raw_osd_loss
            else:
                balanced_cgmvae_loss = cgmvae_loss
                balanced_cl_loss = weighted_cl_loss
                balanced_osd_loss = raw_osd_loss

            # losses = balanced_cgmvae_loss + balanced_cl_loss
            cl_scale = get_cl_scale(epoch, cfg)
            effective_cl_loss = cl_scale * balanced_cl_loss
            losses = balanced_cgmvae_loss + effective_cl_loss + balanced_osd_loss

            encoder_module = model.encoder1 if hasattr(model, "encoder1") else model
            encoder_params = [p for p in encoder_module.parameters() if p.requires_grad]

            #cl_grads = torch.autograd.grad(
            #    balanced_cl_loss, encoder_params, retain_graph=True, allow_unused=True
            #)
            cl_grads = torch.autograd.grad(
                effective_cl_loss, encoder_params, retain_graph=True, allow_unused=True
            )
            cl_grad_norm = grad_norm_from_grads(cl_grads)

            cgmvae_grads = torch.autograd.grad(
                balanced_cgmvae_loss, encoder_params, retain_graph=True, allow_unused=True
            )
            cgmvae_grad_norm = grad_norm_from_grads(cgmvae_grads)
            grad_cos_sim = grad_cosine_similarity(cl_grads, cgmvae_grads)

            #mu_grad = torch.autograd.grad(
            #    balanced_cl_loss, mu, retain_graph=True, allow_unused=True
            #)[0]
            mu_grad = torch.autograd.grad(
                effective_cl_loss, mu, retain_graph=True, allow_unused=True
            )[0]
            mu_grad_norm = mu_grad.detach().norm(2).item() if mu_grad is not None else 0.0

        else:
            mu, logvar, latent_vectors, recon_batch = model(data, cfm)
            losses, loss_recons, loss_cfm, loss_KL = get_vaeloss(recon_batch, df_layer1, mu, logvar)

            loss_cl = torch.tensor(0.0, device=device)
            weighted_cl_loss = torch.tensor(0.0, device=device)
            weighted_ksloss = torch.tensor(0.0, device=device)
            weighted_cov_loss = torch.tensor(0.0, device=device)
            weighted_loss_recons = torch.tensor(0.0, device=device)
            cl_grad_norm = 0.0
            cgmvae_grad_norm = 0.0
            grad_cos_sim = np.nan
            mu_grad_norm = 0.0
            balanced_cgmvae_loss = losses
            balanced_cl_loss = torch.tensor(0.0, device=device)

        losses.backward()

        encoder_module = model.encoder1 if hasattr(model, "encoder1") else model
        total_grad_norm = get_grad_norm(encoder_module)
        ratio = cl_grad_norm / (total_grad_norm + 1e-12)
        cgmvae_ratio = cgmvae_grad_norm / (total_grad_norm + 1e-12)

        train_loss.append(losses.item())
        train_recon.append(loss_recons.item())
        train_cfm.append(loss_cfm.item())
        train_KL.append(loss_KL.item())
        train_CL.append(weighted_cl_loss.item())
        train_weighted_ksloss.append(weighted_ksloss.item())
        train_weighted_cov_loss.append(weighted_cov_loss.item())
        train_weighted_loss_recons.append(weighted_loss_recons.item())
        train_osd_loss.append(raw_osd_loss.detach().cpu().item())
        train_osd_value.append(osd_value.detach().cpu().item())

        train_cl_grad_norms.append(cl_grad_norm)
        train_total_grad_norms.append(total_grad_norm)
        train_mu_grad_norms.append(mu_grad_norm)
        train_cl_to_total_ratios.append(ratio)

        train_balanced_cgmvae_losses.append(balanced_cgmvae_loss.item())
        train_balanced_cl_losses.append(balanced_cl_loss.item())
        train_effective_cl_losses.append(effective_cl_loss.item())
        train_cgmvae_grad_norms.append(cgmvae_grad_norm)
        train_grad_cos_sims.append(grad_cos_sim)
        train_cgmvae_to_total_ratios.append(cgmvae_ratio)

        optimizer.step()

    model.eval()

    print(f"====> Epoch: {epoch} Train loss: {np.mean(train_loss):.4f}")
    print(f"             {epoch} Rec countmat loss: {np.mean(train_recon):.10f}")
    print(f"             {epoch} Rec cfm loss: {np.mean(train_cfm):.10f}")
    print(f"             {epoch} KL loss: {np.mean(train_KL):.10f}")
    print(f"             {epoch} CL loss: {np.mean(train_CL):.10f}")
    print(f"             {epoch} OSD value: {np.mean(train_osd_value):.6f}")
    print(f"             {epoch} OSD loss: {np.mean(train_osd_loss):.10f}")
    print(f"             {epoch} CL grad norm: {np.mean(train_cl_grad_norms):.6f}")
    print(f"             {epoch} Total grad norm: {np.mean(train_total_grad_norms):.6f}")
    print(f"             {epoch} mu grad norm: {np.mean(train_mu_grad_norms):.6f}")
    print(f"             {epoch} CL/Total ratio: {np.mean(train_cl_to_total_ratios):.6f}")
    print(f"             {epoch} Balanced CGMVAE loss: {np.mean(train_balanced_cgmvae_losses):.10f}")
    print(f"             {epoch} CL scale: {cl_scale:.6f}")
    print(f"             {epoch} Balanced CL loss: {np.mean(train_balanced_cl_losses):.10f}")
    print(f"             {epoch} Effective CL loss: {np.mean(train_effective_cl_losses):.10f}")
    print(f"             {epoch} CGMVAE grad norm: {np.mean(train_cgmvae_grad_norms):.6f}")
    print(f"             {epoch} CGMVAE/Total ratio: {np.mean(train_cgmvae_to_total_ratios):.6f}")
    print(f"             {epoch} Grad cosine sim: {np.nanmean(train_grad_cos_sims):.6f}")

    train_losses.append(np.mean(train_loss))
    train_recons.append(np.mean(train_recon))
    train_cfms.append(np.mean(train_cfm))
    train_KLs.append(np.mean(train_KL))
    train_CLs.append(np.mean(train_CL))
    train_weighted_kslosses.append(np.mean(train_weighted_ksloss))
    train_weighted_cov_losses.append(np.mean(train_weighted_cov_loss))
    train_weighted_losses_recons.append(np.mean(train_weighted_loss_recons))
    train_osd_losses.append(np.mean(train_osd_loss))
    train_osd_values.append(np.mean(train_osd_value))

    train_epoch_cl_grad_norms.append(np.mean(train_cl_grad_norms))
    train_epoch_total_grad_norms.append(np.mean(train_total_grad_norms))
    train_epoch_mu_grad_norms.append(np.mean(train_mu_grad_norms))
    train_epoch_cl_to_total_ratios.append(np.mean(train_cl_to_total_ratios))
    train_epoch_cgmvae_grad_norms.append(np.mean(train_cgmvae_grad_norms))
    train_epoch_grad_cos_sims.append(np.nanmean(train_grad_cos_sims))
    train_epoch_cgmvae_to_total_ratios.append(np.mean(train_cgmvae_to_total_ratios))
    train_epoch_balanced_cgmvae_losses.append(np.mean(train_balanced_cgmvae_losses))
    train_epoch_balanced_cl_losses.append(np.mean(train_balanced_cl_losses))
    train_epoch_effective_cl_losses.append(np.mean(train_effective_cl_losses))

    if epoch in cfg.save_epochs: 
        try:
            print(f"Saving outputs for epoch {epoch}...")
            with torch.no_grad():
                mu, log, z, recons = model.forward(test_set.x, test_set.R_S_label, test_set.cfm)
                p_all = model.projection_head(mu)

            meta_columns = [
                "barcode", "subclass", "cell_type", "SCBID", "MouseID",
                "Age", "Strain", "Genotype", "14-6-Bin"
            ]
            meta_df = df[meta_columns].reset_index(drop=True)

            recons_np = recons.detach().cpu().numpy()
            x_hat = recons_np[:, :D_out]
            cfm_hat = recons_np[:, D_out]

            recons_df = pd.DataFrame(x_hat)
            recons_df["CFM"] = cfm_hat
            recons_df = pd.concat([recons_df, meta_df], axis=1)
            recons_df.to_csv(os.path.join(default_path, f"recons_epoch{epoch}.csv"))
            print("Reconstructed features saved")

            latent_vectors_np = z.detach().cpu().numpy()
            latent_space = pd.DataFrame(
                latent_vectors_np, columns=[f"LV{i+1}" for i in range(cfg.latent_dim)]
            )
            latent_space["CFM"] = cfm_hat
            latent_space = pd.concat([latent_space, meta_df], axis=1)
            latent_space.to_csv(os.path.join(default_path, f"latent_variables_epoch{epoch}.csv"))
            print("Latent variables saved")

            latent_mu = mu.detach().cpu().numpy()
            latent_mu_space = pd.DataFrame(
                latent_mu, columns=[f"mu_{i+1}" for i in range(cfg.latent_dim)]
            )
            latent_mu_space["CFM"] = cfm_hat
            latent_mu_space = pd.concat([latent_mu_space, meta_df], axis=1)
            latent_mu_space.to_csv(os.path.join(default_path, f"latent_mu_epoch{epoch}.csv"))
            print("Latent mu saved")

            p_np = p_all.detach().cpu().numpy()
            p_space = pd.DataFrame(
                p_np, columns=[f"p_{i+1}" for i in range(cfg.proj_out_dim)]
            )
            p_space["CFM"] = cfm_hat
            p_space = pd.concat([p_space, meta_df], axis=1)
            p_space.to_csv(os.path.join(default_path, f"projection_head_epoch{epoch}.csv"))
            print("Projection head outputs saved")

            #if epoch == cfg.num_epoch:
            torch.save(model, os.path.join(default_path, f"saved_model_epoch{epoch}.pth"))
            print("Final model path saved")

            loss_frame = pd.DataFrame(
                list(
                    zip_longest(
                        train_losses,
                        train_recons,
                        train_cfms,
                        train_KLs,
                        train_CLs,
                        train_weighted_kslosses,
                        train_weighted_cov_losses,
                        train_weighted_losses_recons,
                        train_epoch_balanced_cgmvae_losses,
                        train_epoch_balanced_cl_losses,
                        train_epoch_effective_cl_losses,
                        train_osd_losses,
                        train_osd_values,
                        train_epoch_cl_grad_norms,
                        train_epoch_cgmvae_grad_norms,
                        train_epoch_total_grad_norms,
                        train_epoch_mu_grad_norms,
                        train_epoch_cl_to_total_ratios,
                        train_epoch_cgmvae_to_total_ratios,
                        train_epoch_grad_cos_sims,
                        fillvalue=np.nan,
                    )
                ),
                columns=[
                    "train_losses",
                    "train_recons",
                    "train_cfms",
                    "train_KLs",
                    "train_CLs",
                    "train_weighted_kslosses",
                    "train_weighted_cov_losses",
                    "train_weighted_losses_recons",
                    "balanced_cgmvae_loss",
                    "balanced_cl_loss",
                    "effective_cl_losses",
                    "train_osd_losses",
                    "train_osd_values",
                    "cl_grad_norm",
                    "cgmvae_grad_norm",
                    "total_grad_norm",
                    "mu_grad_norm",
                    "cl_to_total_ratio",
                    "cgmvae_to_total_ratio",
                    "grad_cos_sim",
                ],
            )
            loss_frame.to_csv(os.path.join(default_path, "loss_over_epochs.csv"))
            print("Tracked variables saved")

        except Exception as e:
            print(f"Error during saving at epoch {epoch}: {e}")
