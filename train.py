import os
import sys
import time
import random
import warnings
import threading
import csv
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
#from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler  # Only for 100 epoch run - to track compute/memory/overhead 

warnings.filterwarnings("ignore")

from VAE_model_architectures import *
from latent_regularizer import *
from loss_cl import *
import train_utils

cfg = train_utils.Config()
device = torch.device(cfg.device)

train_utils.cfg = cfg
train_utils.device = torch.device(cfg.device)

os.environ["PYTHONHASHSEED"] = str(cfg.seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
train_utils.set_seed(cfg.seed)

# Benchmark logging setup 
process = psutil.Process(os.getpid())

benchmark_log = {
    "cpu_percent_samples": [],
    "ram_used_gb_samples": [],
    "ram_percent_samples": [],
}

stop_monitoring = False

def monitor_cpu_ram(interval=5.0): # Logs every 5 seconds
    while not stop_monitoring:
        benchmark_log["cpu_percent_samples"].append(process.cpu_percent(interval=None))
        mem = process.memory_info().rss / 1024**3
        benchmark_log["ram_used_gb_samples"].append(mem)
        benchmark_log["ram_percent_samples"].append(psutil.virtual_memory().percent)
        time.sleep(interval)

df = pd.read_csv(cfg.data_path, index_col=0, dtype={"Strain": str})
df_cfm = df[["Strain", "CFM_14_5_snRNA", "14-6-residual", "14-6-Bin"]].copy()
df_cfm = df_cfm.sort_values(by="CFM_14_5_snRNA", ascending=True)
hue_order = df_cfm["Strain"].unique()
df["CFM_14_5_snRNA"] = train_utils.standardizer(df["CFM_14_5_snRNA"].values)
df = df.sample(frac=1, random_state=cfg.seed)
df_train = df

df_gmm = pd.read_csv(cfg.gmm_path)
gmm_centers = torch.tensor(
    [df_gmm[f"center{i}"].values for i in range(1, cfg.num_gaussian + 1)]
).float()
gmm_centers = gmm_centers.to(device)
ks_weight, cv_weight = estimate_loss_coefficients(
    cfg.batch_size, gmm_centers, cfg.gmm_std, num_samples=100
)
print(f"KS weight: {ks_weight}")
print(f"CV weight: {cv_weight}")
sorted_values = np.sort(df["CFM_14_5_snRNA"].dropna().unique())
print(f"Sorted Unique Standardized Input CFM values: {sorted_values} across {len(sorted_values)} strains")

train_set = train_utils.DataBuilder(df_train)
test_set = train_utils.DataBuilder(df_train)
g = torch.Generator()
g.manual_seed(cfg.seed)
trainloader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=0,
    worker_init_fn=train_utils._seed_worker,
    generator=g,
)
testloader = DataLoader(
    test_set,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=0,
    worker_init_fn=train_utils._seed_worker,
    generator=g,
)

D_in = train_set.x.shape[1]
D_out = D_in

if cfg.model_type == "C-GMVAE":
    model = Autoencoder_CGMVAE(D_in, D_out, cfg.H1, cfg.H2, cfg.H3, cfg.latent_dim).to(device)
elif cfg.model_type == "GMVAE":
    model = Autoencoder_GMVAE(D_in, D_out, cfg.H1, cfg.H2, cfg.H3, cfg.latent_dim).to(device)
elif cfg.model_type == "VAE":
    model = Autoencoder_VAE(D_in, D_out, cfg.H1, cfg.H2, cfg.H3, cfg.latent_dim).to(device)
elif cfg.model_type == "C-VAE":
    model = Autoencoder_CVAE(D_in, D_out, cfg.H1, cfg.H2, cfg.H3, cfg.latent_dim).to(device)
else:
    raise ValueError("Invalid model_type")

if cfg.use_contrastive and (cfg.use_supcon_knn or cfg.use_weighted_nt_xent): # Projection head needs to be initialized along with CGMVAE before creating optimizer
    model.projection_head = ProjectionHead(
        in_dim=cfg.latent_dim,
        hidden_dim=cfg.proj_hidden_dim,
        out_dim=cfg.proj_out_dim,
    ).to(device)

optimizer = train_utils.build_optimizer(model, cfg)
scheduler = train_utils.build_scheduler(optimizer, cfg)

os.makedirs(cfg.output_dir, exist_ok=True)

train_losses, train_recons, train_cfms, train_KLs, train_CLs = [], [], [], [], []
train_weighted_kslosses, train_weighted_cov_losses, train_weighted_losses_recons = [], [], []
train_osd_losses, train_osd_values = [], []
train_epoch_cl_grad_norms, train_epoch_total_grad_norms, train_epoch_mu_grad_norms = [], [], []
train_epoch_cl_to_total_ratios = []
train_epoch_cgmvae_grad_norms, train_epoch_grad_cos_sims = [], []
train_epoch_cgmvae_to_total_ratios = []
train_epoch_balanced_cgmvae_losses, train_epoch_balanced_cl_losses, train_epoch_effective_cl_losses = [], [], []

equal_start_initialized = False
alpha_cgmvae = cfg.init_alpha_cgmvae
beta_cl = cfg.init_beta_cl
gamma_osd = cfg.init_gamma_osd

train_utils.model = model
train_utils.optimizer = optimizer
train_utils.scheduler = scheduler
train_utils.trainloader = trainloader
train_utils.testloader = testloader
train_utils.train_set = train_set
train_utils.test_set = test_set
train_utils.df = df
train_utils.gmm_centers = gmm_centers
train_utils.ks_weight = ks_weight
train_utils.cv_weight = cv_weight
train_utils.D_out = D_out

train_utils.train_losses = train_losses
train_utils.train_recons = train_recons
train_utils.train_cfms = train_cfms
train_utils.train_KLs = train_KLs
train_utils.train_CLs = train_CLs
train_utils.train_weighted_kslosses = train_weighted_kslosses
train_utils.train_weighted_cov_losses = train_weighted_cov_losses
train_utils.train_weighted_losses_recons = train_weighted_losses_recons
train_utils.train_osd_losses = train_osd_losses
train_utils.train_osd_values = train_osd_values
train_utils.train_epoch_cl_grad_norms = train_epoch_cl_grad_norms
train_utils.train_epoch_total_grad_norms = train_epoch_total_grad_norms
train_utils.train_epoch_mu_grad_norms = train_epoch_mu_grad_norms
train_utils.train_epoch_cl_to_total_ratios = train_epoch_cl_to_total_ratios
train_utils.train_epoch_cgmvae_grad_norms = train_epoch_cgmvae_grad_norms
train_utils.train_epoch_grad_cos_sims = train_epoch_grad_cos_sims
train_utils.train_epoch_cgmvae_to_total_ratios = train_epoch_cgmvae_to_total_ratios
train_utils.train_epoch_balanced_cgmvae_losses = train_epoch_balanced_cgmvae_losses
train_utils.train_epoch_balanced_cl_losses = train_epoch_balanced_cl_losses
train_utils.train_epoch_effective_cl_losses = train_epoch_effective_cl_losses

train_utils.equal_start_initialized = equal_start_initialized
train_utils.alpha_cgmvae = alpha_cgmvae
train_utils.beta_cl = beta_cl
train_utils.gamma_osd = gamma_osd

start_time = time.time()
start_datetime = datetime.now()

# Start CPU/RAM monitoring
process.cpu_percent(interval=None)
monitor_thread = threading.Thread(target=monitor_cpu_ram, daemon=True)
monitor_thread.start()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    
epoch_times = []

#with profile( # Only for 100 epoch run - to track compute/memory/overhead 
#    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
#    on_trace_ready=tensorboard_trace_handler(os.path.join(cfg.output_dir, "profiler")),
#    record_shapes=True,
#    profile_memory=True,
#    with_stack=False
#) as prof:
    
for epoch in range(1, cfg.num_epoch + 1):
    epoch_start = time.time()
    
    train_utils.train(epoch, cfg.output_dir)
    if scheduler is not None:
        scheduler.step()
        
    epoch_end = time.time()
    epoch_times.append(epoch_end - epoch_start)
    #prof.step()  # Only for 100 epoch run - to track compute/memory/overhead 

end_time = time.time()
end_datetime = datetime.now()

total_time_sec = end_time - start_time
total_time_min = total_time_sec / 60
total_time_hr = total_time_sec / 3600

# Stop CPU/RAM monitoring
stop_monitoring = True
monitor_thread.join(timeout=2)

cpu_avg = np.mean(benchmark_log["cpu_percent_samples"]) if benchmark_log["cpu_percent_samples"] else np.nan
cpu_max = np.max(benchmark_log["cpu_percent_samples"]) if benchmark_log["cpu_percent_samples"] else np.nan

ram_avg_gb = np.mean(benchmark_log["ram_used_gb_samples"]) if benchmark_log["ram_used_gb_samples"] else np.nan
ram_max_gb = np.max(benchmark_log["ram_used_gb_samples"]) if benchmark_log["ram_used_gb_samples"] else np.nan
ram_avg_percent = np.mean(benchmark_log["ram_percent_samples"]) if benchmark_log["ram_percent_samples"] else np.nan
ram_max_percent = np.max(benchmark_log["ram_percent_samples"]) if benchmark_log["ram_percent_samples"] else np.nan

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
    gpu_peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
else:
    gpu_name = "CPU only"
    gpu_peak_allocated_gb = np.nan
    gpu_peak_reserved_gb = np.nan

logs_file = os.path.join(cfg.output_dir, "training_logs.txt")

with open(logs_file, "w") as f:
    f.write("===== Configs =====\n")
    for key, value in vars(cfg).items():
        f.write(f"{key}: {value}\n")
    f.write("\n===== Loss Scaling =====\n")
    f.write(f"alpha_cgmvae: {train_utils.alpha_cgmvae:.8f}\n")
    f.write(f"beta_cl: {train_utils.beta_cl:.8f}\n")
    f.write(f"gamma_osd: {train_utils.gamma_osd:.8f}\n")
    f.write("\n===== Training time =====\n")
    f.write(f"Training start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Training end time:   {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Total training time (seconds): {total_time_sec:.2f}\n")
    f.write(f"Total training time (minutes): {total_time_min:.2f}\n")
    f.write(f"Total training time (hours):   {total_time_hr:.2f}\n")
    f.write(f"Avg seconds per epoch: {np.mean(epoch_times):.4f}\n")
    f.write(f"Min seconds per epoch: {np.min(epoch_times):.4f}\n")
    f.write(f"Max seconds per epoch: {np.max(epoch_times):.4f}\n")
    f.write("\n===== Hardware Benchmark =====\n")
    f.write(f"Device used: {device}\n")
    f.write(f"GPU name: {gpu_name}\n")
    f.write(f"CPU avg percent: {cpu_avg:.2f}\n")
    f.write(f"CPU max percent: {cpu_max:.2f}\n")
    f.write(f"Process RAM avg GB: {ram_avg_gb:.4f}\n")
    f.write(f"Process RAM max GB: {ram_max_gb:.4f}\n")
    f.write(f"System RAM avg percent: {ram_avg_percent:.2f}\n")
    f.write(f"System RAM max percent: {ram_max_percent:.2f}\n")
    f.write(f"GPU peak memory allocated GB: {gpu_peak_allocated_gb:.4f}\n")
    f.write(f"GPU peak memory reserved GB: {gpu_peak_reserved_gb:.4f}\n")

print(f"Training started: {start_datetime}")
print(f"Training finished: {end_datetime}")
print(f"Total time: {total_time_hr:.2f} hours")
print("Avg seconds/epoch:", np.mean(epoch_times))
print("Max seconds/epoch:", np.max(epoch_times))
print("Min seconds/epoch:", np.min(epoch_times))
print(f"Saved to: {logs_file}")
print(f"Device used: {device}")
print(f"GPU name: {gpu_name}")
print(f"CPU avg percent: {cpu_avg:.2f}")
print(f"CPU max percent: {cpu_max:.2f}")
print(f"Process RAM avg GB: {ram_avg_gb:.4f}")
print(f"Process RAM max GB: {ram_max_gb:.4f}")
print(f"System RAM avg percent: {ram_avg_percent:.2f}")
print(f"System RAM max percent: {ram_max_percent:.2f}")
print(f"GPU peak memory allocated GB: {gpu_peak_allocated_gb:.4f}")
print(f"GPU peak memory reserved GB: {gpu_peak_reserved_gb:.4f}")
