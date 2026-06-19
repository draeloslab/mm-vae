#!/usr/bin/env python3
"""

Downstream analysis pipeline for trained C-GMVAE. Involves:
  1) Generating density-guided latent spac e trajectories from a trained C-GMVAE
  2) Reconstructing trajectory gene features back to original gene space
  3) Curating trajectory-derived gene lists 
  4) Computing Integrated Gradients gene-to-LV importance

Note:
  - Edit only the ExperimentConfig block or pass args from the command line. 
"""
from __future__ import annotations

import argparse
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import joblib
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde, pearsonr
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from VAE_model_architectures import Autoencoder_CGMVAE
except Exception:
    Autoencoder_CGMVAE = None


@dataclass
class ExperimentConfig:
    """
    All paths and hyperparameters used across the full pipeline.
    Only edit this block for a new run, or override values using command-line args.
    """
    seed: int = 29
    device: str = "cpu"
    run_name: str = "cgmvae_pipeline"

    project_dir: Path = Path("/home/varnika/draelos_lab/proj_VAE")
    output_root: Path = Path("/home/varnika/draelos_lab/proj_VAE/results/experiments/training23_supcon_knn16_t0.2_b512_ep5k_seed29_normalstart")
    model_dir: Path = Path("/home/varnika/draelos_lab/proj_VAE/results/experiments/training23_supcon_knn16_t0.2_b512_ep5k_seed29_normalstart")
    data_dir: Path = Path("/nfs/turbo/umms-kaczoro/u19-shared/vae-mouse-hc-updated/14_mon_HC_ADBXD_subset")

    model_path: Path = Path("/home/varnika/draelos_lab/proj_VAE/results/experiments/training23_supcon_knn16_t0.2_b512_ep5k_seed29_normalstart/saved_model_epoch5000.pth")
    rp_gene_csv: Path = Path("/nfs/turbo/umms-kaczoro/u19-shared/vae-mouse-hc-updated/14_mon_HC_ADBXD_subset/all_cells_adbxd_14mon_rp_nor.csv")
    cfm_csv: Path = Path("/nfs/turbo/umms-kaczoro/u19-shared/vae-mouse-hc-updated/14_mon_HC_ADBXD_subset/all_cells_adbxd_14mon_rp_nor.csv")
    latent_csv: Path = Path("/home/varnika/draelos_lab/proj_VAE/results/experiments/training23_supcon_knn16_t0.2_b512_ep5k_seed29_normalstart/latent_variables_epoch5000.csv")
    recons_csv: Path = Path("/home/varnika/draelos_lab/proj_VAE/results/experiments/training23_supcon_knn16_t0.2_b512_ep5k_seed29_normalstart/recons_epoch5000.csv")

    rp_pipeline_path: Path = Path("/nfs/turbo/umms-kaczoro/u19-shared/vae-mouse-hc-updated/14_mon_HC_ADBXD_subset/random_projection.pkl")
    gene_list_path: Path = Path("/nfs/turbo/umms-kaczoro/u19-shared/vae-mouse-hc-updated/14_mon_HC_ADBXD_subset/gene_list.csv")
    original_gene_csv: Optional[Path] = None  # only needed if gene_list_path is not enough

    label_col: str = "14-6-Bin"
    cfm_col: str = "14-6-residual"
    raw_cfm_col: str = "CFM_14_5_snRNA"
    decoded_cfm_col: str = "CFM"  

    gene_count: int = 4842
    latent_dim: int = 10
    h1: int = 128
    h2: int = 64
    h3: int = 32

    # ~~~~ Density-guided interpolation ~~~~
    num_pairs: int = 10
    num_interpolation_steps: int = 48
    density_weights: List[float] = field(default_factory=lambda: [0.1, 0.4, 0.5, 0.9])
    selected_density: float = 0.4
    knn: int = 5
    endpoint_pool_size: int = 100
    generate_reverse: bool = True

    # ~~~~ Gene list curation ~~~~
    percentile_cutoff: float = 99.9
    max_eff_corr_threshold: float = 0.65
    trajectory_count_total: int = 20

    filter_gm_genes: bool = True
    filter_riken_genes: bool = True
    filter_loc_genes: bool = True
    
    # Options - "local", "mygene", "mousemine", "mygene_then_mousemine"
    # Intermine/mousemine is not working properly as it is not supported on Python 3.11 -- yet to debug.
    # Use "mygene" in the meantime instead of "mousemine", "mygene_then_mousemine"
    require_annotated_gene: bool = True
    require_protein_coding: bool = True
    
    gene_annotation_backend: str = "mygene"
    mygene_species: str = "mouse"
    mousemine_url: str = "https://www.mousemine.org/mousemine/service"
    save_gene_annotation_table: bool = True

    # ~~~~ Integrated gradients ~~~~
    run_ig: bool = True
    ig_batch_size: int = 512
    ig_m_steps: int = 50
    ig_baseline_type: str = "zero"
    ig_clip_quantile: Optional[float] = 0.99
    ig_gamma: float = 10.0
    ig_min_genes: int = 10

    # ---- Output subdirectories ----
    trajectories_dirname: str = "reconstructed_pairs"
    reverse_trajectories_dirname: str = "reconstructed_pairs_reverse"
    original_space_dirname: str = "reconstructed_pairs_hd"
    gene_lists_dirname: str = "Gene_lists_0.65"
    ig_dirname: str = "interpretability_outputs"

    @property
    def trajectories_dir(self) -> Path:
        return self.output_root / self.trajectories_dirname

    @property
    def reverse_trajectories_dir(self) -> Path:
        return self.output_root / self.reverse_trajectories_dirname

    @property
    def original_space_dir(self) -> Path:
        return self.output_root / self.original_space_dirname

    @property
    def gene_lists_dir(self) -> Path:
        return self.output_root / self.gene_lists_dirname

    @property
    def ig_dir(self) -> Path:
        return self.output_root / self.ig_dirname


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def standardize(x: np.ndarray) -> np.ndarray:
    denom = np.std(x)
    if np.isclose(denom, 0):
        return x - np.mean(x)
    return (x - np.mean(x)) / denom


def safe_pearsonr(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    if np.isclose(x[mask].std(), 0) or np.isclose(y[mask].std(), 0):
        return np.nan
    return pearsonr(x[mask], y[mask])[0]


def infer_cfm_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find CFM column. Tried: {list(candidates)}. Available: {list(df.columns[-20:])}")


def list_gene_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("gene_")]


class MouseGeneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dim_count: int, label_col: str, cfm_col: str, device: str = "cpu"):
        self.dim_count = dim_count
        self.label_col = label_col
        self.cfm_col = cfm_col
        self.device = device

        self.x = torch.tensor(df.iloc[:, :dim_count].values.astype("float32"), device=device)
        self.R_S_label = torch.tensor(df[label_col].values, device=device)
        self.cfm = torch.tensor(standardize(df[cfm_col].values).astype("float32"), device=device)
        self.len = self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.R_S_label[index], self.cfm[index]

    def __len__(self) -> int:
        return self.len


def load_mouse_gene_data(cfg: ExperimentConfig) -> Tuple[DataLoader, MouseGeneDataset, pd.DataFrame]:
    df = pd.read_csv(cfg.rp_gene_csv, index_col=0, dtype={"Strain": str})
    df = df.sample(frac=1, random_state=42)

    dataset = MouseGeneDataset(
        df=df,
        dim_count=cfg.gene_count,
        label_col=cfg.label_col,
        cfm_col=cfg.cfm_col,
        device=cfg.device,
    )

    generator = torch.Generator().manual_seed(cfg.seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = cfg.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=cfg.ig_batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    return loader, dataset, df


def load_model(cfg: ExperimentConfig) -> nn.Module:
    model = torch.load(cfg.model_path, map_location=cfg.device, weights_only=False)
    model.to(cfg.device)
    model.eval()
    return model


def load_latent_dataframe(cfg: ExperimentConfig) -> pd.DataFrame:
    df_latent = pd.read_csv(cfg.latent_csv, index_col=0)

    latent_cols = list(df_latent.columns[: cfg.latent_dim])
    keep_cols = latent_cols + [cfg.label_col]
    df_latent = df_latent[keep_cols].copy()
    df_latent.columns = [f"z_{i+1}" for i in range(cfg.latent_dim)] + ["label"]

    df_recons = pd.read_csv(cfg.recons_csv)
    cfm_col = infer_cfm_column(df_recons, [cfg.decoded_cfm_col, "cfm", "CFM", "decoded_cfm", cfg.raw_cfm_col])
    df_latent["decoded_cfm"] = df_recons[cfm_col].values

    return df_latent


def get_latent_interpolation_endpoints(
    df_latent: pd.DataFrame,
    latent_dim: int,
    num_pairs: int,
    endpoint_pool_size: int,
    seed: int,
) -> List[Tuple[dict, dict]]:
    pairs = []
    df_sorted = df_latent.sort_values(by="decoded_cfm", ignore_index=True)

    low_candidates = df_sorted.head(endpoint_pool_size)
    high_candidates = df_sorted.tail(endpoint_pool_size)

    for i in range(num_pairs):
        low_sample = low_candidates.sample(1, random_state=seed + i).iloc[0]
        high_sample = high_candidates.sample(1, random_state=seed + 1000 + i).iloc[0]

        tries = 0
        while low_sample["label"] == high_sample["label"] and tries < 10:
            high_sample = high_candidates.sample(1, random_state=seed + 2000 + i + tries).iloc[0]
            tries += 1

        low = {
            "z": low_sample[[f"z_{j+1}" for j in range(latent_dim)]].values.astype("float32"),
            "label": int(low_sample["label"]),
            "cfm": float(low_sample["decoded_cfm"]),
        }
        high = {
            "z": high_sample[[f"z_{j+1}" for j in range(latent_dim)]].values.astype("float32"),
            "label": int(high_sample["label"]),
            "cfm": float(high_sample["decoded_cfm"]),
        }
        pairs.append((low, high))

    return pairs


def log_density(z_point: np.ndarray, kde: gaussian_kde, eps: float = 1e-9) -> float:
    z_point = z_point.reshape(-1, 1)
    return float(np.log(kde(z_point)[0] + eps))


def numerical_log_density_gradient(z_point: np.ndarray, kde: gaussian_kde, epsilon: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(z_point)
    for j in range(len(z_point)):
        perturb = np.zeros_like(z_point)
        perturb[j] = epsilon
        grad[j] = (log_density(z_point + perturb, kde) - log_density(z_point - perturb, kde)) / (2.0 * epsilon)
    return grad


def density_guided_interpolation(
    start_point: np.ndarray,
    end_point: np.ndarray,
    latent_space: np.ndarray,
    steps: int,
    step_size: float,
    density_weight: float,
    log_density_drop_threshold: float = 0.5,
    min_step_size: float = 1e-2,
) -> np.ndarray:
    kde = gaussian_kde(latent_space.T)

    path = [start_point]
    current_point = start_point.copy()

    for _ in range(steps):
        cur_step_size = step_size

        target_dir = end_point - current_point
        target_norm = np.linalg.norm(target_dir)
        if target_norm > 0:
            target_dir = target_dir / target_norm

        density_grad = numerical_log_density_gradient(current_point, kde)
        grad_norm = np.linalg.norm(density_grad)
        if grad_norm > 0:
            density_grad = density_grad / grad_norm

        move_dir = (1.0 - density_weight) * target_dir + density_weight * density_grad
        move_norm = np.linalg.norm(move_dir)
        if move_norm > 0:
            move_dir = move_dir / move_norm

        proposed = current_point + move_dir * cur_step_size

        if log_density(proposed, kde) < log_density(current_point, kde) - log_density_drop_threshold:
            cur_step_size = max(cur_step_size * 0.9, min_step_size)

        next_point = current_point + move_dir * cur_step_size
        path.append(next_point)
        current_point = next_point

    path.append(end_point)
    return np.asarray(path)


def compute_interpolations_between_points(
    start_point: np.ndarray,
    end_point: np.ndarray,
    latent_space: np.ndarray,
    num_steps: int,
    density_weights: Sequence[float],
) -> Dict[float, np.ndarray]:
    total_distance = np.linalg.norm(end_point - start_point)
    if total_distance < 1e-6:
        return {dw: np.tile(start_point, (num_steps + 2, 1)) for dw in density_weights}

    step_size = total_distance / num_steps
    return {
        dw: density_guided_interpolation(start_point, end_point, latent_space, num_steps, step_size, dw)
        for dw in density_weights
    }


def assign_labels_dirichlet_expected(
    interpolated_path: np.ndarray,
    latent_space: np.ndarray,
    labels: np.ndarray,
    k: int,
    alpha_prior: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=k).fit(latent_space)
    all_labels = np.unique(labels)

    assigned_labels, all_probs, uncertainties = [], [], []

    for z in interpolated_path:
        _, idx = nbrs.kneighbors(z.reshape(1, -1))
        neighbor_labels = labels[idx[0]]
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))

        alpha_vec = np.array([label_counts.get(lbl, 0) + alpha_prior for lbl in all_labels])
        probs = alpha_vec / np.sum(alpha_vec)
        assigned = all_labels[np.argmax(probs)]
        entropy = -np.sum(probs * np.log(probs + 1e-8))

        assigned_labels.append(assigned)
        all_probs.append(probs)
        uncertainties.append(entropy)

    return np.asarray(assigned_labels), np.vstack(all_probs), np.asarray(uncertainties)


def decode_trajectory(model: nn.Module, z_path: np.ndarray, labels: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    z_tensor = torch.tensor(z_path, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    with torch.no_grad():
        decoded = model.decode(z_tensor, y_tensor).detach().cpu().numpy()

    gene_recon = decoded[:, :-1]
    decoded_cfm = decoded[:, -1]
    return gene_recon, decoded_cfm


def generate_all_interpolations(
    cfg: ExperimentConfig,
    model: nn.Module,
    df_latent: pd.DataFrame,
    reverse: bool = False,
) -> Dict[str, dict]:
    latent_space = df_latent[[f"z_{j+1}" for j in range(cfg.latent_dim)]].values.astype("float32")
    labels_all = df_latent["label"].values

    pairs = get_latent_interpolation_endpoints(
        df_latent=df_latent,
        latent_dim=cfg.latent_dim,
        num_pairs=cfg.num_pairs,
        endpoint_pool_size=cfg.endpoint_pool_size,
        seed=cfg.seed,
    )

    all_interpolations = {}

    for i, (low, high) in enumerate(pairs, start=1):
        start_point, end_point = (high["z"], low["z"]) if reverse else (low["z"], high["z"])

        paths = compute_interpolations_between_points(
            start_point=start_point,
            end_point=end_point,
            latent_space=latent_space,
            num_steps=cfg.num_interpolation_steps,
            density_weights=cfg.density_weights,
        )

        label_assignments = {}
        for dw, path in paths.items():
            assigned_labels, label_probs, uncertainties = assign_labels_dirichlet_expected(
                interpolated_path=path,
                latent_space=latent_space,
                labels=labels_all,
                k=cfg.knn,
                alpha_prior=1.0,
            )
            gene_recon, decoded_cfm = decode_trajectory(model, path, assigned_labels, cfg.device)
            label_assignments[dw] = {
                "labels": assigned_labels,
                "probs": label_probs,
                "uncertainty": uncertainties,
                "decoded_cfm": decoded_cfm,
                "reconstructed_genes": gene_recon,
            }

        all_interpolations[f"pair_{i}"] = {"paths": paths, "labels": label_assignments}

    return all_interpolations


def save_full_trajectory_and_latents(
    all_interpolations: Dict[str, dict],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)

    for pair_id, data in all_interpolations.items():
        paths_dict = data["paths"]
        labels_dict = data["labels"]

        for dw, path in paths_dict.items():
            if dw not in labels_dict:
                continue

            label_data = labels_dict[dw]
            labels = np.asarray(label_data["labels"])
            probs = np.asarray(label_data["probs"])
            uncertainty = np.asarray(label_data["uncertainty"])
            cfm_values = np.asarray(label_data["decoded_cfm"])
            gene_recon = np.asarray(label_data["reconstructed_genes"])

            n_steps, latent_dim = path.shape
            df_base = pd.DataFrame({
                "pair_id": [pair_id] * n_steps,
                "density_weight": [dw] * n_steps,
                "step": np.arange(n_steps),
                "label": labels,
                "label_prob": np.max(probs, axis=1),
                "uncertainty": uncertainty,
                "decoded_cfm": cfm_values,
            })

            gene_cols = [f"gene_{j}" for j in range(gene_recon.shape[1])]
            df_recon = pd.concat([df_base, pd.DataFrame(gene_recon, columns=gene_cols)], axis=1)
            df_recon.to_csv(out_dir / f"{pair_id}_dw{dw}_recon.csv", index=False)

            latent_cols = [f"z{j}" for j in range(latent_dim)]
            df_latent_out = pd.concat([df_base, pd.DataFrame(path, columns=latent_cols)], axis=1)
            df_latent_out.to_csv(out_dir / f"{pair_id}_dw{dw}_latent.csv", index=False)


def plot_decoded_cfm(
    all_interpolations: Dict[str, dict],
    cfg: ExperimentConfig,
    out_dir: Path,
    title: str = "",
    figsize: Tuple[int, int] = (10, 7),
    smooth: bool = False,
    window: int = 5,
    poly: int = 2,
) -> None:
    ensure_dir(out_dir)

    df_cfm = pd.read_csv(cfg.cfm_csv)
    mean_cfm = df_cfm[cfg.raw_cfm_col].mean()
    std_cfm = df_cfm[cfg.raw_cfm_col].std()

    for pair_id, data in all_interpolations.items():
        cfm_dict = {
            dw: label_data["decoded_cfm"]
            for dw, label_data in data["labels"].items()
        }

        plt.figure(figsize=figsize)
        viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(cfm_dict)))

        for i, (dw, cfm_values) in enumerate(sorted(cfm_dict.items())):
            cfm_values = np.asarray(cfm_values, dtype=float)

            if smooth and len(cfm_values) >= window:
                smoothed = savgol_filter(
                    cfm_values,
                    window_length=window,
                    polyorder=poly,
                )
                smoothed[0] = cfm_values[0]
                smoothed[-1] = cfm_values[-1]
                cfm_values = smoothed

            cfm_values = cfm_values * std_cfm + mean_cfm

            plt.plot(
                range(len(cfm_values)),
                cfm_values,
                label=f"Density weight {dw}",
                color=viridis_colors[i],
                marker="o",
                linewidth=3.2,
                markersize=7,
                alpha=0.9,
            )

        plt.xlabel("Step in trajectory", fontsize=20)
        plt.ylabel("14 month CFM score", fontsize=20)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.title(title, fontsize=12)
        plt.grid(True, linestyle="-", linewidth=1.0, color="gray", alpha=0.8)
        plt.legend(
            fontsize=15,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            borderpad=1.2,
            loc="best",
        )
        plt.tight_layout()

        plt.savefig(out_dir / f"{pair_id}_decoded_cfm_trajectories.png", dpi=300, bbox_inches="tight")
        plt.savefig(out_dir / f"{pair_id}_decoded_cfm_trajectories.jpg", dpi=300, bbox_inches="tight")
        plt.close()
        

def run_trajectory_stage(cfg: ExperimentConfig, model: nn.Module) -> None:
    print("\nGenerating latent trajectories...")
    df_latent = load_latent_dataframe(cfg)

    fwd = generate_all_interpolations(cfg, model, df_latent, reverse=False)
    save_full_trajectory_and_latents(fwd, cfg.trajectories_dir)
    plot_decoded_cfm(fwd, cfg, cfg.output_root / "trajectory_plots")
    print(f"Saved forward trajectories to: {cfg.trajectories_dir}")

    if cfg.generate_reverse:
        rev = generate_all_interpolations(cfg, model, df_latent, reverse=True)
        save_full_trajectory_and_latents(rev, cfg.reverse_trajectories_dir)
        plot_decoded_cfm(rev, cfg, cfg.output_root / "trajectory_plots_reverse")
        print(f"Saved reverse trajectories to: {cfg.reverse_trajectories_dir}")


def rp_to_original_gene_space(pipeline, df_rp: pd.DataFrame) -> np.ndarray:
    rp = pipeline.named_steps["random_projection"]
    scaler = pipeline.named_steps["scaler"]

    y_scaled = df_rp.values
    y = scaler.inverse_transform(y_scaled)

    components = rp.components_  # [n_components, n_original_genes]
    x_hat = y @ np.linalg.pinv(components.T)
    return x_hat


def load_gene_mapping(cfg: ExperimentConfig, n_genes: int) -> pd.DataFrame:
    gene_list = pd.read_csv(cfg.gene_list_path)

    if "gene" in gene_list.columns:
        gene_names = gene_list["gene"].values
    else:
        gene_names = gene_list.iloc[:, 0].values

    if len(gene_names) != n_genes:
        print(f"WARNING: gene_list has {len(gene_names)} genes but reconstructed original space has {n_genes}.")

    gene_ids = [f"gene_orig_{i+1}" for i in range(len(gene_names))]
    return pd.DataFrame({"gene_id": gene_ids, "gene_name": gene_names})


def reconstruct_trajectories_to_original_space(
    cfg: ExperimentConfig,
    traj_dir: Path,
    output_dir: Path,
    pair_offset: int = 0,
) -> None:
    ensure_dir(output_dir)

    pipeline = joblib.load(cfg.rp_pipeline_path)

    rp = pipeline.named_steps["random_projection"]
    n_original_genes = rp.components_.shape[1]
    gene_mapping = load_gene_mapping(cfg, n_original_genes)
    mapping_path = output_dir / "gene_id_to_real_name_mapping.csv"
    gene_mapping.to_csv(mapping_path, index=False)

    for i in range(1, cfg.num_pairs + 1):
        traj_path = traj_dir / f"pair_{i}_dw{cfg.selected_density}_recon.csv"
        if not traj_path.exists():
            print(f"Missing trajectory file: {traj_path}")
            continue

        df_traj = pd.read_csv(traj_path)
        gene_cols = list_gene_feature_cols(df_traj)
        df_rp = df_traj[gene_cols]

        x_hat = rp_to_original_gene_space(pipeline, df_rp)
        out_cols = gene_mapping["gene_id"].values[: x_hat.shape[1]]
        df_orig = pd.DataFrame(x_hat, columns=out_cols, index=df_traj.index)

        for meta_col in ["decoded_cfm", "label"]:
            if meta_col in df_traj.columns:
                df_orig[meta_col] = df_traj[meta_col].values

        out_pair_id = i + pair_offset
        df_orig.to_csv(output_dir / f"pair{out_pair_id}_dw{cfg.selected_density}_recon_orig.csv", index=False)


def run_original_space_stage(cfg: ExperimentConfig) -> None:
    print("\nReconstructing trajectory genes to original gene space...")
    reconstruct_trajectories_to_original_space(cfg, cfg.trajectories_dir, cfg.original_space_dir, pair_offset=0)

    if cfg.generate_reverse:
        reconstruct_trajectories_to_original_space(cfg, cfg.reverse_trajectories_dir, cfg.original_space_dir, pair_offset=cfg.num_pairs)

    print(f"Saved original-space reconstructed trajectories to: {cfg.original_space_dir}")


def get_top_gene_indices_by_percentile(
    weights: pd.Series,
    percentile_cutoff: float,
) -> pd.Index:
    """
    Example:
    percentile_cutoff=99.9 keeps the top 0.1% by absolute RP weight.

    For ~31,483 genes, top 0.1% is about 32 genes.
    But this is computed dynamically from the actual number of genes.
    """
    abs_weights = weights.abs()
    n_genes = abs_weights.shape[0]

    top_fraction = (100.0 - percentile_cutoff) / 100.0
    n_top = max(1, math.ceil(n_genes * top_fraction))

    return abs_weights.nlargest(n_top).index


def filter_gene_symbols(
    gene_names: Sequence[str],
    cfg: ExperimentConfig,
) -> List[str]:
    """
    Local string-rule filtering.

    This does NOT verify whether a gene is real or protein-coding.
    It only removes symbol patterns such as Gm*, LOC*, and *Rik.
    """
    cleaned = []

    for gene in pd.Series(gene_names).dropna().astype(str).unique():
        gene = gene.strip()

        if cfg.filter_gm_genes and gene.startswith("Gm"):
            continue

        if cfg.filter_loc_genes and gene.startswith("LOC"):
            continue

        if cfg.filter_riken_genes and gene.endswith("Rik"):
            continue

        cleaned.append(gene)

    return cleaned


def annotate_genes_with_mygene(
    gene_names: Sequence[str],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    import mygene

    genes = list(pd.Series(gene_names).dropna().astype(str).unique())

    if not genes:
        return pd.DataFrame(columns=[
            "query", "found", "symbol", "name", "gene_type",
            "taxid", "entrezgene", "ensembl_gene", "annotation_backend"
        ])

    mg = mygene.MyGeneInfo()

    results = mg.querymany(
        genes,
        scopes="symbol,alias",
        fields="symbol,name,type_of_gene,taxid,entrezgene,ensembl.gene",
        species=cfg.mygene_species,
        as_dataframe=False,
        verbose=False,
    )

    rows = []

    for r in results:
        ensembl_gene = None

        if isinstance(r.get("ensembl"), dict):
            ensembl_gene = r["ensembl"].get("gene")
        elif isinstance(r.get("ensembl"), list) and len(r["ensembl"]) > 0:
            ensembl_gene = r["ensembl"][0].get("gene")

        rows.append({
            "query": r.get("query"),
            "found": not r.get("notfound", False),
            "symbol": r.get("symbol"),
            "name": r.get("name"),
            "gene_type": r.get("type_of_gene"),
            "taxid": r.get("taxid"),
            "entrezgene": r.get("entrezgene"),
            "ensembl_gene": ensembl_gene,
            "annotation_backend": "mygene",
        })

    return pd.DataFrame(rows)


def annotate_genes_with_mousemine(
    gene_names: Sequence[str],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    from intermine.webservice import Service

    genes = list(pd.Series(gene_names).dropna().astype(str).unique())

    if not genes:
        return pd.DataFrame(columns=[
            "query", "found", "symbol", "name", "gene_type",
            "mgi_id", "annotation_backend"
        ])

    service = Service(cfg.mousemine_url)
    rows = []

    for gene in genes:
        found = False
        symbol = None
        name = None
        mgi_id = None
        gene_type = None

        try:
            query = service.new_query("Gene")
            query.add_view(
                "primaryIdentifier",
                "symbol",
                "name",
                "sequenceOntologyTerm.name",
            )
            query.add_constraint("symbol", "=", gene)

            result_rows = list(query.rows())

            if len(result_rows) > 0:
                r = result_rows[0]
                found = True
                symbol = r.get("symbol")
                name = r.get("name")
                mgi_id = r.get("primaryIdentifier")
                gene_type = r.get("sequenceOntologyTerm.name")

        except Exception as e:
            print(f"MouseMine lookup failed for {gene}: {e}")

        rows.append({
            "query": gene,
            "found": found,
            "symbol": symbol,
            "name": name,
            "gene_type": gene_type,
            "mgi_id": mgi_id,
            "annotation_backend": "mousemine",
        })

    return pd.DataFrame(rows)


def annotate_genes(
    gene_names: Sequence[str],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    backend = cfg.gene_annotation_backend.lower()

    if backend == "local":
        genes = list(pd.Series(gene_names).dropna().astype(str).unique())
        return pd.DataFrame({
            "query": genes,
            "found": True,
            "symbol": genes,
            "name": None,
            "gene_type": None,
            "annotation_backend": "local",
        })

    if backend == "mygene":
        return annotate_genes_with_mygene(gene_names, cfg)

    if backend == "mousemine":
        return annotate_genes_with_mousemine(gene_names, cfg)

    if backend == "mygene_then_mousemine":
        mygene_df = annotate_genes_with_mygene(gene_names, cfg)

        unresolved = mygene_df.loc[
            mygene_df["found"] == False,
            "query"
        ].dropna().unique().tolist()

        if len(unresolved) == 0:
            return mygene_df

        mousemine_df = annotate_genes_with_mousemine(unresolved, cfg)

        resolved_by_mygene = mygene_df[mygene_df["found"] == True]

        return pd.concat(
            [resolved_by_mygene, mousemine_df],
            axis=0,
            ignore_index=True,
        )

    raise ValueError(f"Unknown gene_annotation_backend: {cfg.gene_annotation_backend}")


def filter_genes_using_annotation(
    gene_names: Sequence[str],
    cfg: ExperimentConfig,
    annotation_output_path: Optional[Path] = None,
) -> List[str]:
    """
    Full gene filtering logic:

    1. Apply optional local string filters:
       - Gm*
       - LOC*
       - *Rik

    2. Annotate genes using selected backend.

    3. Optionally require that genes are found in annotation database.

    4. Optionally require protein-coding status.

    If require_protein_coding=False, annotated non-coding genes are kept.
    """

    locally_filtered = filter_gene_symbols(gene_names, cfg)

    annotation_df = annotate_genes(locally_filtered, cfg)

    if annotation_output_path is not None and cfg.save_gene_annotation_table:
        annotation_df.to_csv(annotation_output_path, index=False)

    # If not requiring annotation or protein-coding status, this returns the locally filtered genes directly.
    if not cfg.require_annotated_gene and not cfg.require_protein_coding:
        return sorted(pd.Series(locally_filtered).dropna().astype(str).unique())

    df = annotation_df.copy()

    if cfg.require_annotated_gene:
        df = df[df["found"] == True]

    if cfg.require_protein_coding:
        df = df[
            df["gene_type"]
            .astype(str)
            .str.lower()
            .isin([
                "protein-coding",
                "protein coding gene",
                "protein_coding",
            ])
        ]

    return sorted(df["query"].dropna().astype(str).unique())


def curate_gene_lists_for_pair(
    cfg: ExperimentConfig,
    pair_idx: int,
    mapping_dict: Dict[str, str],
) -> Tuple[set, set]:

    original_path = cfg.original_space_dir / f"pair{pair_idx}_dw{cfg.selected_density}_recon_orig.csv"

    if pair_idx <= cfg.num_pairs:
        reduced_path = cfg.trajectories_dir / f"pair_{pair_idx}_dw{cfg.selected_density}_recon.csv"
    else:
        reduced_path = cfg.reverse_trajectories_dir / f"pair_{pair_idx - cfg.num_pairs}_dw{cfg.selected_density}_recon.csv"

    if not original_path.exists() or not reduced_path.exists():
        print(f"Skipping pair {pair_idx}: missing {original_path} or {reduced_path}")
        return set(), set()

    original_df = pd.read_csv(original_path)
    reduced_df = pd.read_csv(reduced_path)

    original_data = original_df[[c for c in original_df.columns if c.startswith("gene_orig_")]]
    reduced_data = reduced_df[list_gene_feature_cols(reduced_df)]
    reduced_cfm = reduced_df["decoded_cfm"]

    reduced_corr = reduced_data.apply(lambda col: safe_pearsonr(col, reduced_cfm))

    rp = joblib.load(cfg.rp_pipeline_path).named_steps["random_projection"]
    W = rp.components_

    W_df = pd.DataFrame(
        W,
        index=reduced_data.columns,
        columns=original_data.columns,
    )

    gene_max_eff_corr = defaultdict(float)
    gene_occurrences = defaultdict(int)

    for latent_gene_feature, row in W_df.iterrows():
        top_genes = get_top_gene_indices_by_percentile(
            row,
            percentile_cutoff=cfg.percentile_cutoff,
        )

        cfm_corr = reduced_corr.get(latent_gene_feature, np.nan)

        if pd.isna(cfm_corr):
            continue

        for gene in top_genes:
            weight = row[gene]
            eff_corr = cfm_corr * np.sign(weight)

            gene_occurrences[gene] += 1

            if abs(eff_corr) > abs(gene_max_eff_corr[gene]):
                gene_max_eff_corr[gene] = eff_corr

    c1_orig, c2_orig = set(), set()

    for gene, max_eff_corr in gene_max_eff_corr.items():
        if max_eff_corr >= cfg.max_eff_corr_threshold:
            c1_orig.add(gene)
        elif max_eff_corr <= -cfg.max_eff_corr_threshold:
            c2_orig.add(gene)

    def convert_and_filter(gene_set: set, direction: str) -> set:
        real_names = [mapping_dict.get(g) for g in gene_set]
        real_names = [g for g in real_names if pd.notna(g)]

        annotation_path = (
            cfg.gene_lists_dir
            / f"pair_{pair_idx}_{direction}_gene_annotations.csv"
        )

        filtered = filter_genes_using_annotation(
            real_names,
            cfg,
            annotation_output_path=annotation_path,
        )

        return set(filtered)

    return convert_and_filter(c1_orig, "C1"), convert_and_filter(c2_orig, "C2")


def write_gene_set(path: Path, genes: Iterable[str]) -> None:
    genes = sorted(set(g for g in genes if pd.notna(g)))
    pd.Series(genes).to_csv(path, index=False, header=False)


def run_gene_list_stage(cfg: ExperimentConfig) -> None:
    print("\nCurating trajectory-derived gene lists...")
    ensure_dir(cfg.gene_lists_dir)

    mapping_df = pd.read_csv(cfg.original_space_dir / "gene_id_to_real_name_mapping.csv")
    mapping_dict = dict(zip(mapping_df["gene_id"], mapping_df["gene_name"]))

    c1_sets, c2_sets = [], []

    for i in range(1, cfg.trajectory_count_total + 1):
        c1, c2 = curate_gene_lists_for_pair(cfg, i, mapping_dict)

        c1_sets.append(c1)
        c2_sets.append(c2)

        write_gene_set(cfg.gene_lists_dir / f"pair_{i}_dw{cfg.selected_density}_C1_genes.txt", c1)
        write_gene_set(cfg.gene_lists_dir / f"pair_{i}_dw{cfg.selected_density}_C2_genes.txt", c2)

        print(f"Pair {i}: C1={len(c1)} genes, C2={len(c2)} genes")

    c1_nonempty = [s for s in c1_sets if s]
    c2_nonempty = [s for s in c2_sets if s]

    c1_intersection = set.intersection(*c1_nonempty) if c1_nonempty else set()
    c2_intersection = set.intersection(*c2_nonempty) if c2_nonempty else set()

    write_gene_set(cfg.gene_lists_dir / f"intersection_dw{cfg.selected_density}_C1_genes.txt", c1_intersection)
    write_gene_set(cfg.gene_lists_dir / f"intersection_dw{cfg.selected_density}_C2_genes.txt", c2_intersection)

    overlap = c1_intersection & c2_intersection
    write_gene_set(cfg.gene_lists_dir / f"intersection_dw{cfg.selected_density}_C1_C2_overlap_genes.txt", overlap)

    print(f"C1 intersection: {len(c1_intersection)} genes")
    print(f"C2 intersection: {len(c2_intersection)} genes")
    print(f"C1/C2 overlap: {len(overlap)} genes")
    print(f"Saved gene lists to: {cfg.gene_lists_dir}")


def compute_latent_saliency_ig(
    model: nn.Module,
    test_set: MouseGeneDataset,
    feature_names: Sequence[str],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    device = cfg.device
    model.to(device)
    model.eval()

    X_all = test_set.x.to(device)
    y_all = test_set.R_S_label.to(device).long()
    cfm_all = test_set.cfm.to(device)
    N, n_features = X_all.shape

    with torch.no_grad():
        n_probe = min(10, N)
        mu_test, _, _ = model.encode(X_all[:n_probe], y_all[:n_probe], cfm_all[:n_probe])
        latent_dim = mu_test.shape[1]

    if cfg.ig_baseline_type == "mean":
        baseline_vec = X_all.mean(dim=0, keepdim=True)
    elif cfg.ig_baseline_type == "zero":
        baseline_vec = torch.zeros(1, n_features, device=device)
    else:
        raise ValueError(f"Unknown baseline type: {cfg.ig_baseline_type}")

    ig_sum_abs = torch.zeros(latent_dim, n_features, device=device)
    total_cells = 0

    for start in tqdm(range(0, N, cfg.ig_batch_size), desc="Computing IG"):
        end = min(start + cfg.ig_batch_size, N)
        x_batch = X_all[start:end]
        y_batch = y_all[start:end]
        cfm_batch = cfm_all[start:end]

        B = x_batch.shape[0]
        baseline_batch = baseline_vec.expand(B, -1)
        diff = x_batch - baseline_batch

        for k in range(latent_dim):
            grads_accum = torch.zeros_like(x_batch, device=device)

            for step in range(1, cfg.ig_m_steps + 1):
                alpha = float(step) / cfg.ig_m_steps
                x_step = baseline_batch + alpha * diff
                x_step.requires_grad_(True)

                mu_step, _, _ = model.encode(x_step, y_batch, cfm_batch)
                mu_k = mu_step[:, k].sum()

                model.zero_grad(set_to_none=True)
                if x_step.grad is not None:
                    x_step.grad.zero_()

                mu_k.backward()
                grads_accum += x_step.grad.detach()

            avg_grads = grads_accum / float(cfg.ig_m_steps)
            ig_batch = diff * avg_grads
            ig_abs = ig_batch.abs()

            if cfg.ig_clip_quantile is not None and cfg.ig_clip_quantile < 1.0:
                cap = torch.quantile(ig_abs, cfg.ig_clip_quantile, dim=0, keepdim=True)
                cap = torch.clamp(cap, min=1e-12)
                ig_abs = torch.minimum(ig_abs, cap)

            ig_sum_abs[k] += ig_abs.sum(dim=0)

        total_cells += B

    ig_avg = (ig_sum_abs / max(total_cells, 1)).detach().cpu().numpy()
    latent_names = [f"LV{i+1}" for i in range(latent_dim)]
    return pd.DataFrame(ig_avg, index=latent_names, columns=feature_names)


def compute_gene_to_lv_importance(cfg: ExperimentConfig, saliency_df_ig: pd.DataFrame) -> pd.DataFrame:
    pipeline = joblib.load(cfg.rp_pipeline_path)
    rp = pipeline.named_steps["random_projection"]

    gene_mapping = pd.read_csv(cfg.original_space_dir / "gene_id_to_real_name_mapping.csv")
    gene_names = gene_mapping["gene_name"].values

    W_df = pd.DataFrame(
        np.abs(rp.components_),
        index=saliency_df_ig.columns.astype(str),
        columns=gene_names,
    )

    saliency_df_ig.columns = saliency_df_ig.columns.astype(str)
    common_rps = saliency_df_ig.columns.intersection(W_df.index)

    A = saliency_df_ig.loc[:, common_rps].astype(float)
    B = W_df.loc[common_rps, :].astype(float)

    return A @ B


def _elbow_index_desc_chord(scores_desc: np.ndarray, min_genes: int = 5, max_search: Optional[int] = None) -> int:
    y_full = np.asarray(scores_desc, dtype=float)
    n_full = y_full.size
    if n_full == 0:
        return -1

    y = y_full if max_search is None else y_full[: min(n_full, int(max_search))]
    n = y.size
    if n <= max(2, min_genes):
        return n - 1

    x = np.arange(n, dtype=float)
    y_min, y_max = float(y.min()), float(y.max())

    if np.isclose(y_max, y_min):
        return n - 1

    y_norm = (y - y_min) / (y_max - y_min)

    x1, y1 = x[0], y_norm[0]
    x2, y2 = x[-1], y_norm[-1]

    denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    if np.isclose(denom, 0.0):
        return n - 1

    dist = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + (x2 * y1 - y2 * x1)) / denom

    start = max(1, min_genes - 1)
    end = n - 2
    if start > end:
        return min_genes - 1

    return int(np.argmax(dist[start:end + 1]) + start)


def get_elbow_genes(
    gene_to_lv_importance: pd.DataFrame,
    gamma: float,
    min_genes: int,
    include_elbow: bool = True,
    max_search: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    for lv in gene_to_lv_importance.index:
        lv_scores = pd.to_numeric(gene_to_lv_importance.loc[lv], errors="coerce").dropna()
        if lv_scores.empty:
            out[str(lv)] = pd.DataFrame(columns=["gene", "importance"])
            continue

        lv_sorted = lv_scores.sort_values(ascending=False)
        mu = lv_sorted.mean()

        if not np.isclose(mu, 0):
            lv_power = mu * (lv_sorted / mu) ** gamma
        else:
            lv_power = lv_sorted.copy()

        elbow_idx = _elbow_index_desc_chord(lv_power.values, min_genes=min_genes, max_search=max_search)
        cut = elbow_idx + 1 if include_elbow else elbow_idx
        cut = max(cut, min_genes)
        cut = min(cut, len(lv_sorted))

        out[str(lv)] = pd.DataFrame({
            "gene": lv_sorted.index[:cut].to_numpy(),
            "importance": lv_sorted.values[:cut],
            "importance_power": lv_power.values[:cut],
            "rank_1based": np.arange(1, cut + 1),
        })

    return out


def plot_lv_importance_curves(
    gene_to_lv_importance: pd.DataFrame,
    cfg: ExperimentConfig,
    out_dir: Path,
    gamma: Optional[float] = None,
    pad: float = 0.05,
    ncols: int = 3,
) -> None:
    ensure_dir(out_dir)

    if gamma is None:
        gamma = cfg.ig_gamma

    n_lv = gene_to_lv_importance.shape[0]
    nrows = math.ceil(n_lv / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.asarray(axes).flatten()

    for i in range(n_lv):
        lv_scores = pd.to_numeric(
            gene_to_lv_importance.iloc[i, :],
            errors="coerce",
        ).dropna()

        lv_sorted = lv_scores.sort_values(ascending=False)

        mu = lv_sorted.mean()
        if not np.isclose(mu, 0):
            lv_power = mu * (lv_sorted / mu) ** gamma
        else:
            lv_power = lv_sorted.copy()

        x = np.arange(1, len(lv_power) + 1)

        axes[i].plot(x, lv_power.values)
        axes[i].set_title(f"LV{i + 1}")
        axes[i].set_xlabel("Gene Importance rank")
        axes[i].set_ylabel("IG importance score")

        xmin, xmax = 1, len(lv_power)
        xr = max(1, xmax - xmin)
        axes[i].set_xlim(xmin - pad * xr, xmax + pad * xr)

    for j in range(n_lv, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    plt.savefig(out_dir / "all_LV_IG_importance_elbow_curves.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "all_LV_IG_importance_elbow_curves.jpg", dpi=300, bbox_inches="tight")
    plt.close()


def run_ig_stage(cfg: ExperimentConfig, model: nn.Module) -> None:
    print("\nComputing Integrated Gradients...")
    ensure_dir(cfg.ig_dir)

    _, test_set, df_rp = load_mouse_gene_data(cfg)
    feature_names = list(df_rp.columns[: cfg.gene_count])

    saliency_df = compute_latent_saliency_ig(model, test_set, feature_names, cfg)
    saliency_df.to_csv(cfg.ig_dir / "rp_to_LV_saliency_IG.csv")

    gene_to_lv_importance = compute_gene_to_lv_importance(cfg, saliency_df)
    gene_to_lv_importance.to_csv(cfg.ig_dir / "gene_to_LV_importance.csv")

    plot_lv_importance_curves(gene_to_lv_importance, cfg, cfg.ig_dir)

    elbow_genes = get_elbow_genes(gene_to_lv_importance, gamma=cfg.ig_gamma, min_genes=cfg.ig_min_genes)

    master_dfs = []

    for lv, df in elbow_genes.items():
        df.to_csv(cfg.ig_dir / f"elbow_genes_{lv}.csv", index=False)

        df_master = df.copy()
        df_master.insert(0, "latent_variable", lv)
        df_master.insert(1, "elbow_rank", range(1, len(df_master) + 1))
        master_dfs.append(df_master)

    if master_dfs:
        master_elbow_df = pd.concat(master_dfs, axis=0, ignore_index=True)
        master_elbow_df.to_csv(cfg.ig_dir / "master_elbow_genes_all_LVs.csv", index=False)

        common_n_genes = (master_elbow_df.groupby("latent_variable").size().min())
    
        master_elbow_equal_df = (master_elbow_df.groupby("latent_variable", group_keys=False).head(common_n_genes))
    
        master_elbow_equal_df.to_csv(cfg.ig_dir / "master_elbow_genes_all_LVs_equalnumber.csv", index=False)
    
        print(f"Common number of genes per LV: {common_n_genes}")

    print(f"Saved IG outputs to: {cfg.ig_dir}")


def run_pipeline(cfg: ExperimentConfig, stages: Sequence[str]) -> None:
    set_seed(cfg.seed)
    cfg.device = str(torch.device(cfg.device))
    ensure_dir(cfg.output_root)

    print(f"Using device: {cfg.device}")
    print(f"Output root: {cfg.output_root}")

    model = None
    if any(stage in stages for stage in ["trajectories", "ig", "all"]):
        model = load_model(cfg)

    if "all" in stages or "trajectories" in stages:
        run_trajectory_stage(cfg, model)

    if "all" in stages or "original_space" in stages:
        run_original_space_stage(cfg)

    if "all" in stages or "gene_lists" in stages:
        run_gene_list_stage(cfg)

    if ("all" in stages or "ig" in stages) and cfg.run_ig:
        if model is None:
            model = load_model(cfg)
        run_ig_stage(cfg, model)
    elif "ig" in stages and not cfg.run_ig:
        print("IG stage requested, but cfg.run_ig=False. Set --run-ig to enable IG computation.")

    print("\nPipeline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidated C-GMVAE downstream analysis pipeline.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "trajectories", "original_space", "gene_lists", "ig"],
        help="Pipeline stages to run.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--rp-gene-csv", type=Path, default=None)
    parser.add_argument("--cfm-csv", type=Path, default=None)
    parser.add_argument("--latent-csv", type=Path, default=None)
    parser.add_argument("--recons-csv", type=Path, default=None)
    parser.add_argument("--rp-pipeline-path", type=Path, default=None)
    parser.add_argument("--gene-list-path", type=Path, default=None)
    parser.add_argument("--gene-count", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--selected-density", type=float, default=None)
    parser.add_argument("--max-eff-corr-threshold", type=float, default=None)
    parser.add_argument("--run-ig", action="store_true")
    return parser.parse_args()


def apply_arg_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    mapping = {
        "output_root": args.output_root,
        "model_path": args.model_path,
        "rp_gene_csv": args.rp_gene_csv,
        "cfm_csv": args.cfm_csv,
        "latent_csv": args.latent_csv,
        "recons_csv": args.recons_csv,
        "rp_pipeline_path": args.rp_pipeline_path,
        "gene_list_path": args.gene_list_path,
        "gene_count": args.gene_count,
        "latent_dim": args.latent_dim,
        "seed": args.seed,
        "device": args.device,
        "selected_density": args.selected_density,
        "max_eff_corr_threshold": args.max_eff_corr_threshold,
    }

    for key, value in mapping.items():
        if value is not None:
            setattr(cfg, key, value)

    if args.run_ig:
        cfg.run_ig = True

    cfg.gene_lists_dirname = f"Gene_lists_{cfg.max_eff_corr_threshold}"

    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = apply_arg_overrides(ExperimentConfig(), args)
    run_pipeline(cfg, args.stages)

# How to run:
# Open directory which has the downstream_pipeline.py file, open terminal, then open conda environment
# conda activate u19-varnika
# python -m py_compile downstream_pipeline.py
# python downstream_pipeline.py --stages all