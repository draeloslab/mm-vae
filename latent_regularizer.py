import torch
import numpy as np
from itertools import combinations
from sobol_seq import i4_sobol_generate_std_normal

def draw_gmm_samples(num_samples, gmm_centers, gmm_std):
    num_gmm_centers, dimension = gmm_centers.shape

    samples = []
    components = []
    for _ in range(num_samples):
        component = np.random.choice(range(num_gmm_centers))

        component_mean = gmm_centers[component, :]
        component_cov = torch.eye(dimension) * gmm_std
        component_cov = component_cov.to(gmm_centers.device)

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=component_mean, covariance_matrix=component_cov)

        sample = distribution.sample((1,))
        samples.append(sample)
        components.append(component)
    samples = torch.vstack(samples)

    return samples, components

def compute_gmm_covariance(gmm_centers, gmm_std):
    num_gmm_centers, dimension = gmm_centers.shape
    component_cov = torch.eye(dimension) * gmm_std
    component_cov = component_cov.to(gmm_centers.device)

    weighted_gmm_centers = gmm_centers.mean(axis=0)
    gmm_centers = gmm_centers - weighted_gmm_centers

    conditional_expectation = 0
    cconditional_expectation = torch.zeros(
        dimension,
        dimension,
        device=gmm_centers.device,
        dtype=gmm_centers.dtype,
    )
    for component in range(num_gmm_centers):
        center_mean = gmm_centers[component, :].reshape(dimension, 1)
        conditional_expectation += (1 / num_gmm_centers) * torch.mm(center_mean, center_mean.t())

    return component_cov, component_cov + conditional_expectation


def compute_empirical_covariance(embedding_matrix):
    m = torch.mean(embedding_matrix, dim=0)
    sigma = (
            torch.mm((embedding_matrix - m).t(), (embedding_matrix - m))
            / embedding_matrix.shape[0])
    return sigma

def mean_squared_kolmogorov_smirnov_distance_gmm_broadcasting(embedding_matrix, gmm_centers, gmm_std):
    sorted_embeddings = torch.sort(embedding_matrix, dim=-2).values
    emb_num, emb_dim = sorted_embeddings.shape[-2:]
    num_gmm_centers, _ = gmm_centers.shape
    empirical_cdf = torch.linspace(start=1 / emb_num,
        end=1.0,
        steps=emb_num,
        device=embedding_matrix.device,
        dtype=embedding_matrix.dtype).unsqueeze(-1)

    normalized_embedding_distances_to_centers = (sorted_embeddings[:, None] - gmm_centers[None]) / gmm_std
    normal_cdf_per_center = 0.5 * (1 + torch.erf(normalized_embedding_distances_to_centers * 0.70710678118))
    normal_cdf = normal_cdf_per_center.mean(dim=1)

    return torch.nn.functional.mse_loss(normal_cdf, empirical_cdf)

def mean_squared_covariance_gmm(embedding_matrix, gmm_centers, gmm_std):
    sigma = compute_empirical_covariance(embedding_matrix)
    comp_covariance, gmm_covariance = compute_gmm_covariance(gmm_centers, gmm_std)
    comp_covariance = comp_covariance.to(embedding_matrix.device)
    gmm_covariance = gmm_covariance.to(embedding_matrix.device)
    diff = torch.pow(sigma - gmm_covariance, 2)
    mean_cov = torch.mean(diff)
    return mean_cov
