# C-GMVAE Downstream Analysis Pipeline

## Overview

This pipeline performs downstream analysis on a trained Conditional Gaussian Mixture Variational Autoencoder (C-GMVAE) to identify genes and biological pathways associated with cognitive resilience in AD-BXD mice.
The workflow combines two complementary approaches:

1. **Density-guided latent trajectory analysis**, which identifies genes associated with transitions between susceptible and resilient latent states.

2. **Integrated Gradients (IG)**, which identifies genes that most strongly contribute to the latent variables learned by the model.

Together, these approaches provide both a trajectory-based and latent-feature-based interpretation of the learned representation.

## Stage 1: Density-Guided Interpolation (DGI)

### Goal

The goal of DGI is to generate biologically plausible trajectories through latent space connecting cognitively susceptible and resilient cellular states.
Rather than assuming a direct linear transition between two latent representations, DGI constrains trajectories to move through regions of latent space that are occupied by real cells.
This helps avoid trajectories passing through unrealistic latent states that may not correspond to biologically meaningful cell populations.

### Method

For each trajectory:
1. Cells are ranked according to decoded cognitive resilience (CFM) scores.
2. Low-resilience and high-resilience endpoints are selected.
3. A Gaussian kernel density estimator (KDE) is fitted to the latent space.
4. At each interpolation step, movement is determined by a combination of:
   - the direction toward the target endpoint
   - the local gradient of latent density

The update direction is
\[
d=(1-\lambda)u+\lambda\nabla\log p(z)
\]

where:
- \(u\) is the normalized direction toward the endpoint
- \(p(z)\) is the latent-space density estimated by KDE
- \(\lambda\) controls the strength of density guidance
Low values of \(\lambda\) produce trajectories similar to linear interpolation, whereas larger values encourage trajectories to remain within high-density regions.

### Output

Each interpolated latent point is:
1. Assigned a resilience label using k-nearest neighbours.
2. Decoded back into gene-feature space.
3. Assigned a predicted cognitive resilience (CFM) value.
The resulting trajectories represent hypothetical transitions between susceptible and resilient cellular states.

## Stage 2: Reconstruction to Original Gene Space

### Goal

The decoder reconstructs random-projection gene features rather than individual genes. To perform biological interpretation, gene-list curation and pathway enrichment, trajectory outputs must therefore be mapped back into original gene space.

### Method

The original preprocessing pipeline compressed gene expression using random projection:
\[
X' = XP
\]
where:

- \(X\) is the original gene-expression matrix
- \(P\) is the random projection matrix
- \(X'\) is the reduced feature representation

To approximately recover gene-level information, the Moore-Penrose pseudoinverse of the projection matrix is used:
\[
\hat X = X'(P^T)^+
\]
This produces reconstructed gene trajectories that can be analyzed directly.

## Stage 3: Trajectory-Derived CFM-Associated Gene List Curation

### Goal

The purpose of this stage is to identify genes that are:
1. Strongly represented within each latent gene feature.
2. Strongly associated with cognitive resilience (CFM) along trajectories.
3. Consistently observed across multiple trajectories.

Only genes satisfying all three criteria are retained. This tells us which genes change along cognitive resilience trajectories.

### Step 1: Identify Highly Weighted Genes

Each reconstructed RP feature is associated with thousands of original genes through the random projection matrix. For every RP feature, genes are ranked according to the absolute magnitude of their projection weights. Only the strongest contributors are retained:

```python
percentile_cutoff = 99.9
```
This corresponds to the top 0.1% of genes for each RP feature. The rationale is that most genes contribute only weakly to a given RP feature, whereas a small subset dominates the feature's behavior.

### Step 2: Quantify Phenotype Association

For each RP feature, the Pearson correlation with decoded cognitive resilience (CFM) values along the trajectory is computed:

\[
r_i = corr(RP_i, CFM)
\]

This measures whether the feature increases or decreases as resilience changes along the latent trajectory.

### Step 3: Compute Effective Correlation

A gene may contribute positively or negatively to an RP feature depending on the sign of its projection weight. To propagate phenotype association from RP features back to genes, an effective correlation score is computed:

\[
r_{eff}
=
corr(RP_i,CFM)
\times sign(w_{ij})
\]

where:
- \(w_{ij}\) is the projection weight connecting gene \(j\) to RP feature \(i\)
- \(corr(RP_i,CFM)\) measures resilience association

Interpretation:
- Positive effective correlation:
  - gene expression increases with resilience
- Negative effective correlation:
  - gene expression decreases with resilience

### Step 4: Retain Strongly Associated Genes

For each gene, the largest absolute effective correlation observed across all RP features is retained. Genes are then separated into:

#### C1 genes:

\[
r_{eff} \ge 0.65
\]

Genes positively associated with resilience.

#### C2 genes:

\[
r_{eff} \le -0.65
\]

Genes negatively associated with resilience.

The threshold removes weak phenotype associations and focuses the analysis on genes showing strong resilience-related behavior.

### Step 5: Gene Annotation Filtering

Several additional filters are applied to improve interpretability, curation of gene candidates, and downstream pathway enrichment.

Depending on configuration, the following may be removed:

- duplicate entries
- unannotated genes
- non-protein-coding genes
- predicted genes (Gm/LOC)
- RIKEN cDNA genes

For pathway enrichment analysis purposes, all of the above-mentioned categories are removed as pathway databases such as Gene Ontology, Metascape, and g:Profiler primarily annotate well-characterized genes. Removing poorly annotated entries improves enrichment specificity and interpretability. For the purposes of nominating novel gene candidates, one may want to retain certain categories (such as retaining non-protein-coding genes) as necessary.

### Step 6: Consensus Gene Selection

The above procedure is performed independently for every trajectory. Genes are then intersected across all forward and reverse trajectories. Only genes consistently observed across trajectories are retained.

This consensus approach reduces trajectory-specific noise and enriches for genes that repeatedly appear during transitions between susceptible and resilient states. The resulting consensus gene sets form the final inputs for pathway enrichment analysis.

## Stage 4: Integrated Gradients

### Goal

While trajectory analysis identifies genes associated with transitions through latent space, Integrated Gradients identifies genes that contribute most strongly to individual latent variables.
This provides a complementary view of model interpretability, providing information about which genes are responsible for the latent representation itself.

## Pathway Enrichment Analysis:

### Goal

The final curated gene sets are intended for downstream biological interpretation using pathway enrichment analysis. Consensus gene lists generated from Stage 3 and latent-variable-specific gene lists generated from Stage 4 can be analyzed independently or jointly. This can be used to identify biological pathways enriched in the input gene list(s).

### Recommended Tools

#### g:Profiler:

https://biit.cs.ut.ee/gprofiler

g:Profiler performs over-representation analysis across multiple biological databases.

Recommended settings:
- Organism: Mus musculus
- Statistical domain scope: annotated genes only
- Multiple testing correction: g:SCS (default)
- User threshold: 0.05

#### Metascape:

https://metascape.org

Metascape provides complementary pathway enrichment and network-based visualization.

Recommended settings:
- Species: Mus musculus
- Analysis type: Express Analysis or Batch Analysis
- Significance threshold: adjusted p-value < 0.05

## Running the Pipeline on the Terminal:

Edit the ExperimentConfig() class in downstream_pipeline.py as necessary.

Activate the environment:

```bash
conda activate <env_name>
```

Run the full pipeline:

```bash
python downstream_pipeline.py
```

Run individual stages:

```bash
python downstream_pipeline.py --stages trajectories
python downstream_pipeline.py --stages original_space
python downstream_pipeline.py --stages gene_lists
python downstream_pipeline.py --stages ig
```
