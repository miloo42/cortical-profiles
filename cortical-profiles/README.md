# Cortical Depth Profile Analysis Pipeline

Quantitative comparison of cortical cytoarchitecture from brightfield histology using laminar intensity profiles, inspired by the observer-independent methods of Schleicher et al. (1999, 2005) developed at Forschungszentrum Jülich.

## Overview

This pipeline extracts 1D intensity profiles across cortical depth from brightfield microscopy images (Nissl-stained and autofluorescence), normalizes them for thickness and staining variation, computes shape-based feature vectors, and statistically tests whether cortical regions are cytoarchitectonically distinguishable.

## Pipeline structure

The pipeline consists of three sequential stages, each implemented as a Jupyter notebook:

```
0_build_image_index.py       →  Scan image folders, write data/image_index.csv
1_reading_profiles.ipynb     →  Extract & preprocess profiles from images → data/profiles_long.csv
2_plotting_profiles.ipynb    →  Visualise profiles grouped by region and modality
3_distances_nissl.ipynb      →  Statistical comparison (Nissl)
3_distances_af.ipynb         →  Statistical comparison (autofluorescence)

data/
  image_index.csv            ←  metadata for all images (output of step 0)
  profiles_long.csv          ←  long-form profile table (output of step 1)
  profiles_features.csv      ←  Schleicher feature vectors (input for steps 3+)

analysis_Nissl/              ←  Nissl TIF images, gitignored
Autofluorescence_analysis/   ←  AF TIF images, gitignored
```

### Notebook 1 — Profile extraction (`1_reading_profiles.ipynb`)

**Input:** An `image_index.csv` file listing TIFF image paths with metadata columns (`brain_id`, `region`, `section`, `modality`) and the corresponding TIFF image files. Each image is a thin brightfield strip oriented perpendicular to the cortical surface, spanning from the pial surface to the white matter boundary.

**Processing steps:**

1. **Read images** from paths listed in the index CSV using `tifffile`.
2. **Assign pixel sizes** per modality. Autofluorescence images have a resolution of 0.576 px/µm, while Nissl images use 0.62 px/µm, with two exceptions (B07_A2_S1 and B07_S1_S4 which also use 0.576 px/µm).
3. **Compute mean intensity profile** by averaging each image along the horizontal axis (rows), yielding a 1D vector representing intensity as a function of cortical depth.
4. **Detrend the profile** to remove slowly varying staining gradients. A wide moving average (window = 1/3 of profile length) estimates the background, and the profile is divided by this baseline to correct for multiplicative intensity variation. The result is mean-normalized so profiles are comparable across sections.
5. **Resample** the detrended profile to a fixed number of depth bins (500) using quadratic interpolation, normalizing for differences in cortical thickness.
6. **Smooth** the resampled profile with a Gaussian filter (σ = 4 bins) to reduce high-frequency noise while preserving laminar features.
7. **Export** profiles in long format (`profiles_long.csv`) for downstream plotting and analysis.

**Key functions:**

- `detrend_profile(profile, baseline_window)` — Estimates a slowly varying baseline via `uniform_filter1d`, divides the raw profile by this baseline, and mean-normalizes the result. This corrects for multiplicative staining gradients (e.g., differential dye penetration across cortical depth).
- `resample_profile(profile, n_bins)` — Resamples profiles of varying pixel length onto a common 0–100% cortical depth axis using quadratic interpolation. This is the percent-depth normalization step, appropriate for profiles extracted from flat cortical regions where equivolumetric correction is unnecessary.
- `compute_normalized_profile(img)` — Combines the above steps into a single function applied to each image.

**Output:** A long-format CSV (`data/profiles_long.csv`) with columns for `brain_id`, `region`, `section`, `modality`, `bin` (depth position), and `profile` (intensity value).

### Notebook 2 — Visualisation (`2_plotting_profiles.ipynb`)

**Input:** `data/profiles_long.csv`

**Processing:**

- Creates a faceted line plot using seaborn's `FacetGrid`, with rows for cortical regions and columns for each brain × modality combination.
- Individual sections are overlaid as separate lines within each facet to show within-region variability.

**Output:** `profiles_output.png` — a publication-ready overview of all profiles.

### Notebook 3 — Statistical comparison (`3_distances_nissl.ipynb` / `3_distances_af.ipynb`)

These two notebooks are identical in structure, differing only in which modality is selected (Nissl or autofluorescence). Each region is identified by the combination of `brain_id` and `region`.

**Input:** `data/profiles_features.csv` — a CSV file containing precomputed 10-element Schleicher feature vectors per profile (generated in a separate feature extraction step).

The feature vector for each profile consists of 10 shape descriptors, following Schleicher et al. (1999):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Mean intensity | Overall staining density |
| 1 | Centre of gravity | Intensity-weighted mean depth position |
| 2 | Standard deviation | Spread of intensity across depth |
| 3 | Skewness | Asymmetry of the laminar pattern |
| 4 | Kurtosis | Peakedness of the intensity distribution |
| 5–9 | Same five features | Computed on the first derivative of the profile |

The derivative-based features capture the sharpness and position of laminar transitions (layer boundaries) rather than absolute intensity levels.

**Processing steps:**

1. **Parse feature vectors** from CSV string representation back into numeric arrays.

2. **Compute pairwise Mahalanobis distances** between all profiles. The Mahalanobis distance accounts for correlations between features and differences in variance, providing a more meaningful dissimilarity metric than Euclidean distance. The covariance matrix is estimated from the full dataset and regularized (+ 1e-6 on the diagonal) to ensure invertibility.

3. **Visualise the distance matrix** as a heatmap.

4. **PERMANOVA (global test)** — Tests whether within-region distances are significantly smaller than between-region distances using `skbio.stats.distance.permanova` with 9999 permutations. This is a non-parametric test that makes no distributional assumptions.

5. **MDS embedding** — Projects the distance matrix into 2D using multidimensional scaling for visual assessment of region separability.

6. **Pairwise post-hoc tests (Hotelling's T²)** — For each pair of regions, tests whether their mean feature vectors differ using `pingouin.multivariate_ttest`. Because the full 10-feature vector exceeds the sample size in most groups (causing NaN p-values), PCA is first applied to reduce to 3 components before running the pairwise tests. P-values are Bonferroni-corrected for multiple comparisons.

7. **Network visualisation** — Significant pairwise differences are plotted as a network graph using `networkx`, where edges connect regions that are significantly different from each other.

8. **Region distinctiveness ranking** — For each region, computes the mean Mahalanobis distance to all other profiles, producing a ranking from most to least cytoarchitectonically distinct.

## Methods background

### Percent-depth normalization

Cortical thickness varies between regions and individuals. To compare laminar profiles, each profile is resampled to a common depth axis running from 0% (pial surface) to 100% (white matter boundary). This linear normalization is appropriate for profiles extracted from flat cortical regions. For highly curved cortex (gyral crowns and sulcal fundi), equivolumetric normalization (Waehnert et al., 2014; Wagstyl et al., 2018) would better preserve the correspondence between fractional depth and anatomical layers, following Bok's principle that layers adjust their thickness to maintain constant volume.

### Baseline detrending

Brightfield histology commonly shows slowly varying intensity gradients across cortical depth due to differential dye penetration, uneven section thickness, or illumination non-uniformity. Division by a wide moving-average baseline corrects for this multiplicative artifact while preserving the higher-frequency laminar modulations that carry cytoarchitectonic information.

### Schleicher feature vector

Rather than comparing raw profiles point-by-point, each profile is summarized by a 10-element feature vector based on central moments (Schleicher et al., 1999). This provides a compact, noise-robust representation of laminar pattern shape. The original method was developed at Forschungszentrum Jülich using grey level index (GLI) images — binarized maps of cell body area fraction computed via adaptive thresholding — but the same feature extraction applies to continuous intensity profiles.

### Mahalanobis distance

Pairwise dissimilarity between profiles is quantified using the Mahalanobis distance, which accounts for correlations between features and normalizes by variance. This was shown by Schleicher et al. (2000, 2005) to be more sensitive than Euclidean distance for detecting cytoarchitectonic boundaries. The distance matrix serves as input to both visualization (MDS) and statistical testing (PERMANOVA).

### Statistical testing

- **PERMANOVA** tests the global null hypothesis that profiles from different regions are drawn from the same distribution, by comparing within-group to between-group distances in the Mahalanobis distance matrix. It is assumption-free (no requirement for multivariate normality or equal covariances).
- **Hotelling's T²** tests pairwise differences between region means in feature space. Because the 10-dimensional feature vector exceeds the per-group sample size, PCA dimensionality reduction to 3 components is applied first. An alternative pairwise approach avoiding this issue is pairwise PERMANOVA on the distance matrix with FDR correction.

## Dependencies

```
numpy
scipy
pandas
matplotlib
seaborn
tifffile
scikit-learn
scikit-bio
pingouin
networkx
```

Install with:
```bash
pip install numpy scipy pandas matplotlib seaborn tifffile scikit-learn scikit-bio pingouin networkx
```

## Input data requirements

- **Image files:** Single-channel TIFF images of brightfield cortical strips, oriented with the pial surface at the top (row 0) and white matter at the bottom. Each image should cover the full cortical depth from pia to white matter.
- **Image index:** A CSV file (`image_index.csv`) with columns:
  - `file_path` — absolute path to the TIFF file (written by `0_build_image_index.py`)
  - `brain_id` — subject identifier
  - `region` — cortical region label
  - `section` — section identifier (replicate within region)
  - `modality` — `nissl` or `autofluorescence`
- **Feature file:** A CSV file (`profiles_features.csv`) with columns including `brain_id`, `region`, `modality`, and `Features` (string representation of 10-element feature vectors).

## References

- Schleicher, A., Amunts, K., Geyer, S., Morosan, P. & Zilles, K. (1999). Observer-independent method for microstructural parcellation of cerebral cortex: A quantitative approach to cytoarchitectonics. *NeuroImage*, 9, 165–177.
- Schleicher, A., Palomero-Gallagher, N., Morosan, P., Eickhoff, S.B., Kowalski, T., de Vos, K., Amunts, K. & Zilles, K. (2005). Quantitative architectural analysis: a new approach to cortical mapping. *Anatomy and Embryology*, 210, 373–386.
- Schleicher, A. & Zilles, K. (1990). A quantitative approach to cytoarchitectonics: analysis of structural inhomogeneities in nervous tissue using an image analyser. *Journal of Microscopy*, 157, 367–381.
- Wree, A., Schleicher, A. & Zilles, K. (1982). Estimation of volume fractions in nervous tissue with an image analyzer. *Journal of Neuroscience Methods*, 6, 29–43.
- Waehnert, M.D., Dinse, J., Weiss, M., Streicher, M.N., Waehnert, P., Geyer, S., Turner, R. & Bazin, P.L. (2014). Anatomically motivated modeling of cortical laminae. *NeuroImage*, 93, 210–220.
- Wagstyl, K., Lepage, C., Bludau, S., Zilles, K., Fletcher, P.C., Amunts, K. & Evans, A.C. (2020). BigBrain 3D atlas of cortical layers: cortical and laminar thickness gradients diverge in sensory and motor cortices. *PLoS Biology*, 18(4), e3000678.
- Amunts, K., Mohlberg, H., Bludau, S. & Zilles, K. (2020). Julich-Brain: A 3D probabilistic atlas of the human brain's cytoarchitecture. *Science*, 369, 988–992.
