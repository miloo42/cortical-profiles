# Cortical Profiles

Analysis of cortical cytoarchitecture from microscopy images using laminar depth profiles.  
Two modalities: **Nissl** histology and **autofluorescence** imaging across multiple cortical regions and subjects.

## Projects

| Folder | Description |
|--------|-------------|
| [`cortical-profiles/`](cortical-profiles/README.md) | Profile extraction, visualisation, and statistical comparison (Mahalanobis distances, PERMANOVA, Hotelling T²) |
| [`registration-af-brightfield/`](registration-af-brightfield/README.md) | Coarse-to-fine mutual-information registration of AF and brightfield images |

## Data

Image files (`.tif`, `.ndpi`) are excluded from git — see `.gitignore`.  
Processed data tables (`data/*.csv`) are tracked.

## Dependencies

Python 3.10+, conda/pip environment.  
See each project's README for specific package requirements.
