# registration-af-brightfield

Coarse-to-fine mutual-information registration of autofluorescence (AF) images acquired before and after aldehyde fixation (brightfield channel) for sample `occlobe13`, sections s1–s3.

## Project structure

```
registration-af-brightfield/
├── register_BF2fluo_mutual-information.ipynb   # registration notebook
└── raw_images/                                 # input TIF images (not tracked by git)
    ├── occlobe13_s*_AFbefore_yellow_*.tif      # AF before fixation (C1)
    ├── occlobe13_s*_*AFafter_yellow_*.tif      # AF after fixation (C2)
    └── occlobe13_s*_aldehydef_BF_*.tif         # brightfield after fixation (C3)
```

## Workflow

The notebook `register_BF2fluo_mutual-information.ipynb`:

1. Loads C1 (AF before), C2 (AF after), C3 (BF aldehyde) from `raw_images/`.
2. Registers C2 → C1 and C3 → C1 using a coarse-to-fine brute-force search maximising mutual information.
3. Writes registered images as `*_registered.tif` in the project folder.

## Algorithm

- Multi-resolution pyramid (3 levels, 4× downsampling per level).
- Brute-force grid search over rotation angle and x/y shift at each resolution.
- Metric: mutual information from joint histogram (64 bins).
- Final transform applied with cubic interpolation (`order=3`).

## Dependencies

```
tifffile, numpy, scipy, scikit-learn, scikit-image
```

Install: `pip install tifffile numpy scipy scikit-learn scikit-image`
