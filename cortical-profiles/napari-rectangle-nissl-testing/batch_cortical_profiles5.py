"""
Batch Cortical Depth Profiler
==============================
Reads NDPI whole-slide images (lazily via dask/tifffile) and matching
napari Shapes CSV files, computes intensity profiles along the long axis
of each rectangle (averaging across the short axis), and saves per-
rectangle outputs as SVG figures and CSV files.

Expected file naming
--------------------
    images:      <name>.ndpi          e.g.  "2026-01-27 12.22.09.ndpi"
    rectangles:  <name> rec.csv       e.g.  "2026-01-27 12.22.09 rec.csv"
                 <name>_rec.csv       also accepted

Napari shapes CSV columns:
    index, shape-type, vertex-index, axis-0, axis-1

Usage
-----
    python batch_cortical_profiles.py /path/to/data_folder [OPTIONS]

Options (all are flags -- just add them to enable, no value needed)
-------
    --invert          Invert intensity so dark stain = high value.
                      ON by default.
    --no-invert       Disable intensity inversion.
    --normalize       Normalize all profiles to 0-1 range.
    --level LEVEL     Pyramid level to read (0 = full resolution).
                      Rectangle coordinates are automatically rescaled
                      to match the chosen level.  Default: 0.
    --rotate_tiff     Save region TIFFs resampled and rotated so that
                      the image is aligned with the profile plot:
                        top of image  = start of profile (position 0)
                        bottom        = end of profile
                      Without this flag, TIFFs show the axis-aligned
                      bounding box of the rectangle.

Examples
--------
    # Basic run at full resolution:
    python batch_cortical_profiles.py ./Nissl_Scans

    # Downsampled, with rotated TIFFs and normalized profiles:
    python batch_cortical_profiles.py ./Nissl_Scans --level 2 --rotate_tiff --normalize

    # Without inversion:
    python batch_cortical_profiles.py ./Nissl_Scans --no-invert

Outputs  (saved to <data_folder>/profiles/)
-------
    <name>_rect<N>.svg          Intensity profile figure
    <name>_rect<N>.csv          Profile data (distance_px, mean_intensity)
    <name>_rect<N>_region.tif   Grayscale region image (inverted, 8-bit)

    All rectangles from the same image share the same y-axis range in
    their SVG figures so profiles are directly comparable.

Requirements
------------
    pip install tifffile imagecodecs dask numpy scipy matplotlib
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import dask.array as da
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates


# ── Image I/O ─────────────────────────────────────────────────────────

def open_ndpi_dask(image_path: Path, level: int = 0) -> da.Array:
    """
    Open an NDPI file as a dask array at the requested pyramid level.

    Returns a 2D (grayscale) or 3D (Y, X, C) dask array.
    """
    store = tifffile.imread(str(image_path), aszarr=True, level=level)
    image = da.from_zarr(store)
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to float64 grayscale, handling common channel layouts."""
    if image.ndim == 2:
        return image.astype(np.float64)
    if image.ndim == 3:
        # (Y, X, C) with C <= 4  or  (C, Y, X) with C <= 4
        if image.shape[2] <= 4:
            return np.mean(image, axis=2).astype(np.float64)
        elif image.shape[0] <= 4:
            return np.mean(image, axis=0).astype(np.float64)
    raise ValueError(f"Cannot convert shape {image.shape} to grayscale")


def extract_region(dask_image: da.Array, corners: np.ndarray,
                   margin: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the bounding-box region around `corners` from a dask image,
    compute it into memory, and return (gray_region, local_corners).

    Parameters
    ----------
    dask_image : dask array
    corners : (4, 2) array of (row, col) coordinates
    margin : pixel margin around the bounding box

    Returns
    -------
    gray : 2D float64 numpy array (the cropped region, grayscale)
    local_corners : (4, 2) corners in local region coordinates
    """
    row_min = int(np.floor(corners[:, 0].min())) - margin
    row_max = int(np.ceil(corners[:, 0].max())) + margin
    col_min = int(np.floor(corners[:, 1].min())) - margin
    col_max = int(np.ceil(corners[:, 1].max())) + margin

    # Clamp to image bounds
    ndim = dask_image.ndim
    if ndim == 2:
        H, W = dask_image.shape
    elif ndim == 3:
        if dask_image.shape[2] <= 4:  # (Y, X, C)
            H, W = dask_image.shape[0], dask_image.shape[1]
        else:  # (C, Y, X)
            H, W = dask_image.shape[1], dask_image.shape[2]
    else:
        H, W = dask_image.shape[0], dask_image.shape[1]

    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(H, row_max)
    col_max = min(W, col_max)

    # Slice the dask array and compute into memory
    if ndim == 2:
        region = dask_image[row_min:row_max, col_min:col_max].compute()
    elif ndim == 3 and dask_image.shape[2] <= 4:
        region = dask_image[row_min:row_max, col_min:col_max, :].compute()
    elif ndim == 3:
        region = dask_image[:, row_min:row_max, col_min:col_max].compute()
    else:
        region = dask_image[row_min:row_max, col_min:col_max].compute()

    gray = to_grayscale(region)

    local_corners = corners.copy()
    local_corners[:, 0] -= row_min
    local_corners[:, 1] -= col_min

    return gray, local_corners


# ── Rectangle parsing ─────────────────────────────────────────────────

def parse_shapes_csv(csv_path: Path) -> list[np.ndarray]:
    """
    Parse a napari shapes CSV file.

    Expected columns: index, shape-type, vertex-index, axis-0, axis-1

    Returns a list of (4, 2) arrays, one per rectangle, with coordinates
    as (row, col) = (axis-0, axis-1).
    """
    rectangles: dict[int, list[tuple[float, float]]] = {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stype = row['shape-type'].strip()
            # Accept rectangles and 4-vertex polygons (useful when
            # rotated rectangles are drawn as polygons to preserve
            # their vertex positions on save).
            if stype not in ('rectangle', 'polygon'):
                continue
            idx = int(row['index'])
            r = float(row['axis-0'])
            c = float(row['axis-1'])
            rectangles.setdefault(idx, []).append((r, c))

    result = []
    for idx in sorted(rectangles.keys()):
        verts = rectangles[idx]
        if len(verts) != 4:
            print(f"  Warning: shape {idx} has {len(verts)} vertices, "
                  f"skipping (expected 4).")
            continue
        result.append(np.array(verts))

    return result


# ── Profile sampling ──────────────────────────────────────────────────

def get_rectangle_axes(corners: np.ndarray):
    """Return origin, long/short vectors and their lengths from 4 corners."""
    edge1 = corners[1] - corners[0]
    edge2 = corners[3] - corners[0]
    len1, len2 = np.linalg.norm(edge1), np.linalg.norm(edge2)

    if len1 >= len2:
        long_vec, short_vec = edge1, edge2
        long_len, short_len = len1, len2
    else:
        long_vec, short_vec = edge2, edge1
        long_len, short_len = len2, len1

    return corners[0], long_vec, short_vec, long_len, short_len


def sample_rectangle_profile(gray: np.ndarray, corners: np.ndarray,
                              n_long: int = None, n_short: int = None):
    """
    Sample intensity along the long axis of a rectangle, averaging
    across the short axis at each position.

    Parameters
    ----------
    gray : 2D float64 array (already cropped & grayscale)
    corners : (4, 2) in local coordinates

    Returns
    -------
    distances : 1D array of pixel distances along the long axis
    profile : 1D array of mean intensities
    """
    origin, long_vec, short_vec, long_len, short_len = get_rectangle_axes(corners)

    if n_long is None:
        n_long = max(int(np.round(long_len)), 2)
    if n_short is None:
        n_short = max(int(np.round(short_len)), 3)

    long_dir = long_vec / long_len
    short_dir = short_vec / short_len

    t_vals = np.linspace(0, long_len, n_long)
    s_vals = np.linspace(0, short_len, n_short)

    tt, ss = np.meshgrid(t_vals, s_vals, indexing='ij')
    coords = (origin[np.newaxis, np.newaxis, :]
              + tt[..., np.newaxis] * long_dir[np.newaxis, np.newaxis, :]
              + ss[..., np.newaxis] * short_dir[np.newaxis, np.newaxis, :])

    flat_coords = coords.reshape(-1, 2).T
    sampled = map_coordinates(gray, flat_coords, order=1, mode='nearest')
    sampled = sampled.reshape(n_long, n_short)

    return t_vals, np.mean(sampled, axis=1)


# ── File matching ─────────────────────────────────────────────────────

def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """
    Find matching (image.ndpi, shapes_rec.csv) pairs.

    Matching rule: for image "Foo.ndpi", look for "Foo rec.csv" or
    "Foo_rec.csv".
    """
    pairs = []
    for ndpi in sorted(data_dir.glob('*.ndpi')):
        stem = ndpi.stem  # e.g. "2026-01-27 12.22.09"
        candidates = [
            data_dir / f"{stem} rec.csv",
            data_dir / f"{stem}_rec.csv",
        ]
        for csv_path in candidates:
            if csv_path.exists():
                pairs.append((ndpi, csv_path))
                break
        else:
            print(f"  No rectangle CSV found for {ndpi.name}, skipping.")
    return pairs


# ── Plotting ──────────────────────────────────────────────────────────

def save_profile_figure(distances, profile, title, svg_path, y_range=None):
    """Save a clean SVG figure of the intensity profile.

    Parameters
    ----------
    y_range : tuple (ymin, ymax), optional
        If provided, fix the y-axis to these limits so that multiple
        figures from the same image share the same scale.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(distances, profile, color='#2E86AB', linewidth=1.5)
    ax.set_xlabel("Distance along cortical depth (px)")
    ax.set_ylabel("Mean intensity (a.u.)")
    ax.set_title(title)
    if y_range is not None:
        margin = (y_range[1] - y_range[0]) * 0.05
        ax.set_ylim(y_range[0] - margin, y_range[1] + margin)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(svg_path), format='svg')
    plt.close(fig)


def save_profile_csv(distances, profile, csv_path):
    """Save the profile as a two-column CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["distance_px", "mean_intensity"])
        for d, v in zip(distances, profile):
            writer.writerow([f"{d:.2f}", f"{v:.4f}"])


def save_region_tiff(gray: np.ndarray, tif_path: Path, invert: bool = True):
    """
    Save the extracted rectangle region as an inverted grayscale TIFF.

    The image is saved at the same resolution as the analysis (i.e. at
    the pyramid level that was used for processing).  This saves the
    axis-aligned bounding box — fast but not aligned to the rectangle.

    Parameters
    ----------
    gray : 2D float64 array — the cropped region around the rectangle
    tif_path : output path
    invert : if True, invert so dark stain becomes bright
    """
    img = gray.copy()

    if invert:
        img = img.max() - img + img.min()

    # Scale to 8-bit for a compact, viewable TIFF
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min()) * 255.0

    tifffile.imwrite(str(tif_path), img.astype(np.uint8))


def save_region_tiff_rotated(gray: np.ndarray, corners: np.ndarray,
                              tif_path: Path, invert: bool = True):
    """
    Resample the rectangle content into a TIFF that is aligned with
    the intensity profile plot.

    The long axis of the rectangle (= x-axis of the profile) becomes
    the horizontal axis of the TIFF (columns, left-to-right).  The
    short axis (averaging direction) becomes the vertical axis (rows).

    This means you can place the TIFF directly below/above the SVG
    profile and the positions will correspond.

    Parameters
    ----------
    gray : 2D float64 array — the bounding-box crop (local coords)
    corners : (4, 2) array of rectangle corners in local coordinates
    tif_path : output path
    invert : if True, invert so dark stain becomes bright
    """
    origin, long_vec, short_vec, long_len, short_len = get_rectangle_axes(corners)

    n_long = max(int(np.round(long_len)), 2)
    n_short = max(int(np.round(short_len)), 2)

    long_dir = long_vec / long_len
    short_dir = short_vec / short_len

    # Build sampling grid: rows = short axis, cols = long axis
    # so the TIFF horizontal axis matches the profile x-axis
    t_vals = np.linspace(0, long_len, n_long)    # along profile
    s_vals = np.linspace(0, short_len, n_short)   # averaging dir

    tt, ss = np.meshgrid(t_vals, s_vals, indexing='ij')  # (n_long, n_short)
    coords = (origin[np.newaxis, np.newaxis, :]
              + tt[..., np.newaxis] * long_dir[np.newaxis, np.newaxis, :]
              + ss[..., np.newaxis] * short_dir[np.newaxis, np.newaxis, :])

    flat_coords = coords.reshape(-1, 2).T
    sampled = map_coordinates(gray, flat_coords, order=1, mode='nearest')
    # sampled reshaped: rows=long axis, cols=short axis
    rotated = sampled.reshape(n_long, n_short)

    # Transpose so long axis is horizontal (columns) to match profile plot
    rotated = rotated.T  # now shape (n_short, n_long)

    if invert:
        rotated = rotated.max() - rotated + rotated.min()

    # Scale to 8-bit
    if rotated.max() > rotated.min():
        rotated = ((rotated - rotated.min())
                   / (rotated.max() - rotated.min()) * 255.0)

    tifffile.imwrite(str(tif_path), rotated.astype(np.uint8))


# ── Main ──────────────────────────────────────────────────────────────

def get_level_scale(image_path: Path, level: int) -> float:
    """
    Compute the scale factor between level 0 and the requested level.

    Returns the factor by which level-0 coordinates must be divided to
    obtain coordinates in the requested level's pixel space.
    """
    if level == 0:
        return 1.0

    with tifffile.TiffFile(str(image_path)) as tif:
        # Get dimensions of the first two levels we need
        series = tif.series[0]
        shape0 = series.levels[0].shape  # full resolution
        shape_l = series.levels[level].shape

        # Use the largest spatial dimension to compute scale
        # Shapes are typically (Y, X) or (Y, X, C)
        scale = shape0[0] / shape_l[0]

    return scale


def process(data_dir: Path, invert: bool = True, normalize: bool = False,
            level: int = 0, rotate_tiff: bool = False):
    """Process all image–rectangle pairs in data_dir."""
    out_dir = data_dir / "profiles"
    out_dir.mkdir(exist_ok=True)

    pairs = find_pairs(data_dir)
    if not pairs:
        print(f"No matching .ndpi + rec.csv pairs found in {data_dir}")
        return

    print(f"Found {len(pairs)} image–rectangle pair(s).\n")

    for ndpi_path, csv_path in pairs:
        image_name = ndpi_path.stem
        print(f"── {image_name} ──")

        # Parse rectangles
        rectangles = parse_shapes_csv(csv_path)
        if not rectangles:
            print("  No valid rectangles found, skipping.\n")
            continue
        print(f"  {len(rectangles)} rectangle(s) loaded from CSV.")

        # Compute coordinate scale factor for this pyramid level
        scale = get_level_scale(ndpi_path, level)
        if level > 0:
            print(f"  Pyramid level {level}: scale factor = {scale:.2f}x")
            rectangles = [corners / scale for corners in rectangles]

        # Open image lazily
        print(f"  Opening NDPI (level {level}) …")
        dask_image = open_ndpi_dask(ndpi_path, level=level)
        print(f"  Image shape: {dask_image.shape}, "
              f"dtype: {dask_image.dtype}")

        # ── Pass 1: compute all profiles for this image ──
        results = []  # list of (rect_idx, distances_fullres, profile)

        for rect_idx, corners in enumerate(rectangles):
            print(f"  Processing rectangle {rect_idx} …", end=" ")

            # Extract only the needed region into memory
            gray, local_corners = extract_region(dask_image, corners)

            # Compute profile
            distances, profile = sample_rectangle_profile(gray, local_corners)

            # Convert distances back to full-resolution pixel units
            distances_fullres = distances * scale

            if invert:
                profile = profile.max() - profile + profile.min()

            results.append((rect_idx, distances_fullres, profile, gray,
                            local_corners))
            print(f"done  ({len(distances)} samples, "
                  f"{distances_fullres[-1]:.0f} full-res px)")

        # ── Determine shared y-axis range across all rectangles ──
        global_min = min(prof.min() for _, _, prof, _, _ in results)
        global_max = max(prof.max() for _, _, prof, _, _ in results)
        y_range = (global_min, global_max)
        print(f"  Shared y-range: [{global_min:.1f}, {global_max:.1f}]")

        # ── Pass 2: optionally normalize, save profile + region TIFF ──
        for rect_idx, distances_fullres, profile, gray, local_corners in results:
            label = f"{image_name}_rect{rect_idx}"

            if normalize and (profile.max() - profile.min()) > 0:
                profile = ((profile - profile.min())
                           / (profile.max() - profile.min()))
                save_y_range = (0.0, 1.0)
            else:
                save_y_range = y_range

            svg_path = out_dir / f"{label}.svg"
            csv_out = out_dir / f"{label}.csv"
            tif_path = out_dir / f"{label}_region.tif"

            save_profile_figure(
                distances_fullres, profile,
                title=f"{image_name}  —  rect {rect_idx}",
                svg_path=svg_path,
                y_range=save_y_range,
            )
            save_profile_csv(distances_fullres, profile, csv_out)
            if rotate_tiff:
                save_region_tiff_rotated(gray, local_corners, tif_path,
                                         invert=invert)
            else:
                save_region_tiff(gray, tif_path, invert=invert)

        print()

    print(f"All outputs saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch cortical depth profiler for NDPI + napari shapes."
    )
    parser.add_argument(
        "data_dir", type=Path,
        help="Folder containing .ndpi images and matching rec.csv files",
    )
    parser.add_argument(
        "--invert", action="store_true", default=True,
        help="Invert intensity so dark stain = high (default: on)",
    )
    parser.add_argument(
        "--no-invert", action="store_false", dest="invert",
        help="Disable intensity inversion",
    )
    parser.add_argument(
        "--normalize", action="store_true", default=False,
        help="Normalize profiles to 0–1 range",
    )
    parser.add_argument(
        "--level", type=int, default=0,
        help="Pyramid level to read (0 = full resolution, default: 0)",
    )
    parser.add_argument(
        "--rotate_tiff", action="store_true", default=False,
        help="Save region TIFFs rotated and cropped to the rectangle, "
             "aligned with the profile (top=start, bottom=end). "
             "Default: save the axis-aligned bounding box.",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: {args.data_dir} is not a directory.")
        sys.exit(1)

    process(args.data_dir, invert=args.invert, normalize=args.normalize,
            level=args.level, rotate_tiff=args.rotate_tiff)


if __name__ == '__main__':
    main()
