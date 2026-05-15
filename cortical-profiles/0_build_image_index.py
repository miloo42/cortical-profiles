"""
build_image_index.py
Loops through all TIF files in the Nissl (analysis_Nissl) and
Autofluorescence (Autofluorescence_analysis) folders, reads each image
with tifffile, parses the filename, and saves a metadata CSV.
"""

import re
import pathlib
import tifffile
import pandas as pd

ROOT = pathlib.Path(__file__).parent
NISSL_DIR = ROOT / "analysis_Nissl"
AF_DIR    = ROOT / "Autofluorescence_analysis"
OUT_CSV   = ROOT / "data" / "image_index.csv"


# ---------------------------------------------------------------------------
# Nissl filename parser
# ---------------------------------------------------------------------------
# Typical:  B07_AntCing_S3_cropped_16bit_inverted.tif
# Atypical: C2-B07_S1_S4_4x_4ms_nissl_50um_fused_...tif   (channel prefix)
#           Bo7_S1_S3_cropped_16bit_inverted.tif            (typo in brain id)
NISSL_RE = re.compile(
    r"""
    (?:[A-Za-z0-9]+-)?          # optional leading prefix  (e.g. "C2-")
    (?P<brain>[Bb][oO]?\d+)     # brain id: B07, B13, Bo7 …
    _
    (?P<region>[A-Za-z0-9]+)    # cortical region
    _
    (?P<section>S\d+)           # section: S1, S2 …
    (?:_(?P<suffix>.+?))?       # optional trailing processing tags
    \.tif$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_nissl(path: pathlib.Path) -> dict:
    stem = path.name
    m = NISSL_RE.match(stem)
    if m:
        brain = m.group("brain").upper().replace("BO", "B0")  # normalise typo
        return {
            "brain_id":     brain,
            "region":       m.group("region"),
            "section":      m.group("section").upper(),
            "suffix_tags":  m.group("suffix") or "",
            "parse_status": "ok",
        }
    return {"parse_status": "unmatched"}


# ---------------------------------------------------------------------------
# Autofluorescence filename parser
# ---------------------------------------------------------------------------
# AF_LF_200um_B072024A2_S1of3_R1__4x_1s_yellow_14126_stats.tif
# AF_LF_200um_B072024antcing_S2of3_4x_1000ms_yellow_14126_stats.tif
# AF_LF_200um_B132024A1_R1of3_4x_1000ms_yellow_15126_stats.tif
AF_RE = re.compile(
    r"""
    AF_LF_
    (?P<thickness>\d+um)_       # section thickness
    (?P<brain>B\d+)             # brain id
    (?P<year>\d{4})             # acquisition year
    (?P<region>[A-Za-z0-9]+)_  # cortical region (may be lowercase)
    (?P<section>[SR]\d+)of      # section / replicate index  (S or R)
    (?P<total>\d+)              # total sections/replicates
    (?:_R\d+_*)?                # optional replicate tag (e.g. _R1__)
    _?
    (?P<magnification>\d+x)_    # objective magnification
    (?P<exposure>\d+m?s)_       # exposure time (ms or s)
    (?P<channel>[A-Za-z]+)_     # fluorescence channel
    (?P<filter>\d+)_            # filter / wavelength code
    stats\.tif$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_af(path: pathlib.Path) -> dict:
    stem = path.name
    m = AF_RE.match(stem)
    if m:
        return {
            "brain_id":      m.group("brain").upper(),
            "year":          m.group("year"),
            "region":        m.group("region"),
            "section":       m.group("section").upper(),
            "total_sections":m.group("total"),
            "thickness":     m.group("thickness"),
            "magnification": m.group("magnification"),
            "exposure":      m.group("exposure"),
            "channel":       m.group("channel").lower(),
            "filter_code":   m.group("filter"),
            "parse_status":  "ok",
        }
    return {"parse_status": "unmatched"}


# ---------------------------------------------------------------------------
# Image reader
# ---------------------------------------------------------------------------

def read_image_meta(path: pathlib.Path) -> dict:
    """Return shape / dtype info from a tifffile without loading pixel data."""
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0] if tif.series else None
        if series is not None:
            shape = series.shape
            dtype = str(series.dtype)
            axes  = series.axes
        else:
            page = tif.pages[0]
            shape = (len(tif.pages), page.shape[0], page.shape[1])
            dtype = str(page.dtype)
            axes  = "ZYX"
        n_pages = len(tif.pages)
    return {
        "shape":   str(shape),
        "dtype":   dtype,
        "axes":    axes,
        "n_pages": n_pages,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def collect(folder: pathlib.Path, modality: str, parser) -> list[dict]:
    rows = []
    for path in sorted(folder.rglob("*.tif")):
        row: dict = {
            "modality":  modality,
            "file_path": str(path),
            "filename":  path.name,
            "subfolder": path.parent.name,
        }
        # Parse filename
        row.update(parser(path))

        # Read image metadata
        try:
            row.update(read_image_meta(path))
        except Exception as exc:
            row["read_error"] = str(exc)

        rows.append(row)
    return rows


def main():
    records = []
    records.extend(collect(NISSL_DIR, "nissl",            parse_nissl))
    records.extend(collect(AF_DIR,    "autofluorescence", parse_af))

    df = pd.DataFrame(records)

    # Reorder: put key columns first
    front_cols = [
        "modality", "brain_id", "region", "section",
        "file_path", "filename", "subfolder",
        "shape", "dtype", "axes", "n_pages",
        "parse_status",
    ]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows → {OUT_CSV}")

    unmatched = df[df["parse_status"] == "unmatched"]
    if not unmatched.empty:
        print(f"\nWarning: {len(unmatched)} filenames did not match the parser:")
        for fn in unmatched["filename"]:
            print(f"  {fn}")


if __name__ == "__main__":
    main()
