"""
Signal Comparison Metrics for Fluorescence Microscopy Line Profiles.

Computes 7 complementary similarity metrics on red/green channel intensity
profiles and produces 3 summary figures.
"""

import csv
import numpy as np
from scipy import stats, signal, spatial
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend for saving PNGs

# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------

def parse_european_float(s: str) -> float:
    """Convert European-format string like '29,000' to 29.0."""
    return float(s.replace(",", "."))


def parse_csv(path: str) -> list[dict]:
    """
    Parse Values.csv into a list of dicts, one per image.

    Each dict has keys 'x', 'y1' (red), 'y2' (green) as numpy arrays.
    The CSV has 3 image blocks at column offsets 0, 5, 10 separated by
    empty columns, with European decimal format (commas in quoted fields).
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Skip header rows (row 0 = image names, row 1 = column labels)
    data_rows = rows[2:]

    # Column offsets for each image block: (x_col, y1_col, y2_col)
    offsets = [(0, 1, 2), (5, 6, 7), (10, 11, 12)]
    images = []

    for x_col, y1_col, y2_col in offsets:
        xs, y1s, y2s = [], [], []
        for row in data_rows:
            # Check that columns exist and are non-empty
            if len(row) <= y2_col:
                break
            if not row[x_col].strip() or not row[y1_col].strip():
                break
            xs.append(parse_european_float(row[x_col]))
            y1s.append(parse_european_float(row[y1_col]))
            y2s.append(parse_european_float(row[y2_col]))
        images.append({
            "x": np.array(xs),
            "y1": np.array(y1s),  # red channel
            "y2": np.array(y2s),  # green channel
        })

    return images


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pearson(y1: np.ndarray, y2: np.ndarray) -> tuple[float, float]:
    """Return (Pearson r, p-value)."""
    r, p = stats.pearsonr(y1, y2)
    return r, p


def compute_cosine(y1: np.ndarray, y2: np.ndarray) -> float:
    """Return cosine similarity (1 - cosine distance)."""
    return 1.0 - spatial.distance.cosine(y1, y2)


def compute_xcorr(y1: np.ndarray, y2: np.ndarray) -> dict:
    """
    Normalized cross-correlation.

    Returns dict with 'lags', 'corr' (full correlation array),
    'peak_corr', and 'peak_lag'.
    """
    # Normalize to zero-mean, unit-variance
    y1n = (y1 - y1.mean()) / (y1.std() + 1e-12)
    y2n = (y2 - y2.mean()) / (y2.std() + 1e-12)

    corr = signal.correlate(y1n, y2n, mode="full") / len(y1)
    lags = signal.correlation_lags(len(y1), len(y2), mode="full")

    peak_idx = np.argmax(corr)
    return {
        "lags": lags,
        "corr": corr,
        "peak_corr": corr[peak_idx],
        "peak_lag": lags[peak_idx],
    }


def compute_manders(y1: np.ndarray, y2: np.ndarray) -> tuple[float, float]:
    """
    Manders' M1 and M2 coefficients with mean-intensity threshold.

    M1 = fraction of y1 signal that overlaps with y2 (above threshold).
    M2 = fraction of y2 signal that overlaps with y1 (above threshold).
    """
    thresh1 = y1.mean()
    thresh2 = y2.mean()

    # Pixels where both channels are above their respective thresholds
    coloc_mask = (y1 > thresh1) & (y2 > thresh2)

    m1 = y1[coloc_mask].sum() / (y1.sum() + 1e-12)
    m2 = y2[coloc_mask].sum() / (y2.sum() + 1e-12)
    return m1, m2


def compute_icq(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Li's Intensity Correlation Quotient.

    ICQ = (fraction of pixels where (y1-mean1)*(y2-mean2) > 0) - 0.5
    Range: -0.5 (segregated) to +0.5 (dependent staining).
    """
    diff1 = y1 - y1.mean()
    diff2 = y2 - y2.mean()
    product = diff1 * diff2
    fraction_positive = np.sum(product > 0) / len(product)
    return fraction_positive - 0.5


def compute_dtw(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Dynamic Time Warping distance, normalized by (n + m).

    Simple O(n*m) DP — fast enough for ~300-point profiles.
    """
    n, m = len(y1), len(y2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(y1[i - 1] - y2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return dtw_matrix[n, m] / (n + m)


def compute_emd(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Earth Mover's Distance treating profiles as weighted spatial distributions.

    Each signal is normalized to sum to 1 (distribution over pixel positions),
    then scipy's wasserstein_distance computes the work to transform one into
    the other, in units of pixels.
    """
    positions = np.arange(len(y1), dtype=float)
    w1 = y1 / (y1.sum() + 1e-12)
    w2 = y2 / (y2.sum() + 1e-12)
    return stats.wasserstein_distance(positions, positions, w1, w2)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_signal_overlays(images: list[dict], path: str) -> None:
    """Plot red + green channel profiles for each image."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    fig.suptitle("Signal Profiles — Red (Y1) vs Green (Y2) Channels", fontsize=13)

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.plot(img["x"], img["y1"], color="tab:red", alpha=0.85, linewidth=1.2,
                label="Y1 (red)")
        ax.plot(img["x"], img["y2"], color="tab:green", alpha=0.85, linewidth=1.2,
                label="Y2 (green)")
        ax.set_title(f"Image {i + 1}  (n={len(img['x'])})")
        ax.set_xlabel("Position (pixels)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cross_correlations(xcorr_results: list[dict], path: str) -> None:
    """Plot cross-correlation vs lag for each image, with peak annotated."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Normalized Cross-Correlation", fontsize=13)

    for i, (xc, ax) in enumerate(zip(xcorr_results, axes)):
        ax.plot(xc["lags"], xc["corr"], color="steelblue", linewidth=1)
        ax.axvline(xc["peak_lag"], color="crimson", linestyle="--", alpha=0.7)
        ax.annotate(
            f"peak = {xc['peak_corr']:.3f}\nlag = {xc['peak_lag']}",
            xy=(xc["peak_lag"], xc["peak_corr"]),
            xytext=(0.60, 0.85), textcoords="axes fraction",
            fontsize=8, color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
        )
        ax.set_title(f"Image {i + 1}")
        ax.set_xlabel("Lag (pixels)")
        ax.set_ylabel("Correlation")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_summary_table(table_df: pd.DataFrame, path: str) -> None:
    """Render the metrics summary as a visual table saved to PNG."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    ax.set_title("Signal Comparison Metrics — Summary", fontsize=13, pad=12)

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        rowLabels=table_df.index,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    # Style header row
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:  # row labels
            cell.set_facecolor("#D9E2F3")
            cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv_path = "Values.csv"

    print("=" * 60)
    print("Signal Comparison — Fluorescence Microscopy Line Profiles")
    print("=" * 60)

    # --- Parse data ---
    images = parse_csv(csv_path)
    for i, img in enumerate(images):
        print(f"  Image {i+1}: {len(img['x'])} data points")

    # --- Compute all metrics ---
    all_results = []
    xcorr_results = []

    for i, img in enumerate(images):
        y1, y2 = img["y1"], img["y2"]

        r, p = compute_pearson(y1, y2)
        cos_sim = compute_cosine(y1, y2)
        xc = compute_xcorr(y1, y2)
        xcorr_results.append(xc)
        m1, m2 = compute_manders(y1, y2)
        icq = compute_icq(y1, y2)
        dtw_dist = compute_dtw(y1, y2)
        emd_dist = compute_emd(y1, y2)
        rmse = np.sqrt(np.mean((y1 - y2) ** 2))

        all_results.append({
            "Image": f"Image {i+1}",
            "Pearson r": f"{r:.4f}",
            "Pearson p": f"{p:.2e}",
            "Cosine Sim": f"{cos_sim:.4f}",
            "XCorr Peak": f"{xc['peak_corr']:.4f}",
            "XCorr Lag": f"{xc['peak_lag']}",
            "Manders M1": f"{m1:.4f}",
            "Manders M2": f"{m2:.4f}",
            "ICQ": f"{icq:.4f}",
            "DTW (norm)": f"{dtw_dist:.2f}",
            "EMD (px)": f"{emd_dist:.2f}",
            "RMSE": f"{rmse:.2f}",
        })

    # --- Build summary table ---
    df = pd.DataFrame(all_results).set_index("Image")
    print("\n" + df.to_string())

    # --- Generate figures ---
    print("\nGenerating figures...")
    plot_signal_overlays(images, "signal_overlays.png")
    plot_cross_correlations(xcorr_results, "cross_correlations.png")
    plot_summary_table(df, "summary_table.png")

    # --- Interpretation guide ---
    print("\n" + "=" * 60)
    print("Interpretation Guide")
    print("=" * 60)
    print("""
  Pearson r    | Shape correlation (-1 to +1). High = signals co-vary.
  Cosine Sim   | Vector similarity (0 to 1). Close to Pearson for centered data.
  XCorr Peak   | Best correlation at optimal lag. Lag ≈ 0 means no spatial shift.
  Manders M1   | Fraction of red signal overlapping green (above mean threshold).
  Manders M2   | Fraction of green signal overlapping red (above mean threshold).
  ICQ          | Li's Intensity Correlation Quotient (-0.5 to +0.5).
               |   > 0 = dependent (co-localized), < 0 = segregated.
  DTW (norm)   | Shape distance allowing warping (0 = identical). Lower is better.
  EMD (px)     | Spatial distribution distance in pixels. Lower is more similar.
  RMSE         | Root Mean Square Error (amplitude-sensitive baseline).
""")


if __name__ == "__main__":
    main()
