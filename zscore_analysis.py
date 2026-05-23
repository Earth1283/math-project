import numpy as np
import matplotlib.pyplot as plt

BLUE   = "#0072B2"
ORANGE = "#D55E00"

BG      = "#ffffff"
AX_BG   = "#ffffff"
TEXT_C  = "#1a1a1a"
GRID_C  = "#e4e4e4"
SPINE_C = "#444444"

def load_csv(file_path: str) -> np.ndarray:
    try:
        return np.loadtxt(file_path, delimiter=',', skiprows=1)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except ValueError as e:
        print(f"Error: Could not parse numeric data from {file_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        raise

def load_and_align_data(co2_path: str, temp_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    co2_data  = load_csv(co2_path)
    temp_data = load_csv(temp_path)

    co2_data  = co2_data[co2_data[:, 0]   >= 1959]
    temp_data = temp_data[temp_data[:, 0] >= 1959]

    common_years  = np.intersect1d(co2_data[:, 0], temp_data[:, 0])
    co2_filtered  = co2_data[np.isin(co2_data[:, 0],   common_years)]
    temp_filtered = temp_data[np.isin(temp_data[:, 0], common_years)]

    co2_filtered  = co2_filtered[co2_filtered[:, 0].argsort()]
    temp_filtered = temp_filtered[temp_filtered[:, 0].argsort()]

    return common_years, co2_filtered[:, 1], temp_filtered[:, 1]

def calculate_zscore(data: np.ndarray) -> np.ndarray:
    return (data - np.mean(data)) / np.std(data)

def plot_zscores(years: np.ndarray, co2_z: np.ndarray, temp_z: np.ndarray) -> None:
    correlation = np.corrcoef(co2_z, temp_z)[0, 1]

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   AX_BG,
        "axes.edgecolor":   SPINE_C,
        "axes.labelcolor":  TEXT_C,
        "axes.titlecolor":  TEXT_C,
        "text.color":       TEXT_C,
        "xtick.color":      TEXT_C,
        "ytick.color":      TEXT_C,
        "grid.color":       GRID_C,
        "grid.linewidth":   0.8,
        "legend.facecolor": AX_BG,
        "legend.edgecolor": SPINE_C,
        "font.family":      "sans-serif",
        "font.size":        12,
    })

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    ax.plot(years, co2_z,  label=r'$\mathrm{CO}_2$ Concentration (Z-score)',
            color=BLUE,   linewidth=2.5, zorder=3)
    ax.plot(years, temp_z, label='Temperature Change (Z-score)',
            color=ORANGE, linewidth=2.5, alpha=0.9, zorder=3)

    ax.fill_between(years, co2_z, temp_z,
                    where=(co2_z > temp_z), alpha=0.10, color=BLUE)
    ax.fill_between(years, co2_z, temp_z,
                    where=(co2_z < temp_z), alpha=0.10, color=ORANGE)

    ax.axhline(0, color=SPINE_C, linestyle='--', linewidth=1.2, alpha=0.8)

    ax.text(0.02, 0.96,
            f"Pearson Correlation\n$R = {correlation:.4f}$",
            transform=ax.transAxes, fontsize=13, verticalalignment='top',
            color=TEXT_C,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=AX_BG,
                      edgecolor=SPINE_C, alpha=0.95))

    ax.set_title(r'Standardized Trend Comparison: $\mathrm{CO}_2$ vs. Temperature',
                 fontsize=18, fontweight='bold', pad=14)
    ax.set_xlabel('Year',                           fontsize=14, labelpad=8)
    ax.set_ylabel('Standard Deviations (Z-score)',  fontsize=14, labelpad=8)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig('zscore_comparison.png', facecolor=BG)
    plt.close()
    print("Plot saved to zscore_comparison.png")

if __name__ == "__main__":
    years, co2, temp = load_and_align_data(
        'data/co2-ppm.csv', 'data/surface-air-temp-change.csv'
    )
    co2_z  = calculate_zscore(co2)
    temp_z = calculate_zscore(temp)

    print(f"Standardization Complete.")
    print(f"CO2  Z-score: mean={np.mean(co2_z):.4f}, std={np.std(co2_z):.4f}")
    print(f"Temp Z-score: mean={np.mean(temp_z):.4f}, std={np.std(temp_z):.4f}")

    plot_zscores(years, co2_z, temp_z)
