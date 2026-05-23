import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

BLUE   = "#0072B2"
ORANGE = "#D55E00"
GREEN  = "#009E73"
RED    = "#CC79A7"
GRAY   = "#999999"

BG      = "#ffffff"
AX_BG   = "#ffffff"
TEXT_C  = "#1a1a1a"
GRID_C  = "#e4e4e4"
SPINE_C = "#444444"

def load_csv(file_path: str) -> np.ndarray:
    return np.loadtxt(file_path, delimiter=',', skiprows=1)

def linear_func(x, a, b):       return a * x + b
def quadratic_func(x, a, b, c): return a * x**2 + b * x + c
def exponential_func(x, a, b, c, x0): return a * np.exp(b * (x - x0)) + c
def rational_1_1(x, p1, p2, q1):      return (p1 * x + p2) / (q1 * x + 1)
def rational_2_1(x, p1, p2, p3, q1):  return (p1 * x**2 + p2 * x + p3) / (q1 * x + 1)

def check_rational_stability(q1: float, x_data: np.ndarray) -> bool:
    denominators = q1 * x_data + 1
    return np.all(denominators > 0) or np.all(denominators < 0)

def fit_models(x_data, y_data, model_name):
    popt = None
    try:
        if model_name == "linear":
            popt, _ = curve_fit(linear_func, x_data, y_data, p0=[1.5, -2800])
            return linear_func, popt
        elif model_name == "quadratic":
            popt, _ = curve_fit(quadratic_func, x_data, y_data, p0=[0.01, -40, 40000])
            return quadratic_func, popt
        elif model_name == "exponential":
            x0_init = x_data.min()
            popt, _ = curve_fit(exponential_func, x_data, y_data, p0=[315, 0.01, 0, x0_init],
                                bounds=([0, 0, -np.inf, x_data.min()], [np.inf, np.inf, np.inf, x_data.max()]))
            return exponential_func, popt
        elif model_name == "rational_2_1":
            popt, _ = curve_fit(rational_2_1, x_data, y_data, p0=[0.001, 0.1, 300, 0.0001])
            if popt is not None and check_rational_stability(popt[3], x_data):
                return rational_2_1, popt
    except:
        pass
    return None, None

def find_year_reaches_limit(model_fn, params, start_year, limit=685, max_year=2300):
    years_fine = np.linspace(start_year, max_year, int((max_year - start_year) * 100) + 1)
    preds = model_fn(years_fine, *params)
    idx = np.where(preds >= limit)[0]
    if len(idx) > 0:
        return years_fine[idx[0]]
    return None

def main():
    data = load_csv('data/co2-ppm.csv')
    years_hist, co2_hist = data[:, 0], data[:, 1]

    f_lin, p_lin = fit_models(years_hist, co2_hist, "linear")

    mask2 = years_hist > 1990
    f_exp, p_exp = fit_models(years_hist[mask2], co2_hist[mask2], "exponential")
    f_rat, p_rat = fit_models(years_hist[mask2], co2_hist[mask2], "rational_2_1")

    year_685_lin = find_year_reaches_limit(f_lin, p_lin, 2023)
    year_685_exp = find_year_reaches_limit(f_exp, p_exp, 2023)
    year_685_rat = find_year_reaches_limit(f_rat, p_rat, 2023)

    # ── Visualization ────────────────────────────────────────────────────────
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

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    plot_years = np.linspace(1959, 2210, 1000)
    ax.plot(plot_years, f_lin(plot_years, *p_lin),
            color=BLUE,   linewidth=2.2, label='Global Linear Projection', zorder=3)
    ax.plot(plot_years, f_exp(plot_years, *p_exp),
            color=ORANGE, linewidth=2.2, label='Piecewise Baseline (Exponential)', zorder=3)
    ax.plot(plot_years, f_rat(plot_years, *p_rat),
            color=GREEN,  linewidth=2.2, label='Piecewise Rational (2/1)', zorder=3)

    ax.scatter(years_hist, co2_hist,
               color=GRAY, alpha=0.25, s=18, zorder=2, label='Historical Data')

    ax.axhline(685, color=RED, linestyle='--', linewidth=1.8,
               label='Threshold: 685 ppm', zorder=4)

    crossings = [
        (year_685_lin, BLUE,   "Global Linear"),
        (year_685_exp, ORANGE, "Piecewise Baseline"),
        (year_685_rat, GREEN,  "Piecewise Rational"),
    ]
    crossings.sort()
    offsets = [18, 46, 18]

    for i, (yr, color, name) in enumerate(crossings):
        if yr:
            ax.plot(yr, 685, 'o', color=color, markersize=11,
                    markeredgecolor=AX_BG, markeredgewidth=1.5, zorder=6)
            ax.annotate(f"{yr:.0f}", (yr, 685),
                        textcoords="offset points", xytext=(0, offsets[i]),
                        ha='center', fontsize=12, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.35', facecolor=AX_BG,
                                  edgecolor=color, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color=color, alpha=0.4))
            ax.axvline(yr, color=color, linestyle=':', alpha=0.2, zorder=1)

    ax.set_title(r'Timeline to 685 ppm $\mathrm{CO}_2$: Model Comparison',
                 fontsize=18, fontweight='bold', pad=14)
    ax.set_xlabel('Year',                              fontsize=13, labelpad=8)
    ax.set_ylabel(r'$\mathrm{CO}_2$ Concentration (ppm)', fontsize=13, labelpad=8)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1950, 2220)
    ax.set_ylim(300, 750)

    plt.tight_layout()
    output_path = os.path.join("co2_projections", "co2_threshold_685ppm.png")
    plt.savefig(output_path, facecolor=BG)
    plt.close()
    print(f"Threshold visualization saved to {output_path}")
    print(f"Years reaching 685 ppm:")
    print(f"  Linear:   {year_685_lin:.1f}")
    print(f"  Baseline: {year_685_exp:.1f}")
    print(f"  Rational: {year_685_rat:.1f}")

if __name__ == "__main__":
    main()
