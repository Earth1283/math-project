import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from co2_prediction import load_csv, linear_func, exponential_func, rational_2_1, fit_models

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

def main():
    data = load_csv('data/co2-ppm.csv')
    years_hist, co2_hist = data[:, 0], data[:, 1]

    years_val = np.array([2022, 2023, 2024])
    co2_val   = np.array([418.53, 421.08, 424.61])

    _, p_lin, _ = fit_models(years_hist, co2_hist, "linear")

    mask2 = years_hist > 1990
    _, p_exp, _ = fit_models(years_hist[mask2], co2_hist[mask2], "exponential")
    _, p_rat, _ = fit_models(years_hist[mask2], co2_hist[mask2], "rational_2_1")

    pred_lin = linear_func(years_val, *p_lin)
    pred_exp = exponential_func(years_val, *p_exp)
    pred_rat = rational_2_1(years_val, *p_rat)

    def get_metrics(true, pred):
        mae  = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        return mae, rmse

    m_lin = get_metrics(co2_val, pred_lin)
    m_exp = get_metrics(co2_val, pred_exp)
    m_rat = get_metrics(co2_val, pred_rat)

    print("--- Model Validation Report (2022-2024) ---")
    print(f"Global Linear:      MAE={m_lin[0]:.4f}, RMSE={m_lin[1]:.4f}")
    print(f"Piecewise Baseline: MAE={m_exp[0]:.4f}, RMSE={m_exp[1]:.4f}")
    print(f"Piecewise Rational: MAE={m_rat[0]:.4f}, RMSE={m_rat[1]:.4f}")

    better_model = "Rational (Piecewise)" if m_rat[1] < m_exp[1] else "Baseline (Piecewise)"
    print(f"\nWinning Model: {better_model}")

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

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    years_plot = np.linspace(2010, 2025, 200)

    ax.scatter(years_hist[-10:], co2_hist[-10:],
               color=GRAY, alpha=0.6, s=50, zorder=2, label='Historical (2011–2020)')
    ax.scatter(years_val, co2_val,
               color=TEXT_C, marker='X', s=120, zorder=5,
               edgecolors=AX_BG, linewidths=0.8, label='Observed (2022–2024)')

    ax.plot(years_plot, linear_func(years_plot, *p_lin),
            color=BLUE,   linestyle='--', linewidth=2,
            label=f'Global Linear  (RMSE = {m_lin[1]:.2f} ppm)')
    ax.plot(years_plot, exponential_func(years_plot, *p_exp),
            color=ORANGE, linewidth=2.5,
            label=f'Post-1990 Exponential  (RMSE = {m_exp[1]:.2f} ppm)')
    ax.plot(years_plot, rational_2_1(years_plot, *p_rat),
            color=GREEN,  linewidth=2.5,
            label=f'Post-1990 Rational  (RMSE = {m_rat[1]:.2f} ppm)')

    for y, v in zip(years_val, co2_val):
        ax.annotate(f"{v:.2f} ppm", (y, v),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=11, fontweight='bold', color=TEXT_C,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=AX_BG,
                              edgecolor=SPINE_C, alpha=0.9))

    ax.set_title(r'Model Validation: 2022–2024 Projections vs. Observed Data',
                 fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Year',                              fontsize=13, labelpad=8)
    ax.set_ylabel(r'$\mathrm{CO}_2$ Concentration (ppm)', fontsize=13, labelpad=8)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(2010.5, 2024.8)
    ax.set_ylim(390, 432)

    plt.tight_layout()
    output_path = os.path.join("co2_projections", "model_validation_2024.png")
    plt.savefig(output_path, facecolor=BG)
    plt.close()
    print(f"\nValidation plot saved to {output_path}")

if __name__ == "__main__":
    main()
