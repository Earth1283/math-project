import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from co2_prediction import load_csv

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

def hybrid_model(x, a, b, c, d, e):
    t = x - 1959
    return a * t**2 + b * t + c + d * np.exp(e * t)

def main():
    data = load_csv('data/co2-ppm.csv')
    years_hist, co2_hist = data[:, 0], data[:, 1]

    years_val = np.array([2022, 2023, 2024])
    co2_val   = np.array([418.53, 421.08, 424.61])

    x_train = np.concatenate([years_hist, years_val])
    y_train = np.concatenate([co2_hist,   co2_val])

    p0 = [0.005, 0.8, 310, 5.0, 0.02]
    try:
        popt, _ = curve_fit(hybrid_model, x_train, y_train, p0=p0, maxfev=50000)
        print("Integrated Global Model Fit Successful (1959-2024).")
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    y_train_pred = hybrid_model(x_train, *popt)
    r2_total     = r2_score(y_train, y_train_pred)
    mae_total    = mean_absolute_error(y_train, y_train_pred)
    rmse_total   = np.sqrt(mean_squared_error(y_train, y_train_pred))

    print("\n--- INTEGRATED GLOBAL HYBRID PERFORMANCE ---")
    print(f"Model Scope: Entire Dataset (1959-2024)")
    print(f"Overall R²:   {r2_total:.6f}")
    print(f"Overall MAE:  {mae_total:.4f} ppm")
    print(f"Overall RMSE: {rmse_total:.4f} ppm")

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), dpi=300,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # ── Top: Model Fit ────────────────────────────────────────────────────
    years_plot = np.linspace(1959, 2024.5, 400)
    y_plot     = hybrid_model(years_plot, *popt)

    ax1.scatter(years_hist, co2_hist,
                color=BLUE, alpha=0.45, s=36,
                edgecolors='none', label='Historical Data (1959–2020)', zorder=2)
    ax1.scatter(years_val, co2_val,
                color=RED, marker='X', s=160, zorder=5,
                edgecolors=AX_BG, linewidths=1.0, label='Recent Data (2022–2024)')
    ax1.plot(years_plot, y_plot,
             color=GREEN, linewidth=3, label='Hybrid Ensemble Fit', zorder=3)

    stats_text = (
        f"$R^2 = {r2_total:.6f}$\n"
        f"MAE  = {mae_total:.4f} ppm\n"
        f"RMSE = {rmse_total:.4f} ppm"
    )
    ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes,
             fontsize=12, verticalalignment='top', color=TEXT_C,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=AX_BG,
                       edgecolor=SPINE_C, alpha=0.95))

    eq_label = (
        f"$y(t) = ({popt[0]:.5f}t^2 + {popt[1]:.4f}t + {popt[2]:.1f})"
        f" + {popt[3]:.3f}e^{{{popt[4]:.4f}t}}$"
    )
    ax1.plot([], [], ' ', label=eq_label)

    ax1.set_title(r'Global $\mathrm{CO}_2$ Fit: Integrated Hybrid Ensemble Model',
                  fontsize=18, fontweight='bold', pad=12)
    ax1.set_xlabel('Year',                              fontsize=13, labelpad=6)
    ax1.set_ylabel(r'$\mathrm{CO}_2$ Concentration (ppm)', fontsize=13, labelpad=6)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(1955, 2026)
    ax1.set_ylim(310, 440)

    # ── Bottom: Residuals ─────────────────────────────────────────────────
    residuals = y_train - y_train_pred

    ax2.scatter(y_train_pred, residuals,
                color=GREEN, alpha=0.55, s=40,
                edgecolors='none', label='Residuals', zorder=3)
    ax2.axhline(0, color=TEXT_C, linestyle='--', linewidth=1.4, alpha=0.6)

    ax2.set_title('Residual Plot: Hybrid Model Accuracy (1959–2024)',
                  fontsize=15, fontweight='bold', pad=10)
    ax2.set_xlabel(r'Predicted $\mathrm{CO}_2$ Concentration (ppm)',
                   fontsize=13, labelpad=6)
    ax2.set_ylabel('Residuals (ppm)', fontsize=13, labelpad=6)
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=2.5)
    output_path = os.path.join("co2_projections", "hybrid_ensemble_fit.png")
    plt.savefig(output_path, facecolor=BG)
    plt.close()
    print(f"\nEnsemble analysis plot saved to {output_path}")

if __name__ == "__main__":
    main()
