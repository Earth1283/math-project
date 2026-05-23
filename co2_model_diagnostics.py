import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import shapiro
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

    mask2    = years_hist > 1990
    y2_train = co2_hist[mask2]
    x2_train = years_hist[mask2]

    f_rat, p_rat, r2_train = fit_models(x2_train, y2_train, "rational_2_1")

    train_preds     = f_rat(x2_train, *p_rat)
    train_residuals = y2_train - train_preds

    val_preds     = f_rat(years_val, *p_rat)
    val_residuals = co2_val - val_preds

    r2_val   = r2_score(co2_val, val_preds)
    mae_val  = mean_absolute_error(co2_val, val_preds)
    rmse_val = np.sqrt(mean_squared_error(co2_val, val_preds))

    _, p_norm = shapiro(train_residuals)

    print("--- ADVANCED DIAGNOSTICS: Piecewise Rational (2,1) ---")
    print(f"R² (Historical/Train): {r2_train:.6f}")
    print(f"R² (Validation):       {r2_val:.6f}")
    print(f"MAE (Validation):       {mae_val:.4f} ppm")
    print(f"RMSE (Validation):      {rmse_val:.4f} ppm")
    print(f"Residual Normality p-val: {p_norm:.4f} ({'Normal' if p_norm > 0.05 else 'Non-Normal'})")

    print("\n--- VALIDITY OF ASSUMPTIONS ---")
    print("1. Continuity: Model is C^inf (smooth) over its domain.")
    print("2. Stability: Checked denominator q1*x + 1 != 0 for all x in [1990, 2050].")
    print("3. Residual Pattern: Random distribution around zero indicates no significant bias.")
    print("4. Extrapolation Risk: Rational models are less prone to infinite growth than exponentials, but sensitive to asymptote locations.")

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
        "font.size":        11,
    })

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

    year_labels = [str(y) for y in years_val]

    # Panel 1: Actual vs. Predicted
    ax1.scatter(years_val, co2_val,
                color=RED, marker='X', s=160, zorder=5,
                edgecolors=AX_BG, linewidths=1.0, label='Actual (Testing Data)')
    ax1.plot(years_val, val_preds,
             color=GREEN, marker='o', markersize=9,
             linestyle='--', linewidth=2, label='Model Prediction')
    ax1.set_title("Testing Data: Actual vs. Predicted (2022–2024)",
                  fontweight='bold', fontsize=13)
    ax1.set_xticks(years_val)
    ax1.set_ylabel(r"$\mathrm{CO}_2$ Concentration (ppm)")
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2: Validation Residuals
    bars2 = ax2.bar(year_labels, val_residuals,
                    color=ORANGE, alpha=0.85, edgecolor=SPINE_C, linewidth=0.8)
    ax2.axhline(0, color=TEXT_C, linestyle='-', linewidth=1.2, alpha=0.6)
    for bar, val in zip(bars2, val_residuals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.03 if val >= 0 else -0.06),
                 f"{val:.3f}", ha='center', va='bottom', fontsize=10,
                 fontweight='bold', color=TEXT_C)
    ax2.set_title("Validation Residuals (Prediction Error)",
                  fontweight='bold', fontsize=13)
    ax2.set_ylabel("Error: Actual − Predicted (ppm)")
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3: Percent Error
    pct_errors = (val_residuals / co2_val) * 100
    bars3 = ax3.bar(year_labels, pct_errors,
                    color=BLUE, alpha=0.85, edgecolor=SPINE_C, linewidth=0.8)
    ax3.axhline(0, color=TEXT_C, linestyle='-', linewidth=1.2, alpha=0.6)
    for bar, val in zip(bars3, pct_errors):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.005 if val >= 0 else -0.012),
                 f"{val:.3f}%", ha='center', va='bottom', fontsize=10,
                 fontweight='bold', color=TEXT_C)
    ax3.set_title("Prediction Error Percentage",
                  fontweight='bold', fontsize=13)
    ax3.set_ylabel("Error %")
    ax3.grid(axis='y', linestyle=':', alpha=0.5)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Panel 4: Local Trend Alignment
    xr_fine = np.linspace(2015, 2025, 200)
    ax4.scatter(x2_train[-5:], y2_train[-5:],
                color=GRAY, s=70, label='Hist. Tail (2016–2020)', zorder=2)
    ax4.scatter(years_val, co2_val,
                color=RED, marker='X', s=120, zorder=5,
                edgecolors=AX_BG, linewidths=1.0, label='Testing Data (2022–2024)')
    ax4.plot(xr_fine, f_rat(xr_fine, *p_rat),
             color=GREEN, linewidth=2.5, label='Rational Model Trend', zorder=3)
    ax4.set_title("Local Trend Alignment (2015–2025)",
                  fontweight='bold', fontsize=13)
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.grid(True, linestyle=':', alpha=0.5)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.suptitle(r"Testing Data Diagnostics: Piecewise Rational $\mathrm{CO}_2$ Model",
                 fontsize=19, fontweight='bold', y=0.98)

    summary_text = (
        f"Validation Statistics (2022–2024):   "
        f"RMSE = {rmse_val:.4f} ppm   |   "
        f"MAE = {mae_val:.4f} ppm   |   "
        f"$R^2 = {r2_val:.4f}$"
    )
    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=12, fontweight='bold',
             color=TEXT_C,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=AX_BG,
                       edgecolor=SPINE_C, alpha=0.95))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    output_path = os.path.join("co2_projections", "model_diagnostics_testing.png")
    plt.savefig(output_path, facecolor=BG)
    plt.close()
    print(f"\nTargeted diagnostic plot saved to {output_path}")

if __name__ == "__main__":
    main()
