import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from co2_prediction import load_csv

# Style Constants
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
RED = "#CC79A7"
GRAY = "#999999"

# f(x) = Quadratic + Exponential
# Centering x around the start of the dataset (1959) to improve fitting stability
def hybrid_model(x, a, b, c, d, e):
    t = x - 1959
    quad_part = a * t**2 + b * t + c
    exp_part = d * np.exp(e * t)
    return quad_part + exp_part

def main():
    # 1. Load Data
    data = load_csv('data/co2-ppm.csv')
    years_hist, co2_hist = data[:, 0], data[:, 1]
    
    # Validation/Testing Data (Include in the new global fit)
    years_val = np.array([2022, 2023, 2024])
    co2_val = np.array([418.53, 421.08, 424.61])
    
    # 2. Fit Hybrid Model to ALL Data (1959-2024)
    # Combine historical and recent data for the ultimate fit
    x_train = np.concatenate([years_hist, years_val])
    y_train = np.concatenate([co2_hist, co2_val])
    
    # Initial guesses optimized for the 1959 start point
    p0 = [0.005, 0.8, 310, 5.0, 0.02]
    
    try:
        popt, _ = curve_fit(hybrid_model, x_train, y_train, p0=p0, maxfev=50000)
        print("Integrated Global Model Fit Successful (1959-2024).")
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    # 3. Predictions and Evaluation
    y_train_pred = hybrid_model(x_train, *popt)
    r2_total = r2_score(y_train, y_train_pred)
    mae_total = mean_absolute_error(y_train, y_train_pred)
    rmse_total = np.sqrt(mean_squared_error(y_train, y_train_pred))

    print("\n--- INTEGRATED GLOBAL HYBRID PERFORMANCE ---")
    print(f"Model Scope: Entire Dataset (1959-2024)")
    print(f"Overall R²:   {r2_total:.6f}")
    print(f"Overall MAE:  {mae_total:.4f} ppm")
    print(f"Overall RMSE: {rmse_total:.4f} ppm")

    # 4. Visualization (Two-Panel Style matching regression_analysis.png)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18), dpi=300)
    
    # --- Top Subplot: Model Fit ---
    years_plot = np.linspace(1959, 2024.5, 400)
    y_plot = hybrid_model(years_plot, *popt)
    
    # Scatter data
    ax1.scatter(years_hist, co2_hist, color=BLUE, alpha=0.5, s=40, label='Historical Data (1959-2020)')
    ax1.scatter(years_val, co2_val, color=RED, marker='X', s=150, label='Recent Data (2022-2024)', zorder=5)
    
    # Plot fit
    ax1.plot(years_plot, y_plot, color=GREEN, linewidth=3.5, label='Hybrid Ensemble Fit')
    
    # Statistics Box
    stats_text = (f"Model Statistics:\n"
                  f"$R^2 = {r2_total:.6f}$\n"
                  f"MAE = {mae_total:.4f} ppm\n"
                  f"RMSE = {rmse_total:.4f} ppm")
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
             fontsize=12, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Equation in Legend
    eq_label = (f"Fit Equation:\n"
                f"$y(t) = ({popt[0]:.5f}t^2 + {popt[1]:.4f}t + {popt[2]:.1f}) + {popt[3]:.3f}e^{{{popt[4]:.4f}t}}$")
    ax1.plot([], [], ' ', label=eq_label) # Dummy for legend entry

    ax1.set_title('Global CO$_2$ Fit: Integrated Hybrid Ensemble Model', fontsize=20, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('CO$_2$ Concentration (ppm)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(1955, 2026)
    ax1.set_ylim(310, 435)

    # --- Bottom Subplot: Residuals ---
    residuals = y_train - y_train_pred
    ax2.scatter(y_train_pred, residuals, color=GREEN, alpha=0.6, s=50, edgecolor='white', label='Residuals')
    ax2.axhline(0, color='black', linestyle='--', linewidth=2)
    
    ax2.set_title('Residual Plot: Hybrid Model Accuracy (1959-2024)', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Predicted CO$_2$ Concentration (ppm)', fontsize=14)
    ax2.set_ylabel('Residuals (ppm)', fontsize=14)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    output_path = os.path.join("co2_projections", "hybrid_ensemble_fit.png")
    plt.savefig(output_path)
    print(f"\nProfessional ensemble analysis plot saved to {output_path}")
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('CO$_2$ Concentration (ppm)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(1985, 2055)
    plt.ylim(340, 560)
    
    output_path = os.path.join("co2_projections", "hybrid_ensemble_fit.png")
    plt.savefig(output_path)
    print(f"\nEnsemble plot saved to {output_path}")

if __name__ == "__main__":
    main()
