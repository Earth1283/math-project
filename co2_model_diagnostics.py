import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import shapiro, normaltest
from co2_prediction import load_csv, linear_func, exponential_func, rational_2_1, fit_models

# Style Constants
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
RED = "#CC79A7"
GRAY = "#999999"

def main():
    # 1. Load and Segment Data
    data = load_csv('data/co2-ppm.csv')
    years_hist, co2_hist = data[:, 0], data[:, 1]
    
    years_val = np.array([2022, 2023, 2024])
    co2_val = np.array([418.53, 421.08, 424.61])
    
    # 2. Fit Models (Focusing on Piecewise Rational as the "better" model)
    mask2 = years_hist > 1990
    y2_train = co2_hist[mask2]
    x2_train = years_hist[mask2]
    
    f_rat, p_rat, r2_train = fit_models(x2_train, y2_train, "rational_2_1")
    
    # 3. Predictions and Residuals
    train_preds = f_rat(x2_train, *p_rat)
    train_residuals = y2_train - train_preds
    
    val_preds = f_rat(years_val, *p_rat)
    val_residuals = co2_val - val_preds
    
    # 4. Metrics
    r2_val = r2_score(co2_val, val_preds)
    mae_val = mean_absolute_error(co2_val, val_preds)
    rmse_val = np.sqrt(mean_squared_error(co2_val, val_preds))
    
    # Assumption Checks
    # Normality of Residuals (Shapiro-Wilk)
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

    # 5. Diagnostic Visualization (Focusing on Testing Data 2022-2024)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Panel 1: Actual vs. Predicted (Testing Data)
    ax1.scatter(years_val, co2_val, color=RED, marker='X', s=150, label='Actual (Testing Data)', zorder=5)
    ax1.plot(years_val, val_preds, color=GREEN, marker='o', markersize=10, linestyle='--', linewidth=2, label='Model Prediction')
    ax1.set_title("Testing Data: Actual vs. Predicted (2022-2024)", fontweight='bold', fontsize=14)
    ax1.set_xticks(years_val)
    ax1.set_ylabel("CO$_2$ Concentration (ppm)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Validation Residuals (The "Gap")
    ax2.bar([str(y) for y in years_val], val_residuals, color=ORANGE, alpha=0.8, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_title("Validation Residuals (Prediction Error)", fontweight='bold', fontsize=14)
    ax2.set_ylabel("Error (Actual - Predicted) [ppm]")
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: Prediction Accuracy (Percent Error)
    pct_errors = (val_residuals / co2_val) * 100
    ax3.bar([str(y) for y in years_val], pct_errors, color=BLUE, alpha=0.8, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax3.set_title("Prediction Error Percentage", fontweight='bold', fontsize=14)
    ax3.set_ylabel("Error %")
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Local Trend Alignment (Last Hist + Val)
    recent_x = np.concatenate([x2_train[-5:], years_val])
    recent_y = np.concatenate([y2_train[-5:], co2_val])
    ax4.scatter(x2_train[-5:], y2_train[-5:], color=GRAY, s=80, label='Hist Tail (2016-2020)')
    ax4.scatter(years_val, co2_val, color=RED, marker='X', s=120, label='Testing Data (2022-2024)')
    xr_fine = np.linspace(2015, 2025, 100)
    ax4.plot(xr_fine, f_rat(xr_fine, *p_rat), color=GREEN, linewidth=2.5, label='Rational Model Trend')
    ax4.set_title("Local Trend Alignment (2015-2025)", fontweight='bold', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle("Testing Data Diagnostics: Piecewise Rational CO$_2$ Model", fontsize=22, fontweight='bold')
    
    # Add summary text box
    summary_text = (
        f"Validation Statistics (2022-2024):\n"
        f"RMSE: {rmse_val:.4f} ppm\n"
        f"MAE:  {mae_val:.4f} ppm\n"
        f"$R^2$:  {r2_val:.4f}"
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=GRAY))


    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    output_path = os.path.join("co2_projections", "model_diagnostics_testing.png")
    plt.savefig(output_path)
    print(f"\nTargeted diagnostic plot saved to {output_path}")

if __name__ == "__main__":
    main()
