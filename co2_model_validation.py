import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from co2_prediction import load_csv, linear_func, exponential_func, rational_2_1, fit_models

# Style Constants
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
RED = "#CC79A7"
GRAY = "#999999"

def main():
    # 1. Load Historical Data (1959 - 2020)
    data = load_csv('data/co2-ppm.csv')
    years_hist, co2_hist = data[:, 0], data[:, 1]
    
    # 2. Define Validation Data (2022 - 2024)
    years_val = np.array([2022, 2023, 2024])
    co2_val = np.array([418.53, 421.08, 424.61])
    
    # 3. Fit Models to Historical Data
    # Global Linear
    _, p_lin, _ = fit_models(years_hist, co2_hist, "linear")
    
    # Segment 2 Models (Post-1990)
    mask2 = years_hist > 1990
    _, p_exp, _ = fit_models(years_hist[mask2], co2_hist[mask2], "exponential")
    _, p_rat, _ = fit_models(years_hist[mask2], co2_hist[mask2], "rational_2_1")
    
    # 4. Generate Predictions for Validation Years
    pred_lin = linear_func(years_val, *p_lin)
    pred_exp = exponential_func(years_val, *p_exp)
    pred_rat = rational_2_1(years_val, *p_rat)
    
    # 5. Evaluate Models
    def get_metrics(true, pred):
        mae = mean_absolute_error(true, pred)
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
    
    # 6. Visualization
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Historical Tail
    plt.scatter(years_hist[-10:], co2_hist[-10:], color=GRAY, alpha=0.5, label='Historical (2011-2020)')
    
    # Validation Data
    plt.scatter(years_val, co2_val, color='black', marker='X', s=100, label='Real Data (2022-2024)', zorder=5)
    
    # Model Projections (zoomed in)
    years_plot = np.linspace(2010, 2025, 100)
    plt.plot(years_plot, linear_func(years_plot, *p_lin), color=BLUE, linestyle='--', label=f'Linear Projection (RMSE={m_lin[1]:.2f})')
    plt.plot(years_plot, exponential_func(years_plot, *p_exp), color=ORANGE, linewidth=2, label=f'Piecewise Baseline (RMSE={m_exp[1]:.2f})')
    plt.plot(years_plot, rational_2_1(years_plot, *p_rat), color=GREEN, linewidth=2, label=f'Piecewise Rational (RMSE={m_rat[1]:.2f})')
    
    # Formatting
    plt.title('Model Validation: 2022-2024 Projections vs. Real-World Data', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('CO$_2$ Concentration (ppm)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(2010.5, 2024.5)
    plt.ylim(390, 430)
    
    # Annotate specific values
    for y, v in zip(years_val, co2_val):
        plt.annotate(f"{v:.2f}", (y, v), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

    output_path = os.path.join("co2_projections", "model_validation_2024.png")
    plt.savefig(output_path)
    print(f"\nValidation plot saved to {output_path}")

if __name__ == "__main__":
    main()
