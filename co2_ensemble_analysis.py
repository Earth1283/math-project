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
    
    # Validation Data
    years_val = np.array([2022, 2023, 2024])
    co2_val = np.array([418.53, 421.08, 424.61])
    
    # 2. Fit Hybrid Model to ALL Data (1959-2020)
    # Using the full historical range
    x_train = years_hist
    y_train = co2_hist
    
    # Initial guesses optimized for the 1959 start point
    # p0 = [a, b, c, d, e]
    # a: small quad, b: linear growth, c: intercept at ~315, d: small exp start, e: small growth rate
    p0 = [0.005, 0.8, 310, 5.0, 0.02]
    
    try:
        popt, _ = curve_fit(hybrid_model, x_train, y_train, p0=p0, maxfev=50000)
        print("Global Hybrid Model Fit Successful (Fitting 1959-2020).")
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    # 3. Predictions and Evaluation
    y_train_pred = hybrid_model(x_train, *popt)
    r2_train = r2_score(y_train, y_train_pred)
    
    y_val_pred = hybrid_model(years_val, *popt)
    mae_val = mean_absolute_error(co2_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(co2_val, y_val_pred))
    r2_val = r2_score(co2_val, y_val_pred)

    print("\n--- GLOBAL HYBRID ENSEMBLE PERFORMANCE ---")
    print(f"Model Scope: Full Dataset (1959-2024)")
    print(f"R² (Training 1959-2020): {r2_train:.6f}")
    print(f"R² (Validation 2022-2024): {r2_val:.6f}")
    print(f"MAE (Validation): {mae_val:.4f} ppm")
    print(f"RMSE (Validation):{rmse_val:.4f} ppm")

    # 4. Visualization
    plt.figure(figsize=(14, 10), dpi=300)
    
    # Extended range for projection
    years_plot = np.linspace(1959, 2050, 300)
    y_plot = hybrid_model(years_plot, *popt)
    
    # Plot components
    plt.scatter(years_hist, co2_hist, color=GRAY, alpha=0.5, s=25, label='Historical Data (1959-2020)')
    plt.scatter(years_val, co2_val, color=RED, marker='X', s=120, label='Real Data (2022-2024)', zorder=5)
    
    plt.plot(years_plot, y_plot, color=GREEN, linewidth=3, label=f'Global Hybrid Model ($R^2$={r2_train:.4f})')
    
    # 2050 Projection
    val_2050 = hybrid_model(2050, *popt)
    plt.plot(2050, val_2050, 'o', color=GREEN, markersize=10)
    plt.annotate(f"2050 Prediction: {val_2050:.1f} ppm", (2050, val_2050), 
                 xytext=(-40, 30), textcoords='offset points', 
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
                 fontsize=12, fontweight='bold', color=GREEN,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GREEN, alpha=0.9))

    # Equation annotation
    eq_str = (f"$y(t) = ({popt[0]:.5f}t^2 + {popt[1]:.4f}t + {popt[2]:.1f}) + {popt[3]:.3f}e^{{{popt[4]:.4f}t}}$\n"
              f"where $t = Year - 1959$")
    plt.text(0.05, 0.95, f"Global Hybrid Equation:\n{eq_str}", 
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=GREEN))

    plt.title('Global CO$_2$ Forecasting: Integrated Quadratic-Exponential Model', fontsize=18, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('CO$_2$ Concentration (ppm)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(1955, 2055)
    plt.ylim(300, 560)
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
