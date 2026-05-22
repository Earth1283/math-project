import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from typing import Callable, Optional

# Style Constants (Colorblind-friendly)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
RED = "#CC79A7"
GRAY = "#999999"

def load_csv(file_path: str) -> np.ndarray:
    return np.loadtxt(file_path, delimiter=',', skiprows=1)

def linear_func(x, a, b): return a * x + b
def quadratic_func(x, a, b, c): return a * x**2 + b * x + c
def exponential_func(x, a, b, c, x0): return a * np.exp(b * (x - x0)) + c
def rational_1_1(x, p1, p2, q1): return (p1 * x + p2) / (q1 * x + 1)
def rational_2_1(x, p1, p2, p3, q1): return (p1 * x**2 + p2 * x + p3) / (q1 * x + 1)

def check_rational_stability(q1: float, x_data: np.ndarray) -> bool:
    denominators = q1 * x_data + 1
    return np.all(denominators > 0) or np.all(denominators < 0)

def fit_models(x_data: np.ndarray, y_data: np.ndarray, model_name: str):
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
    except: pass
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
    
    # Global Linear
    f_lin, p_lin = fit_models(years_hist, co2_hist, "linear")
    
    # Segment 2 Models (Post-1990)
    mask2 = years_hist > 1990
    f_exp, p_exp = fit_models(years_hist[mask2], co2_hist[mask2], "exponential")
    f_rat, p_rat = fit_models(years_hist[mask2], co2_hist[mask2], "rational_2_1")
    
    # Threshold crossings
    year_685_lin = find_year_reaches_limit(f_lin, p_lin, 2023)
    year_685_exp = find_year_reaches_limit(f_exp, p_exp, 2023)
    year_685_rat = find_year_reaches_limit(f_rat, p_rat, 2023)
    
    # Visualization
    plt.figure(figsize=(14, 10), dpi=300)
    
    # Extended range to see all crossings
    plot_years = np.linspace(1959, 2210, 1000)
    
    plt.plot(plot_years, f_lin(plot_years, *p_lin), color=BLUE, linewidth=2, label='Global Linear Projection')
    plt.plot(plot_years, f_exp(plot_years, *p_exp), color=ORANGE, linewidth=2, label='Piecewise Baseline (Exponential)')
    plt.plot(plot_years, f_rat(plot_years, *p_rat), color=GREEN, linewidth=2, label='Piecewise Rational (Rational 2/1)')
    
    # Historical Data
    plt.scatter(years_hist, co2_hist, color=GRAY, alpha=0.3, s=20, label='Historical Data')
    
    # Target Line
    plt.axhline(685, color=RED, linestyle='--', linewidth=1.5, label='Threshold: 685 ppm')
    
    # Crossing Markers and Annotations
    crossings = [
        (year_685_lin, BLUE, "Global Linear"),
        (year_685_exp, ORANGE, "Piecewise Baseline"),
        (year_685_rat, GREEN, "Piecewise Rational")
    ]
    
    # Sort by year to handle overlapping labels
    crossings.sort()
    
    offsets = [15, 45, 15] # Alternate offsets to avoid stacking
    for i, (yr, color, name) in enumerate(crossings):
        if yr:
            plt.plot(yr, 685, 'o', color=color, markersize=10, markeredgecolor='white', zorder=5)
            plt.annotate(f"{yr:.1f}", (yr, 685), textcoords="offset points", xytext=(0, offsets[i]),
                         ha='center', fontsize=12, fontweight='bold', color=color,
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.9),
                         arrowprops=dict(arrowstyle="-", color=color, alpha=0.5))
            plt.axvline(yr, color=color, linestyle=':', alpha=0.3)

    plt.title('Timeline to 685 ppm CO$_2$: Model Comparison', fontsize=18, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('CO$_2$ Concentration (ppm)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.xlim(1950, 2220)
    plt.ylim(300, 750)
    
    output_path = os.path.join("co2_projections", "co2_threshold_685ppm.png")
    plt.savefig(output_path)
    print(f"Threshold visualization saved to {output_path}")
    print(f"Years reaching 685 ppm:")
    print(f" - Linear:    {year_685_lin:.1f}")
    print(f" - Baseline:  {year_685_exp:.1f}")
    print(f" - Rational:  {year_685_rat:.1f}")

if __name__ == "__main__":
    main()
