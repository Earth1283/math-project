import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from typing import Callable, Optional

# Style Constants (Colorblind-friendly)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
RED = "#CC79A7"
GRAY = "#999999"

def load_csv(file_path: str) -> np.ndarray:
    try:
        return np.loadtxt(file_path, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

# --- Model Definitions (Reused from nonlinear_analysis.py) ---
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
    r2 = -np.inf
    func = None

    try:
        if model_name == "linear":
            func = linear_func
            popt, _ = curve_fit(func, x_data, y_data, p0=[1.5, -2800])
        elif model_name == "quadratic":
            func = quadratic_func
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.01, -40, 40000])
        elif model_name == "exponential":
            func = exponential_func
            x0_init = x_data.min()
            popt, _ = curve_fit(func, x_data, y_data, p0=[315, 0.01, 0, x0_init],
                                bounds=([0, 0, -np.inf, x_data.min()], [np.inf, np.inf, np.inf, x_data.max()]))
        elif model_name == "rational_1_1":
            func = rational_1_1
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.1, 300, 0.0001])
            if popt is not None and not check_rational_stability(popt[2], x_data): popt = None
        elif model_name == "rational_2_1":
            func = rational_2_1
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.001, 0.1, 300, 0.0001])
            if popt is not None and not check_rational_stability(popt[3], x_data): popt = None
        
        if popt is not None:
            y_pred = func(x_data, *popt)
            r2 = r2_score(y_data, y_pred)
            return func, popt, r2
    except:
        pass
    return None, None, -np.inf

def get_best_fit(x_data: np.ndarray, y_data: np.ndarray, models_to_test: list[str]):
    best_res = (None, None, None, -np.inf)
    for name in models_to_test:
        f, p, r = fit_models(x_data, y_data, name)
        if r > best_res[3]:
            best_res = (name, f, p, r)
    return best_res

class CO2Predictor:
    def __init__(self, years, co2):
        self.years = years
        self.co2 = co2
        self.split_year = 1990
        
    def predict_global_linear(self, target_years):
        name, func, popt, r2 = get_best_fit(self.years, self.co2, ["linear"])
        return func(target_years, *popt), name, popt

    def predict_piecewise(self, target_years, models_to_test):
        # Segment 1: Pre-1990
        mask1 = self.years <= self.split_year
        s1_name, s1_func, s1_popt, _ = get_best_fit(self.years[mask1], self.co2[mask1], models_to_test)
        
        # Segment 2: Post-1990
        mask2 = self.years > self.split_year
        s2_name, s2_func, s2_popt, _ = get_best_fit(self.years[mask2], self.co2[mask2], models_to_test)
        
        # For prediction, we use the Segment 2 model for any year > 1990
        predictions = []
        for y in target_years:
            if y <= self.split_year:
                predictions.append(s1_func(y, *s1_popt))
            else:
                predictions.append(s2_func(y, *s2_popt))
        return np.array(predictions), (s1_name, s2_name), (s1_popt, s2_popt), (s1_func, s2_func)

def find_year_reaches_limit(model_fn, params, start_year, limit=685, max_year=2500):
    years_fine = np.linspace(start_year, max_year, (max_year - start_year) * 10 + 1)
    preds = model_fn(years_fine, *params)
    idx = np.where(preds >= limit)[0]
    if len(idx) > 0:
        return years_fine[idx[0]]
    return None

def main():
    data = load_csv('data/co2-ppm.csv')
    years_hist, co2_hist = data[:, 0], data[:, 1]
    
    predictor = CO2Predictor(years_hist, co2_hist)
    
    future_years = np.arange(1959, 2051)
    
    # 1. Global Linear
    y_lin, name_lin, popt_lin = predictor.predict_global_linear(future_years)
    year_685_lin = find_year_reaches_limit(linear_func, popt_lin, 2023)
    
    # 2. Piecewise Baseline
    baseline_models = ["linear", "quadratic", "exponential"]
    y_base, names_base, popts_base, funcs_base = predictor.predict_piecewise(future_years, baseline_models)
    year_685_base = find_year_reaches_limit(funcs_base[1], popts_base[1], 2023)
    
    # 3. Piecewise Rational
    rational_models = ["linear", "quadratic", "exponential", "rational_1_1", "rational_2_1"]
    y_rat, names_rat, popts_rat, funcs_rat = predictor.predict_piecewise(future_years, rational_models)
    year_685_rat = find_year_reaches_limit(funcs_rat[1], popts_rat[1], 2023)
    
    # Results
    print(f"--- Model Selection ---")
    print(f"Piecewise Baseline (Seg 2): {names_base[1].capitalize()}")
    print(f"Piecewise Rational (Seg 2): {names_rat[1].capitalize()}")
    
    print(f"\n--- CO2 Predictions for 2050 ---")
    print(f"Global Linear:      {y_lin[-1]:.2f} ppm | Reaches 685 ppm in: {year_685_lin:.1f}" if year_685_lin else f"Global Linear:      {y_lin[-1]:.2f} ppm | Reaches 685 ppm in: >2500")
    print(f"Piecewise Baseline: {y_base[-1]:.2f} ppm | Reaches 685 ppm in: {year_685_base:.1f}" if year_685_base else f"Piecewise Baseline: {y_base[-1]:.2f} ppm | Reaches 685 ppm in: >2500")
    print(f"Piecewise Rational: {y_rat[-1]:.2f} ppm | Reaches 685 ppm in: {year_685_rat:.1f}" if year_685_rat else f"Piecewise Rational: {y_rat[-1]:.2f} ppm | Reaches 685 ppm in: >2500")
    
    # Plotting
    plt.figure(figsize=(14, 10), dpi=300)
    
    plt.scatter(years_hist, co2_hist, color=GRAY, alpha=0.5, s=30, label='Historical Data')
    plt.plot(future_years, y_lin, color=BLUE, linewidth=2.5, label=f'Global Linear (2050: {y_lin[-1]:.1f})')
    plt.plot(future_years, y_base, color=ORANGE, linewidth=2.5, label=f'Piecewise Baseline (2050: {y_base[-1]:.1f})')
    plt.plot(future_years, y_rat, color=GREEN, linewidth=2.5, label=f'Piecewise Rational (2050: {y_rat[-1]:.1f})')
    
    # Horizontal line at 685 ppm
    plt.axhline(685, color=RED, linestyle='--', alpha=0.7, label='Target: 685 ppm')
    
    # Annotate years
    if year_685_lin and year_685_lin < 2100: plt.axvline(year_685_lin, color=BLUE, linestyle=':', alpha=0.5)
    if year_685_base and year_685_base < 2100: plt.axvline(year_685_base, color=ORANGE, linestyle=':', alpha=0.5)
    if year_685_rat and year_685_rat < 2100: plt.axvline(year_685_rat, color=GREEN, linestyle=':', alpha=0.5)

    plt.title('CO$_2$ Concentration Projections to 2050', fontsize=18, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('CO$_2$ Concentration (ppm)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(1955, 2060)
    plt.ylim(300, 750)
    
    output_path = "co2_projections_2050.png"
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()
