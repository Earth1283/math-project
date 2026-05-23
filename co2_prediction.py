import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from typing import Callable, Optional

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
    try:
        return np.loadtxt(file_path, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def linear_func(x, a, b):             return a * x + b
def quadratic_func(x, a, b, c):       return a * x**2 + b * x + c
def exponential_func(x, a, b, c, x0): return a * np.exp(b * (x - x0)) + c
def rational_1_1(x, p1, p2, q1):      return (p1 * x + p2) / (q1 * x + 1)
def rational_2_1(x, p1, p2, p3, q1):  return (p1 * x**2 + p2 * x + p3) / (q1 * x + 1)

def check_rational_stability(q1: float, x_data: np.ndarray) -> bool:
    denominators = q1 * x_data + 1
    return np.all(denominators > 0) or np.all(denominators < 0)

def fit_models(x_data: np.ndarray, y_data: np.ndarray, model_name: str):
    popt = None
    r2   = -np.inf
    func = None

    try:
        if model_name == "linear":
            func  = linear_func
            popt, _ = curve_fit(func, x_data, y_data, p0=[1.5, -2800])
        elif model_name == "quadratic":
            func  = quadratic_func
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.01, -40, 40000])
        elif model_name == "exponential":
            func  = exponential_func
            x0_init = x_data.min()
            popt, _ = curve_fit(func, x_data, y_data, p0=[315, 0.01, 0, x0_init],
                                bounds=([0, 0, -np.inf, x_data.min()],
                                        [np.inf, np.inf, np.inf, x_data.max()]))
        elif model_name == "rational_1_1":
            func  = rational_1_1
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.1, 300, 0.0001])
            if popt is not None and not check_rational_stability(popt[2], x_data):
                popt = None
        elif model_name == "rational_2_1":
            func  = rational_2_1
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.001, 0.1, 300, 0.0001])
            if popt is not None and not check_rational_stability(popt[3], x_data):
                popt = None

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
        self.years      = years
        self.co2        = co2
        self.split_year = 1990

    def predict_global_linear(self, target_years):
        name, func, popt, r2 = get_best_fit(self.years, self.co2, ["linear"])
        return func(target_years, *popt), name, popt

    def predict_piecewise(self, target_years, models_to_test):
        mask1 = self.years <= self.split_year
        s1_name, s1_func, s1_popt, _ = get_best_fit(self.years[mask1], self.co2[mask1], models_to_test)

        mask2 = self.years > self.split_year
        s2_name, s2_func, s2_popt, _ = get_best_fit(self.years[mask2], self.co2[mask2], models_to_test)

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
    idx   = np.where(preds >= limit)[0]
    if len(idx) > 0:
        return years_fine[idx[0]]
    return None

def main():
    data = load_csv('data/co2-ppm.csv')
    years_hist, co2_hist = data[:, 0], data[:, 1]

    predictor    = CO2Predictor(years_hist, co2_hist)
    future_years = np.arange(1959, 2051)

    y_lin, name_lin, popt_lin = predictor.predict_global_linear(future_years)
    year_685_lin  = find_year_reaches_limit(linear_func, popt_lin, 2023)

    baseline_models = ["linear", "quadratic", "exponential"]
    y_base, names_base, popts_base, funcs_base = predictor.predict_piecewise(future_years, baseline_models)
    year_685_base = find_year_reaches_limit(funcs_base[1], popts_base[1], 2023)

    rational_models = ["linear", "quadratic", "exponential", "rational_1_1", "rational_2_1"]
    y_rat, names_rat, popts_rat, funcs_rat = predictor.predict_piecewise(future_years, rational_models)
    year_685_rat  = find_year_reaches_limit(funcs_rat[1], popts_rat[1], 2023)

    print(f"--- Model Selection ---")
    print(f"Piecewise Baseline (Seg 2): {names_base[1].capitalize()}")
    print(f"Piecewise Rational (Seg 2): {names_rat[1].capitalize()}")

    print(f"\n--- CO2 Predictions for 2050 ---")
    if year_685_lin:
        print(f"Global Linear:      {y_lin[-1]:.2f} ppm | Reaches 685 ppm in: {year_685_lin:.1f}")
    else:
        print(f"Global Linear:      {y_lin[-1]:.2f} ppm | Reaches 685 ppm in: >2500")
    if year_685_base:
        print(f"Piecewise Baseline: {y_base[-1]:.2f} ppm | Reaches 685 ppm in: {year_685_base:.1f}")
    else:
        print(f"Piecewise Baseline: {y_base[-1]:.2f} ppm | Reaches 685 ppm in: >2500")
    if year_685_rat:
        print(f"Piecewise Rational: {y_rat[-1]:.2f} ppm | Reaches 685 ppm in: {year_685_rat:.1f}")
    else:
        print(f"Piecewise Rational: {y_rat[-1]:.2f} ppm | Reaches 685 ppm in: >2500")

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

    ax.scatter(years_hist, co2_hist,
               color=GRAY, alpha=0.35, s=25, zorder=2, label='Historical Data')
    ax.plot(future_years, y_lin,
            color=BLUE,   linewidth=2.5, zorder=3,
            label=f'Global Linear  (2050: {y_lin[-1]:.1f} ppm)')
    ax.plot(future_years, y_base,
            color=ORANGE, linewidth=2.5, zorder=3,
            label=f'Piecewise Baseline  (2050: {y_base[-1]:.1f} ppm)')
    ax.plot(future_years, y_rat,
            color=GREEN,  linewidth=2.5, zorder=3,
            label=f'Piecewise Rational  (2050: {y_rat[-1]:.1f} ppm)')

    ax.axhline(685, color=RED, linestyle='--', linewidth=1.8, alpha=0.85,
               label='Threshold: 685 ppm', zorder=4)

    if year_685_lin  and year_685_lin  < 2100:
        ax.axvline(year_685_lin,  color=BLUE,   linestyle=':', alpha=0.25, zorder=1)
    if year_685_base and year_685_base < 2100:
        ax.axvline(year_685_base, color=ORANGE, linestyle=':', alpha=0.25, zorder=1)
    if year_685_rat  and year_685_rat  < 2100:
        ax.axvline(year_685_rat,  color=GREEN,  linestyle=':', alpha=0.25, zorder=1)

    ax.set_title(r'$\mathrm{CO}_2$ Concentration Projections to 2050',
                 fontsize=18, fontweight='bold', pad=14)
    ax.set_xlabel('Year',                              fontsize=13, labelpad=8)
    ax.set_ylabel(r'$\mathrm{CO}_2$ Concentration (ppm)', fontsize=13, labelpad=8)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1955, 2060)
    ax.set_ylim(300, 750)

    plt.tight_layout()
    output_path = os.path.join("co2_projections", "co2_projections_2050.png")
    plt.savefig(output_path, facecolor=BG)
    plt.close()
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()
