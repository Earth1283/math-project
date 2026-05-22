import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from typing import Callable, Optional

OUTPUT_DIR = "nonlinear_results/rational_exploration"

def load_csv(file_path: str) -> np.ndarray:
    """
    Loads a CSV file into a NumPy array, skipping the header.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        A NumPy array containing the numeric data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is not numeric.
        Exception: For any other unexpected errors.
    """
    try:
        return np.loadtxt(file_path, delimiter=',', skiprows=1)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except ValueError as e:
        print(f"Error: Could not parse numeric data from {file_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        raise

def load_and_align_data(co2_path: str, temp_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads CO2 and temperature data, filters for years >= 1959, 
    and aligns them on common years.
    
    Args:
        co2_path: Path to the CO2 ppm CSV file.
        temp_path: Path to the surface air temperature change CSV file.
        
    Returns:
        A tuple (years, co2_values, temp_values) as NumPy arrays.
    """
    co2_data = load_csv(co2_path)
    temp_data = load_csv(temp_path)
    
    # Filter for years >= 1959
    co2_data = co2_data[co2_data[:, 0] >= 1959]
    temp_data = temp_data[temp_data[:, 0] >= 1959]
    
    # Align on common years
    common_years = np.intersect1d(co2_data[:, 0], temp_data[:, 0])
    
    co2_f = co2_data[np.isin(co2_data[:, 0], common_years)]
    temp_f = temp_data[np.isin(temp_data[:, 0], common_years)]
    
    # Sort by year and extract values
    co2_sorted = co2_f[co2_f[:, 0].argsort()]
    temp_sorted = temp_f[temp_f[:, 0].argsort()]
    
    return common_years, co2_sorted[:, 1], temp_sorted[:, 1]

def segment_data(
    years: np.ndarray, 
    co2: np.ndarray, 
    temp: np.ndarray, 
    breakpoint_year: int = 1990
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Segments data into two parts based on a breakpoint year.
    
    Args:
        years: NumPy array of years.
        co2: NumPy array of CO2 values.
        temp: NumPy array of temperature values.
        breakpoint_year: The year at which to split the data (inclusive for the first segment).
        
    Returns:
        A tuple of two segments, each containing (years, co2, temp).
    """
    idx = np.where(years <= breakpoint_year)[0][-1]
    return (years[:idx+1], co2[:idx+1], temp[:idx+1]), (years[idx+1:], co2[idx+1:], temp[idx+1:])

def linear_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Linear model: y = ax + b
    
    Args:
        x: Input data array.
        a: Slope parameter.
        b: Intercept parameter.
        
    Returns:
        The calculated y values as a NumPy array.
    """
    return a * x + b

def quadratic_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Quadratic model: y = ax^2 + bx + c
    
    Args:
        x: Input data array.
        a: Quadratic coefficient.
        b: Linear coefficient.
        c: Constant term.
        
    Returns:
        The calculated y values as a NumPy array.
    """
    return a * x**2 + b * x + c

def exponential_func(x: np.ndarray, a: float, b: float, c: float, x0: float) -> np.ndarray:
    """
    Exponential model: y = a * e^(b * (x - x0)) + c
    
    Args:
        x: Input data array.
        a: Amplitude coefficient.
        b: Growth rate coefficient.
        c: Vertical offset.
        x0: Horizontal shift.
        
    Returns:
        The calculated y values as a NumPy array.
    """
    return a * np.exp(b * (x - x0)) + c

def rational_1_1(x: np.ndarray, p1: float, p2: float, q1: float) -> np.ndarray:
    """
    Rational model (1,1): y = (p1*x + p2) / (q1*x + 1)
    
    Args:
        x: Input data array.
        p1: Polynomial coefficient in numerator.
        p2: Constant term in numerator.
        q1: Polynomial coefficient in denominator.
        
    Returns:
        The calculated y values as a NumPy array.
    """
    return (p1 * x + p2) / (q1 * x + 1)

def rational_2_1(x: np.ndarray, p1: float, p2: float, p3: float, q1: float) -> np.ndarray:
    """
    Rational model (2,1): y = (p1*x^2 + p2*x + p3) / (q1*x + 1)
    
    Args:
        x: Input data array.
        p1: Quadratic coefficient in numerator.
        p2: Linear coefficient in numerator.
        p3: Constant term in numerator.
        q1: Polynomial coefficient in denominator.
        
    Returns:
        The calculated y values as a NumPy array.
    """
    return (p1 * x**2 + p2 * x + p3) / (q1 * x + 1)

def check_rational_stability(q1: float, x_data: np.ndarray) -> bool:
    """
    Checks if the rational function is stable (no poles in the range of x_data).
    Denominator q1*x + 1 must have the same sign for all x in x_data.
    
    Args:
        q1: Polynomial coefficient in denominator.
        x_data: Input data array to check for stability.
        
    Returns:
        True if the function is stable over x_data, False otherwise.
    """
    denominators = q1 * x_data + 1
    return np.all(denominators > 0) or np.all(denominators < 0)


def fit_models(
    x_data: np.ndarray, 
    y_data: np.ndarray, 
    model_name: str
) -> tuple[Optional[Callable], Optional[np.ndarray], float, Optional[np.ndarray], float]:
    """
    Fits a specified model to the data and calculates R^2 and RMSE.
    
    Args:
        x_data: Independent variable data.
        y_data: Dependent variable data.
        model_name: Name of the model to fit ('linear', 'quadratic', 'exponential').
        
    Returns:
        A tuple (func, popt, r2, y_pred, rmse). If fit fails, popt and y_pred are None, r2 is -inf, and rmse is inf.
    """
    popt = None
    y_pred = None
    r2 = -np.inf
    rmse = np.inf
    func = None

    try:
        if model_name == "linear":
            func = linear_func
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.01, -3])
        elif model_name == "quadratic":
            func = quadratic_func
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.0001, 0.01, -3])
        elif model_name == "exponential":
            func = exponential_func
            x0_init = x_data.min()
            # Bounds: a > 0, b > 0
            popt, _ = curve_fit(
                func, x_data, y_data, 
                p0=[1, 0.01, -3, x0_init],
                bounds=([0, 0, -np.inf, x_data.min()], [np.inf, np.inf, np.inf, x_data.max()])
            )
        elif model_name == "rational_1_1":
            func = rational_1_1
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.01, -3, 0.001])
            if popt is not None and not check_rational_stability(popt[2], x_data):
                popt = None
        elif model_name == "rational_2_1":
            func = rational_2_1
            popt, _ = curve_fit(func, x_data, y_data, p0=[0.0001, 0.01, -3, 0.001])
            if popt is not None and not check_rational_stability(popt[3], x_data):
                popt = None
        
        if popt is not None:
            y_pred = func(x_data, *popt)
            r2 = r2_score(y_data, y_pred)
            rmse = np.sqrt(mean_squared_error(y_data, y_pred))
            
    except RuntimeError as e:
        print(f"Error: Fit for {model_name} failed due to convergence issues: {e}")
        popt = None
        y_pred = None
        r2 = -np.inf
        rmse = np.inf
    except ValueError as e:
        print(f"Error: Fit for {model_name} failed due to invalid data: {e}")
        popt = None
        y_pred = None
        r2 = -np.inf
        rmse = np.inf
    except Exception as e:
        print(f"Warning: Fit for {model_name} failed unexpectedly: {e}")
        popt = None
        y_pred = None
        r2 = -np.inf
        rmse = np.inf

    return func, popt, r2, y_pred, rmse

def get_best_fit(x_data: np.ndarray, y_data: np.ndarray, models_to_test: list[str]) -> tuple[str, Callable, np.ndarray, float, float]:
    """
    Identifies the model with the highest R^2 for the given data among specified models.
    
    Args:
        x_data: Independent variable data.
        y_data: Dependent variable data.
        models_to_test: List of model names to compete.
        
    Returns:
        A tuple (model_name, function, parameters, r2, rmse).
    """
    best_model = None
    best_func = None
    best_popt = None
    best_r2 = -np.inf
    best_rmse = np.inf
    
    for model_name in models_to_test:
        func, popt, r2, _, rmse = fit_models(x_data, y_data, model_name)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name
            best_func = func
            best_popt = popt
            best_rmse = rmse
            
    return best_model, best_func, best_popt, best_r2, best_rmse

# Style Constants (Colorblind-friendly)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#999999"

def generate_equation_string(model_name: str, popt: np.ndarray) -> str:
    """
    Generates a LaTeX-formatted equation string for the fitted model.
    """
    if model_name == "linear":
        return f"$y = {popt[0]:.4f}x {'+' if popt[1] >= 0 else '-'} {abs(popt[1]):.4f}$"
    elif model_name == "quadratic":
        return f"$y = {popt[0]:.4f}x^2 {'+' if popt[1] >= 0 else '-'} {abs(popt[1]):.4f}x {'+' if popt[2] >= 0 else '-'} {abs(popt[2]):.4f}$"
    elif model_name == "exponential":
        return f"$y = {popt[0]:.4f}e^{{{popt[1]:.4f}(x - {popt[3]:.4f})}} {'+' if popt[2] >= 0 else '-'} {abs(popt[2]):.4f}$"
    elif model_name == "rational_1_1":
        return f"$y = ({popt[0]:.4f}x {'+' if popt[1] >= 0 else '-'} {abs(popt[1]):.4f}) / ({popt[2]:.4f}x + 1)$"
    elif model_name == "rational_2_1":
        return f"$y = ({popt[0]:.4f}x^2 {'+' if popt[1] >= 0 else '-'} {abs(popt[1]):.4f}x {'+' if popt[2] >= 0 else '-'} {abs(popt[2]):.4f}) / ({popt[3]:.4f}x + 1)$"
    return ""

def plot_combined_results(
    s1_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    s2_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    s1_best: tuple[str, Callable, np.ndarray, float, float],
    s2_best: tuple[str, Callable, np.ndarray, float, float],
    global_residuals: np.ndarray,
    piecewise_residuals: np.ndarray,
    global_r2: float,
    piecewise_r2: float,
    output_dir: str,
    file_prefix: str,
    breakpoint_year: int = 1990
):
    """
    Creates and saves a multi-panel HD plot showing fits and residuals.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    years1, co2_1, temp_1 = s1_data
    years2, co2_2, temp_2 = s2_data
    all_co2 = np.concatenate([co2_1, co2_2])
    
    # Calculate split point for plotting continuity
    split_co2 = (co2_1[-1] + co2_2[0]) / 2.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16), dpi=300)
    
    # --- Top Subplot: Piecewise Fits ---
    # Scatter actual data
    ax1.scatter(co2_1, temp_1, color=BLUE, alpha=0.3, s=40, label=f'Data (pre-{breakpoint_year})')
    ax1.scatter(co2_2, temp_2, color=ORANGE, alpha=0.3, s=40, label=f'Data (post-{breakpoint_year})')
    
    # Plot Segment 1 Best Fit
    name1, func1, popt1, r2_1, rmse1 = s1_best
    x_range1 = np.linspace(co2_1.min(), split_co2, 100)
    ax1.plot(x_range1, func1(x_range1, *popt1), color=BLUE, linewidth=3, 
             label=f'Seg 1: {name1.capitalize()} ($R^2={r2_1:.4f}$)')
    
    # Plot Segment 2 Best Fit
    name2, func2, popt2, r2_2, rmse2 = s2_best
    x_range2 = np.linspace(split_co2, co2_2.max(), 100)
    ax1.plot(x_range2, func2(x_range2, *popt2), color=ORANGE, linewidth=3, 
             label=f'Seg 2: {name2.capitalize()} ($R^2={r2_2:.4f}$)')
    
    # Vertical line for breakpoint
    ax1.axvline(split_co2, color=GRAY, linestyle='--', linewidth=1.5, alpha=0.8, label=f'Breakpoint ({breakpoint_year})')
    
    # Annotations
    eq1 = generate_equation_string(name1, popt1)
    eq2 = generate_equation_string(name2, popt2)
    stats_text = f"Segment 1 Fit:\n{eq1}\n\nSegment 2 Fit:\n{eq2}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             fontsize=12, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_xlabel('$CO_2$ Concentration (ppm)', fontsize=14)
    ax1.set_ylabel('Temperature Change (°C)', fontsize=14)
    ax1.set_title(f'Piecewise Nonlinear Fit: {file_prefix.replace("_", " ").title()}', fontsize=18, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Bottom Subplot: Residuals ---
    ax2.scatter(all_co2, global_residuals, color=GRAY, alpha=0.4, s=30, label=f'Global Linear Residuals ($R^2={global_r2:.4f}$)')
    ax2.scatter(all_co2, piecewise_residuals, color=GREEN, alpha=0.7, s=40, label=f'Piecewise Nonlinear Residuals ($R^2={piecewise_r2:.4f}$)')
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('$CO_2$ Concentration (ppm)', fontsize=14)
    ax2.set_ylabel('Residual (°C)', fontsize=14)
    ax2.set_title('Residual Comparison: Global vs. Piecewise', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{file_prefix}_analysis.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Combined analysis plot saved to {output_path}")

def main():
    """Main execution function for nonlinear analysis."""
    co2_path = 'data/co2-ppm.csv'
    temp_path = 'data/surface-air-temp-change.csv'
    
    try:
        years, co2, temp = load_and_align_data(co2_path, temp_path)
        s1, s2 = segment_data(years, co2, temp)
        print("Data segmented successfully.")
        
        # Global Linear baseline
        _, _, g_r2, g_pred, g_rmse = fit_models(co2, temp, "linear")
        global_residuals = temp - g_pred

        # --- PASS 1: Baseline Nonlinear (Linear, Quad, Exp) ---
        print("\n--- PASS 1: Baseline Nonlinear Analysis ---")
        baseline_models = ["linear", "quadratic", "exponential"]
        s1_best_bl = get_best_fit(s1[1], s1[2], baseline_models)
        s2_best_bl = get_best_fit(s2[1], s2[2], baseline_models)
        
        # Combined metrics
        _, f1_bl, p1_bl, _, _ = s1_best_bl
        pred1_bl = f1_bl(s1[1], *p1_bl)
        _, f2_bl, p2_bl, _, _ = s2_best_bl
        pred2_bl = f2_bl(s2[1], *p2_bl)
        
        p_pred_bl = np.concatenate([pred1_bl, pred2_bl])
        p_r2_bl = r2_score(temp, p_pred_bl)
        p_res_bl = temp - p_pred_bl
        
        plot_combined_results(s1, s2, s1_best_bl, s2_best_bl, global_residuals, p_res_bl, 
                             g_r2, p_r2_bl, "nonlinear_results", "baseline")

        # --- PASS 2: Rational Exploration (All models) ---
        print("\n--- PASS 2: Rational Exploration Analysis ---")
        all_models = ["linear", "quadratic", "exponential", "rational_1_1", "rational_2_1"]
        s1_best_rt = get_best_fit(s1[1], s1[2], all_models)
        s2_best_rt = get_best_fit(s2[1], s2[2], all_models)
        
        # Combined metrics
        _, f1_rt, p1_rt, _, _ = s1_best_rt
        pred1_rt = f1_rt(s1[1], *p1_rt)
        _, f2_rt, p2_rt, _, _ = s2_best_rt
        pred2_rt = f2_rt(s2[1], *p2_rt)
        
        p_pred_rt = np.concatenate([pred1_rt, pred2_rt])
        p_r2_rt = r2_score(temp, p_pred_rt)
        p_res_rt = temp - p_pred_rt
        
        plot_combined_results(s1, s2, s1_best_rt, s2_best_rt, global_residuals, p_res_rt, 
                             g_r2, p_r2_rt, "nonlinear_results", "rational_exploration")

        print("\n" + "="*40)
        print("FINAL COMPARISON SUMMARY")
        print(f"Global Linear R²:      {g_r2:.4f} | RMSE: {g_rmse:.4f}")
        print(f"Baseline Piecewise R²: {p_r2_bl:.4f}")
        print(f"Rational Piecewise R²: {p_r2_rt:.4f}")
        print(f"Total R² Improvement:  {p_r2_rt - g_r2:.4f}")
        print("="*40)

    except Exception as e:
        print(f"Failed to execute nonlinear analysis: {e}")
        raise

if __name__ == "__main__":
    main()
