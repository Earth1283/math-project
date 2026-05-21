import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from typing import Callable, Optional

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
    """
    return (p1 * x + p2) / (q1 * x + 1)

def rational_2_1(x: np.ndarray, p1: float, p2: float, p3: float, q1: float) -> np.ndarray:
    """
    Rational model (2,1): y = (p1*x^2 + p2*x + p3) / (q1*x + 1)
    """
    return (p1 * x**2 + p2 * x + p3) / (q1 * x + 1)

def check_rational_stability(q1: float, x_data: np.ndarray) -> bool:
    """
    Checks if the rational function is stable (no poles in the range of x_data).
    Denominator q1*x + 1 must have the same sign for all x in x_data.
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

def get_best_fit(x_data: np.ndarray, y_data: np.ndarray) -> tuple[str, Callable, np.ndarray, float, float]:
    """
    Identifies the model with the highest R^2 for the given data.
    
    Args:
        x_data: Independent variable data.
        y_data: Dependent variable data.
        
    Returns:
        A tuple (model_name, function, parameters, r2, rmse).
    """
    best_model = None
    best_func = None
    best_popt = None
    best_r2 = -np.inf
    best_rmse = np.inf
    
    for model_name in ["linear", "quadratic", "exponential", "rational_1_1", "rational_2_1"]:
        func, popt, r2, _, rmse = fit_models(x_data, y_data, model_name)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name
            best_func = func
            best_popt = popt
            best_rmse = rmse
            
    return best_model, best_func, best_popt, best_r2, best_rmse

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

def plot_piecewise_comparison(
    s1_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    s2_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    s1_best: tuple[str, Callable, np.ndarray, float, float],
    s2_best: tuple[str, Callable, np.ndarray, float, float],
    breakpoint_year: int = 1990
):
    """
    Creates and saves a high-definition piecewise comparison plot.
    
    Args:
        s1_data: Segment 1 (years, co2, temp).
        s2_data: Segment 2 (years, co2, temp).
        s1_best: Best fit info for Segment 1.
        s2_best: Best fit info for Segment 2.
        breakpoint_year: The year where segments split.
    """
    os.makedirs("nonlinear_results/rational_exploration", exist_ok=True)
    
    plt.figure(figsize=(14, 8), dpi=300)
    
    # Extract data
    years1, co2_1, temp_1 = s1_data
    years2, co2_2, temp_2 = s2_data
    
    # Scatter actual data
    plt.scatter(co2_1, temp_1, color='lightskyblue', alpha=0.5, label='Actual Data (Seg 1)')
    plt.scatter(co2_2, temp_2, color='salmon', alpha=0.5, label='Actual Data (Seg 2)')
    
    # Plot Segment 1 Best Fit
    name1, func1, popt1, r2_1, rmse1 = s1_best
    x_range1 = np.linspace(co2_1.min(), co2_1.max(), 100)
    plt.plot(x_range1, func1(x_range1, *popt1), color='blue', linewidth=2, 
             label=f'Seg 1: {name1.capitalize()} (R²={r2_1:.4f}, RMSE={rmse1:.4f})')
    
    # Plot Segment 2 Best Fit
    name2, func2, popt2, r2_2, rmse2 = s2_best
    x_range2 = np.linspace(co2_2.min(), co2_2.max(), 100)
    plt.plot(x_range2, func2(x_range2, *popt2), color='red', linewidth=2, 
             label=f'Seg 2: {name2.capitalize()} (R²={r2_2:.4f}, RMSE={rmse2:.4f})')
    
    # Add Equation Annotations
    eq1 = generate_equation_string(name1, popt1)
    eq2 = generate_equation_string(name2, popt2)
    
    plt.text(0.05, 0.95, f"Seg 1: {eq1}", transform=plt.gca().transAxes, 
             fontsize=10, color='blue', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.text(0.05, 0.88, f"Seg 2: {eq2}", transform=plt.gca().transAxes, 
             fontsize=10, color='red', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Vertical line for breakpoint
    breakpoint_co2 = co2_1[-1]
    plt.axvline(breakpoint_co2, color='gray', linestyle='--', alpha=0.7, label=f'Breakpoint ({breakpoint_year})')
    
    # HD Labels and Formatting
    plt.xlabel('CO2 Concentration (ppm)', fontsize=12)
    plt.ylabel('Surface Air Temperature Change (°C)', fontsize=12)
    plt.title('Piecewise Nonlinear Fit Comparison: CO2 vs Temperature', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = "nonlinear_results/rational_exploration/piecewise_fit_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_residual_comparison(
    co2: np.ndarray,
    global_residuals: np.ndarray,
    piecewise_residuals: np.ndarray,
    global_r2: float,
    piecewise_r2: float
):
    """
    Creates and saves a high-definition residual comparison plot.
    
    Args:
        co2: CO2 concentration data (X-axis).
        global_residuals: Residuals from the global linear model.
        piecewise_residuals: Residuals from the piecewise nonlinear model.
        global_r2: R^2 score of the global linear model.
        piecewise_r2: R^2 score of the piecewise nonlinear model.
    """
    os.makedirs("nonlinear_results/rational_exploration", exist_ok=True)
    
    plt.figure(figsize=(14, 8), dpi=300)
    
    # Plot residuals
    plt.scatter(co2, global_residuals, color='gray', alpha=0.5, label=f'Global Linear Residuals (R²={global_r2:.4f})')
    plt.scatter(co2, piecewise_residuals, color='blue', alpha=0.7, label=f'Piecewise Nonlinear Residuals (R²={piecewise_r2:.4f})')
    
    # Reference line
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)
    
    # HD Labels and Formatting
    plt.xlabel('CO2 Concentration (ppm)', fontsize=12)
    plt.ylabel('Temperature Residual (°C)', fontsize=12)
    plt.title('Residual Analysis: Global Linear vs. Piecewise Nonlinear Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = "nonlinear_results/rational_exploration/residual_improvement.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Residual plot saved to {output_path}")

if __name__ == "__main__":
    co2_path = 'data/co2-ppm.csv'
    temp_path = 'data/surface-air-temp-change.csv'
    
    try:
        years, co2, temp = load_and_align_data(co2_path, temp_path)
        s1, s2 = segment_data(years, co2, temp)
        
        print("Data segmented successfully.")
        
        # 1. Global Linear Model
        _, g_popt, g_r2, g_pred, g_rmse = fit_models(co2, temp, "linear")
        global_residuals = temp - g_pred
        
        # 2. Piecewise Nonlinear Model
        print("\nTesting Models for Segment 1:")
        for m in ["linear", "quadratic", "exponential", "rational_1_1", "rational_2_1"]:
            _, _, r2, _, _ = fit_models(s1[1], s1[2], m)
            print(f"  {m:15}: R^2 = {r2:.4f}")

        print("\nTesting Models for Segment 2:")
        for m in ["linear", "quadratic", "exponential", "rational_1_1", "rational_2_1"]:
            _, _, r2, _, _ = fit_models(s2[1], s2[2], m)
            print(f"  {m:15}: R^2 = {r2:.4f}")

        # Identify best fits
        s1_best = get_best_fit(s1[1], s1[2])
        s2_best = get_best_fit(s2[1], s2[2])
        
        print(f"Segment 1 Best Fit: {s1_best[0].capitalize()} (R^2 = {s1_best[3]:.4f}, RMSE = {s1_best[4]:.4f})")
        print(f"Segment 2 Best Fit: {s2_best[0].capitalize()} (R^2 = {s2_best[3]:.4f}, RMSE = {s2_best[4]:.4f})")
        
        # Calculate Piecewise Residuals
        # Seg 1 residuals
        _, f1, p1, _, _ = s1_best
        res1 = s1[2] - f1(s1[1], *p1)
        
        # Seg 2 residuals
        _, f2, p2, _, _ = s2_best
        res2 = s2[2] - f2(s2[1], *p2)
        
        piecewise_residuals = np.concatenate([res1, res2])
        
        # Combined R^2 and RMSE for Piecewise
        piecewise_pred = np.concatenate([f1(s1[1], *p1), f2(s2[1], *p2)])
        combined_r2 = r2_score(temp, piecewise_pred)
        combined_rmse = np.sqrt(mean_squared_error(temp, piecewise_pred))
        
        # Generate plots
        plot_piecewise_comparison(s1, s2, s1_best, s2_best)
        plot_residual_comparison(co2, global_residuals, piecewise_residuals, g_r2, combined_r2)
        
        # Final Summary
        print("-" * 30)
        print("ANALYSIS SUMMARY")
        print(f"Global Linear R^2:    {g_r2:.4f}  | RMSE: {g_rmse:.4f}")
        print(f"Piecewise Nonlinear R^2: {combined_r2:.4f}  | RMSE: {combined_rmse:.4f}")
        print(f"R^2 Improvement:      {combined_r2 - g_r2:.4f}")
        print(f"RMSE Reduction:       {g_rmse - combined_rmse:.4f}")
        print("-" * 30)

    except Exception as e:
        print(f"Failed to execute nonlinear analysis: {e}")
        raise
