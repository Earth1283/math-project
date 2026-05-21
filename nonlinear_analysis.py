import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
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


def fit_models(
    x_data: np.ndarray, 
    y_data: np.ndarray, 
    model_name: str
) -> tuple[Optional[Callable], Optional[np.ndarray], float, Optional[np.ndarray]]:
    """
    Fits a specified model to the data and calculates R^2.
    
    Args:
        x_data: Independent variable data.
        y_data: Dependent variable data.
        model_name: Name of the model to fit ('linear', 'quadratic', 'exponential').
        
    Returns:
        A tuple (func, popt, r2, y_pred). If fit fails, popt and y_pred are None, and r2 is -inf.
    """
    popt = None
    y_pred = None
    r2 = -np.inf
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
        
        if popt is not None:
            y_pred = func(x_data, *popt)
            r2 = r2_score(y_data, y_pred)
            
    except RuntimeError as e:
        print(f"Error: Fit for {model_name} failed due to convergence issues: {e}")
        popt = None
        y_pred = None
        r2 = -np.inf
    except ValueError as e:
        print(f"Error: Fit for {model_name} failed due to invalid data: {e}")
        popt = None
        y_pred = None
        r2 = -np.inf
    except Exception as e:
        print(f"Warning: Fit for {model_name} failed unexpectedly: {e}")
        popt = None
        y_pred = None
        r2 = -np.inf

    return func, popt, r2, y_pred

def get_best_fit(x_data: np.ndarray, y_data: np.ndarray) -> tuple[str, Callable, np.ndarray, float]:
    """
    Identifies the model with the highest R^2 for the given data.
    
    Args:
        x_data: Independent variable data.
        y_data: Dependent variable data.
        
    Returns:
        A tuple (model_name, function, parameters, r2).
    """
    best_model = None
    best_func = None
    best_popt = None
    best_r2 = -np.inf
    
    for model_name in ["linear", "quadratic", "exponential"]:
        func, popt, r2, _ = fit_models(x_data, y_data, model_name)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name
            best_func = func
            best_popt = popt
            
    return best_model, best_func, best_popt, best_r2

def plot_piecewise_comparison(
    s1_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    s2_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    s1_best: tuple[str, Callable, np.ndarray, float],
    s2_best: tuple[str, Callable, np.ndarray, float],
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
    os.makedirs("nonlinear_results", exist_ok=True)
    
    plt.figure(figsize=(14, 8), dpi=300)
    
    # Extract data
    years1, co2_1, temp_1 = s1_data
    years2, co2_2, temp_2 = s2_data
    
    # Scatter actual data
    plt.scatter(co2_1, temp_1, color='lightskyblue', alpha=0.5, label='Actual Data (Seg 1)')
    plt.scatter(co2_2, temp_2, color='salmon', alpha=0.5, label='Actual Data (Seg 2)')
    
    # Plot Segment 1 Best Fit
    name1, func1, popt1, r2_1 = s1_best
    x_range1 = np.linspace(co2_1.min(), co2_1.max(), 100)
    plt.plot(x_range1, func1(x_range1, *popt1), color='blue', linewidth=2, 
             label=f'Seg 1: {name1.capitalize()} (R²={r2_1:.4f})')
    
    # Plot Segment 2 Best Fit
    name2, func2, popt2, r2_2 = s2_best
    x_range2 = np.linspace(co2_2.min(), co2_2.max(), 100)
    plt.plot(x_range2, func2(x_range2, *popt2), color='red', linewidth=2, 
             label=f'Seg 2: {name2.capitalize()} (R²={r2_2:.4f})')
    
    # Vertical line for breakpoint
    breakpoint_co2 = co2_1[-1]
    plt.axvline(breakpoint_co2, color='gray', linestyle='--', alpha=0.7, label=f'Breakpoint ({breakpoint_year})')
    
    # HD Labels and Formatting
    plt.xlabel('CO2 Concentration (ppm)', fontsize=12)
    plt.ylabel('Surface Air Temperature Change (°C)', fontsize=12)
    plt.title('Piecewise Nonlinear Fit Comparison: CO2 vs Temperature', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = "nonlinear_results/piecewise_fit_comparison.png"
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
    os.makedirs("nonlinear_results", exist_ok=True)
    
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
    
    output_path = "nonlinear_results/residual_improvement.png"
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
        _, g_popt, g_r2, g_pred = fit_models(co2, temp, "linear")
        global_residuals = temp - g_pred
        
        # 2. Piecewise Nonlinear Model
        # Identify best fits
        s1_best = get_best_fit(s1[1], s1[2])
        s2_best = get_best_fit(s2[1], s2[2])
        
        print(f"Segment 1 Best Fit: {s1_best[0].capitalize()} (R^2 = {s1_best[3]:.4f})")
        print(f"Segment 2 Best Fit: {s2_best[0].capitalize()} (R^2 = {s2_best[3]:.4f})")
        
        # Calculate Piecewise Residuals
        # Seg 1 residuals
        _, f1, p1, _ = s1_best
        res1 = s1[2] - f1(s1[1], *p1)
        
        # Seg 2 residuals
        _, f2, p2, _ = s2_best
        res2 = s2[2] - f2(s2[1], *p2)
        
        piecewise_residuals = np.concatenate([res1, res2])
        
        # Combined R^2 for Piecewise
        piecewise_pred = np.concatenate([f1(s1[1], *p1), f2(s2[1], *p2)])
        combined_r2 = r2_score(temp, piecewise_pred)
        
        # Generate plots
        plot_piecewise_comparison(s1, s2, s1_best, s2_best)
        plot_residual_comparison(co2, global_residuals, piecewise_residuals, g_r2, combined_r2)
        
        # Final Summary
        print("-" * 30)
        print("ANALYSIS SUMMARY")
        print(f"Global Linear R^2:    {g_r2:.4f}")
        print(f"Piecewise Nonlinear R^2: {combined_r2:.4f}")
        print(f"Total Improvement:    {combined_r2 - g_r2:.4f}")
        print("-" * 30)

    except Exception as e:
        print(f"Failed to execute nonlinear analysis: {e}")
        raise
