import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
    """
    Loads a CSV file into a NumPy array, skipping the header.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        A NumPy array containing the numeric data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is not numeric.
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
    
    # Filter for year >= 1959 (Column 0 is Year)
    co2_data = co2_data[co2_data[:, 0] >= 1959]
    temp_data = temp_data[temp_data[:, 0] >= 1959]
    
    # Align on common years
    co2_years = co2_data[:, 0]
    temp_years = temp_data[:, 0]
    common_years = np.intersect1d(co2_years, temp_years)
    
    # Filter both datasets to only include common years
    co2_filtered = co2_data[np.isin(co2_years, common_years)]
    temp_filtered = temp_data[np.isin(temp_years, common_years)]
    
    # Sort by year to ensure they are perfectly aligned
    co2_filtered = co2_filtered[co2_filtered[:, 0].argsort()]
    temp_filtered = temp_filtered[temp_filtered[:, 0].argsort()]
    
    # Extract years, CO2, and Temp
    years = co2_filtered[:, 0]
    co2_values = co2_filtered[:, 1]
    temp_values = temp_filtered[:, 1]
    
    return years, co2_values, temp_values

def fit_model(X: np.ndarray, y: np.ndarray) -> tuple[LinearRegression, np.ndarray, float, float, float]:
    """
    Fits a linear regression model to the data.
    
    Args:
        X: Feature array (CO2 values).
        y: Target array (Temperature values).
        
    Returns:
        A tuple (model, X_reshaped, r2, coef, intercept).
    """
    X_reshaped = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_reshaped, y)
    
    r2 = model.score(X_reshaped, y)
    coef = model.coef_[0]
    intercept = model.intercept_
    
    return model, X_reshaped, r2, coef, intercept

def plot_results(X: np.ndarray, y: np.ndarray, model: LinearRegression) -> None:
    y_pred    = model.predict(X)
    residuals = y - y_pred

    coef      = model.coef_[0]
    intercept = model.intercept_
    r2        = model.score(X, y)
    equation  = f"$\\hat{{y}} = {coef:.4f}x + ({intercept:.4f})$"

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=300,
                                   gridspec_kw={"height_ratios": [2, 1]})

    # Top: Regression Line
    ax1.scatter(X, y,
                color=BLUE, alpha=0.55, s=40,
                edgecolors='none', label='Actual Data', zorder=2)
    ax1.plot(X, y_pred,
             color=RED, linewidth=2.8,
             label=f'Regression Line\n{equation}', zorder=3)

    stats_text = f"$R^2 = {r2:.4f}$\nSlope = {coef:.4f} °C / ppm"
    ax1.text(0.05, 0.96, stats_text, transform=ax1.transAxes,
             fontsize=13, verticalalignment='top', color=TEXT_C,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=AX_BG,
                       edgecolor=SPINE_C, alpha=0.95))

    ax1.set_title(r'Linear Regression: $\mathrm{CO}_2$ vs. Temperature Change',
                  fontsize=16, fontweight='bold', pad=12)
    ax1.set_xlabel(r'$\mathrm{CO}_2$ Concentration (ppm)', fontsize=13, labelpad=8)
    ax1.set_ylabel('Temperature Change (°C)',               fontsize=13, labelpad=8)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom: Residuals
    ax2.scatter(y_pred, residuals,
                color=GREEN, alpha=0.6, s=40,
                edgecolors='none', label='Residuals', zorder=2)
    ax2.axhline(y=0, color=TEXT_C, linestyle='--', linewidth=1.4, alpha=0.6)

    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Predicted Temperature Change (°C)', fontsize=13, labelpad=8)
    ax2.set_ylabel('Residuals (°C)',                   fontsize=13, labelpad=8)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=2.5)
    plt.savefig('regression_analysis.png', facecolor=BG)
    plt.close()
    print("Plots saved to regression_analysis.png")

if __name__ == "__main__":
    CO2_FILE = 'data/co2-ppm.csv'
    TEMP_FILE = 'data/surface-air-temp-change.csv'
    
    try:
        years, co2, temp = load_and_align_data(CO2_FILE, TEMP_FILE)
        print(f"Data aligned successfully.")
        print(f"Number of samples: {len(years)}")
        print("\nFirst 5 entries (Year, CO2, Temp):")
        for i in range(min(5, len(years))):
            print(f"{int(years[i])}: {co2[i]:.2f} ppm, {temp[i]:.2f} °C")
            
        # Fit model
        model, X_reshaped, r2, coef, intercept = fit_model(co2, temp)
        
        print("\nModel Statistics:")
        print(f"R² Score:    {r2:.4f}")
        print(f"Coefficient: {coef:.4f}")
        print(f"Intercept:   {intercept:.4f}")
        
        # Plot results
        plot_results(X_reshaped, temp, model)
        
    except Exception as e:
        print(f"Failed to load and align data: {e}")
