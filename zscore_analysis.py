import numpy as np
import matplotlib.pyplot as plt

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
    Loads and aligns CO2 and temperature data for years >= 1959.
    
    Args:
        co2_path: Path to the CO2 ppm CSV file.
        temp_path: Path to the surface air temperature change CSV file.
        
    Returns:
        A tuple (years, co2_values, temp_values) as NumPy arrays.
    """
    co2_data = load_csv(co2_path)
    temp_data = load_csv(temp_path)
    
    co2_data = co2_data[co2_data[:, 0] >= 1959]
    temp_data = temp_data[temp_data[:, 0] >= 1959]
    
    common_years = np.intersect1d(co2_data[:, 0], temp_data[:, 0])
    co2_filtered = co2_data[np.isin(co2_data[:, 0], common_years)]
    temp_filtered = temp_data[np.isin(temp_data[:, 0], common_years)]
    
    # Sort by year to ensure they are perfectly aligned
    co2_filtered = co2_filtered[co2_filtered[:, 0].argsort()]
    temp_filtered = temp_filtered[temp_filtered[:, 0].argsort()]
    
    return common_years, co2_filtered[:, 1], temp_filtered[:, 1]

def calculate_zscore(data: np.ndarray) -> np.ndarray:
    """
    Calculates the Z-score for a given numeric array.
    
    Args:
        data: A NumPy array of numeric values.
        
    Returns:
        A NumPy array containing the standardized Z-scores.
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def plot_zscores(years: np.ndarray, co2_z: np.ndarray, temp_z: np.ndarray) -> None:
    """
    Creates and saves a time-series plot comparing standardized CO2 and temperature trends.
    """
    # Calculate correlation coefficient (R)
    correlation = np.corrcoef(co2_z, temp_z)[0, 1]
    
    # High-Definition Figure
    plt.figure(figsize=(14, 8), dpi=300)
    
    plt.plot(years, co2_z, label='$CO_2$ Concentration (Z-score)', color=BLUE, linewidth=2.5)
    plt.plot(years, temp_z, label='Temperature Change (Z-score)', color=ORANGE, linewidth=2.5, alpha=0.8)

    # Reference line at mean (0)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    # Correlation Annotation Box
    plt.text(0.02, 0.95, f"Correlation Coefficient:\n$R = {correlation:.4f}$", 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title('Standardized Trend Comparison: $CO_2$ vs. Temperature', fontsize=18, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Standard Deviations (Z-score)', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('zscore_comparison.png')
    print("Plot saved to zscore_comparison.png")

if __name__ == "__main__":
    years, co2, temp = load_and_align_data('data/co2-ppm.csv', 'data/surface-air-temp-change.csv')
    co2_z = calculate_zscore(co2)
    temp_z = calculate_zscore(temp)
    
    print(f"Standardization Complete.")
    print(f"CO2 Z-score: mean={np.mean(co2_z):.4f}, std={np.std(co2_z):.4f}")
    print(f"Temp Z-score: mean={np.mean(temp_z):.4f}, std={np.std(temp_z):.4f}")
    
    plot_zscores(years, co2_z, temp_z)
