import numpy as np
import matplotlib.pyplot as plt

def load_csv(file_path: str) -> np.ndarray:
    """Loads a CSV file into a NumPy array, skipping the header."""
    return np.loadtxt(file_path, delimiter=',', skiprows=1)

def load_and_align_data(co2_path: str, temp_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads and aligns CO2 and temperature data for years >= 1959."""
    co2_data = load_csv(co2_path)
    temp_data = load_csv(temp_path)
    
    co2_data = co2_data[co2_data[:, 0] >= 1959]
    temp_data = temp_data[temp_data[:, 0] >= 1959]
    
    common_years = np.intersect1d(co2_data[:, 0], temp_data[:, 0])
    co2_filtered = co2_data[np.isin(co2_data[:, 0], common_years)]
    temp_filtered = temp_data[np.isin(temp_data[:, 0], common_years)]
    
    return common_years, co2_filtered[:, 1], temp_filtered[:, 1]

def calculate_zscore(data: np.ndarray) -> np.ndarray:
    """Calculates the Z-score for a given numeric array."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

if __name__ == "__main__":
    years, co2, temp = load_and_align_data('data/co2-ppm.csv', 'data/surface-air-temp-change.csv')
    co2_z = calculate_zscore(co2)
    temp_z = calculate_zscore(temp)
    
    print(f"Standardization Complete.")
    print(f"CO2 Z-score: mean={np.mean(co2_z):.4f}, std={np.std(co2_z):.4f}")
    print(f"Temp Z-score: mean={np.mean(temp_z):.4f}, std={np.std(temp_z):.4f}")
