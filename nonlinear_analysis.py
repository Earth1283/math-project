import numpy as np
import os

def load_csv(file_path: str) -> np.ndarray:
    try:
        return np.loadtxt(file_path, delimiter=',', skiprows=1)
    except FileNotFoundError:
        # Re-raise to match test expectation
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    except ValueError as e:
        # Re-raise to match test expectation
        raise ValueError(f"Error: Could not parse numeric data from {file_path}: {e}")

def load_and_align_data(co2_path: str, temp_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def segment_data(years, co2, temp, breakpoint_year=1990) -> tuple:
    idx = np.where(years <= breakpoint_year)[0][-1]
    return (years[:idx+1], co2[:idx+1], temp[:idx+1]), (years[idx+1:], co2[idx+1:], temp[idx+1:])

if __name__ == "__main__":
    co2_path = 'data/co2-ppm.csv'
    temp_path = 'data/surface-air-temp-change.csv'
    
    if os.path.exists(co2_path) and os.path.exists(temp_path):
        years, co2, temp = load_and_align_data(co2_path, temp_path)
        s1, s2 = segment_data(years, co2, temp)
        print(f"Data segmented. S1: {len(s1[0])}, S2: {len(s2[0])}")
    else:
        print("Data files not found. Skipping main block execution.")
