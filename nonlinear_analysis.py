import numpy as np
import os

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

if __name__ == "__main__":
    co2_path = 'data/co2-ppm.csv'
    temp_path = 'data/surface-air-temp-change.csv'
    
    try:
        years, co2, temp = load_and_align_data(co2_path, temp_path)
        s1, s2 = segment_data(years, co2, temp)
        
        print("Data segmented successfully.")
        print(f"Segment 1: {int(s1[0][0])}-{int(s1[0][-1])} ({len(s1[0])} samples)")
        print(f"Segment 2: {int(s2[0][0])}-{int(s2[0][-1])} ({len(s2[0])} samples)")
    except Exception as e:
        print(f"Failed to execute nonlinear analysis: {e}")
