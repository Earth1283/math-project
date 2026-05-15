import numpy as np
from sklearn.linear_model import LinearRegression

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
        
    except Exception as e:
        print(f"Failed to load and align data: {e}")
