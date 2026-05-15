import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)

def load_and_align_data(co2_path, temp_path):
    co2_data = load_csv(co2_path)
    temp_data = load_csv(temp_path)
    
    # Filter for year >= 1959
    co2_data = co2_data[co2_data[:, 0] >= 1959]
    temp_data = temp_data[temp_data[:, 0] >= 1959]
    
    # Align on common years
    co2_years = co2_data[:, 0]
    temp_years = temp_data[:, 0]
    common_years = np.intersect1d(co2_years, temp_years)
    
    co2_filtered = co2_data[np.isin(co2_years, common_years)]
    temp_filtered = temp_data[np.isin(temp_years, common_years)]
    
    # Sort to ensure matching
    co2_filtered = co2_filtered[co2_filtered[:, 0].argsort()]
    temp_filtered = temp_filtered[temp_filtered[:, 0].argsort()]
    
    return common_years, co2_filtered[:, 1], temp_filtered[:, 1]

if __name__ == "__main__":
    years, co2, temp = load_and_align_data('data/co2-ppm.csv', 'data/surface-air-temp-change.csv')
    print(f"Data aligned. Number of samples: {len(years)}")
    print(f"First 5 years: {years[:5]}")
    print(f"First 5 CO2: {co2[:5]}")
    print(f"First 5 Temp: {temp[:5]}")
