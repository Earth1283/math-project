# Linear Regression: CO2 vs. Temperature Change

## 1. Overview
Create a linear regression model to analyze the relationship between atmospheric CO2 concentrations (ppm) and global surface air temperature changes (Degrees C) from 1959 onwards. The goal is to quantify the relationship and visualize both the fit and the residuals.

## 2. Requirements
- **Data Source**: 
    - `data/co2-ppm.csv` (CO2 levels)
    - `data/surface-air-temp-change.csv` (Temperature change)
- **Timeframe**: 1959 to the latest common year.
- **Model**: Linear regression using `scikit-learn`.
- **Outputs**:
    - A fitting plot showing data points and the regression line.
    - A residual plot to evaluate model assumptions.
    - Model statistics ($R^2$, Coefficient, Intercept) printed to the console.

## 3. Technical Strategy

### 3.1 Data Pipeline
1.  **Loading**: Load datasets using a method compatible with the environment (e.g., standard CSV module or NumPy).
2.  **Cleaning**:
    - Align column names.
    - Filter for year >= 1959.
    - Merge/Join data on the "Year" column to ensure synchronized data points.
3.  **Preparation**: Reshape CO2 data as the independent variable ($X$) and Temperature change as the dependent variable ($y$).

### 3.2 Modeling
- **Library**: `sklearn.linear_model.LinearRegression`.
- **Validation**: Calculate $R^2$ score and residuals.

### 3.3 Visualization
- **Library**: `matplotlib`.
- **Layout**: Two subplots arranged vertically.
- **Plot A (Fit)**: Scatter plot of CO2 vs. Temp with the predicted line.
- **Plot B (Residuals)**: Scatter plot of predicted values vs. residuals ($y_{true} - y_{pred}$) with a horizontal line at 0.

## 4. Implementation Plan (Summary)
1.  Read CSV files.
2.  Filter and merge data for years >= 1959.
3.  Instantiate and fit the `LinearRegression` model.
4.  Generate predictions and residuals.
5.  Construct the two-pane visualization using Matplotlib.
6.  Display statistics and save the plot (optional, but will display).
