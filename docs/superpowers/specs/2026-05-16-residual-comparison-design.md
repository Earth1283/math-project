# Design Spec: Residual Comparison Plot

## 1. Goal
Create a visualization that demonstrates the superiority of the piecewise nonlinear approach by comparing its residuals against those of a global linear model.

## 2. Components

### 2.1 Model Fitting
- **Global Linear Model**: Fit a single linear model ($y = ax + b$) to the entire dataset.
- **Piecewise Nonlinear Model**: Use the best-fitting models already identified for each segment (Segment 1 and Segment 2).

### 2.2 Residual Calculation
- $Residual = y_{actual} - y_{predicted}$
- **Global Residuals**: Calculated across the entire dataset using the global linear model.
- **Piecewise Residuals**: Concatenation of residuals from Segment 1 and Segment 2 best fits.

### 2.3 Visualization (`plot_residual_comparison`)
- **Figure Size**: 14x8 inches.
- **Resolution**: 300 DPI.
- **Styling**:
    - Global Residuals: Gray scatter points (`color='gray'`, `alpha=0.5`, `label='Global Linear Residuals'`).
    - Piecewise Residuals: Blue scatter points (`color='blue'`, `alpha=0.7`, `label='Piecewise Nonlinear Residuals'`).
    - Reference Line: Horizontal line at $y=0$ (`color='black'`, `linestyle='--'`).
- **Annotations**:
    - HD Labels: "CO2 Concentration (ppm)" (X) and "Temperature Residual (°C)" (Y).
    - Title: "Residual Analysis: Global Linear vs. Piecewise Nonlinear Model".
    - Legend: Including $R^2$ values for both models.
    - Grid: Enabled for readability.
- **Output**: `nonlinear_results/residual_improvement.png`.

## 3. Data Flow
1. `main` block:
    - Load and align data.
    - Segment data.
    - Fit global linear model -> Calculate global residuals and $R^2$.
    - Get best piecewise fits (existing logic) -> Calculate piecewise residuals and combined $R^2$.
    - Call `plot_residual_comparison`.
    - Print summary.

## 4. Verification
- Run `python3 nonlinear_analysis.py`.
- Confirm `nonlinear_results/residual_improvement.png` exists and is visually correct.
- Confirm console output shows $R^2$ improvement.
