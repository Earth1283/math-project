# Nonlinear Analysis Enhancement Design

**Date:** 2026-05-16
**Topic:** RMSE Metrics and Equation Annotations

## 1. Purpose
Enhance the nonlinear analysis report and visualizations by adding Root Mean Square Error (RMSE) metrics and explicit mathematical equations for the fitted models.

## 2. Requirements
1. **RMSE Metric**:
    - Calculate RMSE for all fitted models.
    - Include RMSE in the legend of `piecewise_fit_comparison.png`.
    - Include RMSE in the final console summary (Global vs. Piecewise).
2. **Equation Annotations**:
    - Generate LaTeX-formatted equation strings for the best-fit models in each segment.
    - Add these strings as text annotations on the `piecewise_fit_comparison.png` plot.
    - Example format: $y = 0.012x - 3.5$.

## 3. Architecture Changes

### `fit_models`
- Update return signature to include `rmse`.
- Calculate `rmse` using `np.sqrt(mean_squared_error(y_data, y_pred))`.

### `get_best_fit`
- Update return signature to include `rmse`.

### `plot_piecewise_comparison`
- Update to accept RMSE for each segment.
- Update legend labels to include RMSE.
- Logic to generate LaTeX strings based on `model_name` and `popt`.
- Add `plt.text` or `ax.annotate` to place equation strings on the plot.

### `main` block
- Update logic to handle RMSE from `fit_models` and `get_best_fit`.
- Update the final summary printout.

## 4. Equation Generation Logic
- **Linear**: $y = ax + b$
- **Quadratic**: $y = ax^2 + bx + c$
- **Exponential**: $y = a \cdot e^{b(x - x_0)} + c$
- Values will be rounded to 4 decimal places for readability.

## 5. Success Criteria
- `piecewise_fit_comparison.png` shows:
    - Segment 1 & 2 best fit equations as text.
    - Legend showing R² and RMSE for both segments.
- Console output shows:
    - Global Linear RMSE.
    - Piecewise Nonlinear RMSE.
    - R² comparison (existing).
- No regressions in existing functionality.
