# Piecewise Nonlinear Analysis: Regime Change in CO2 vs. Temperature

## 1. Overview
Create a Python analysis suite (`nonlinear_analysis.py`) and a Manim explainer (`nonlinear-explainer.py`) to explore the hypothesis that the relationship between $CO_2$ and global temperature has fundamentally changed behavior over time. We will use piecewise nonlinear regression to fit different mathematical models to different eras.

## 2. Requirements
- **Data Source**: 
    - `data/co2-ppm.csv`
    - `data/surface-air-temp-change.csv`
- **Segmentation**: Split data into two eras (Breakpoint: Year 1990).
- **Modeling**: 
    - Attempt to fit Linear, Quadratic, and Exponential models to *each* segment independently.
    - Identify the best-fitting equation for each segment based on $R^2$ score.
- **Outputs**:
    - A folder `nonlinear_results/` containing HD (300 DPI) plots.
    - A Manim video rendered in 1080p.

## 3. Technical Strategy

### 3.1 Data Pipeline
1.  **Loading**: Re-use robust loading and alignment logic from `linear.py`.
2.  **Segmentation**: Split the aligned arrays at the index corresponding to the breakpoint year (1990).
3.  **Fitting**: Use `scipy.optimize.curve_fit` for the Exponential model and `numpy.polyfit` for the Linear/Quadratic models.

### 3.2 Matplotlib Assets (`nonlinear_results/`)
- **`piecewise_fit_comparison.png`**: Highlights the breakpoint and shows the best nonlinear fit for each segment, with equations annotated.
- **`residual_improvement.png`**: Compares residuals from a global linear fit vs. the new piecewise nonlinear fit to demonstrate superior accuracy.

### 3.3 Manim Explainer Sequence
- **Scene 1**: Show the global linear trend line from the previous explainer.
- **Scene 2**: Animate the line "snapping" at the year 1990.
- **Scene 3**: Animate each segment "bending" into its respective best nonlinear form (e.g., Segment 1 stays linear, Segment 2 bends into an exponential curve).
- **Scene 4**: Morph the specific equations into view over their respective segments.

## 4. Evaluation Metrics
- $R^2$ for every model attempted.
- Root Mean Square Error (RMSE) comparison between global and piecewise approaches.
