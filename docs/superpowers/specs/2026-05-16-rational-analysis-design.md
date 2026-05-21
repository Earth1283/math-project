# Piecewise Rational Analysis: Minimizing Error in CO2-Temperature Modeling

## 1. Overview
Extend the piecewise analysis suite (`nonlinear_analysis.py`) to include **rational functions**. Rational fits provide extreme flexibility and may capture the "bending" relationship between $CO_2$ and temperature with significantly lower error than simple polynomials or exponentials.

## 2. Requirements
- **Data Source**: Aligned $CO_2$ and Temperature data since 1959.
- **Models to Add**:
    - **Rational [1/1]**: $y = \frac{p_1 x + p_2}{q_1 x + 1}$
    - **Rational [2/1]**: $y = \frac{p_1 x^2 + p_2 x + p_3}{q_1 x + 1}$
- **Logic**:
    - Automatic selection of the best-fit model (Rational vs. Polynomial vs. Exponential) for each segment based on $R^2$ and RMSE.
    - **Stability Constraint**: Disqualify any rational fit that has a zero in the denominator within the segment's data range.
- **Organization**:
    - Stash current baseline results in `nonlinear_results/baseline/`.
    - Save new results in `nonlinear_results/rational_exploration/`.

## 3. Technical Strategy

### 3.1 Model Expansion
- Implement `rational_1_1(x, p1, p2, q1)` and `rational_2_1(x, p1, p2, p3, q1)`.
- Use `scipy.optimize.curve_fit` with robust `p0` guesses.
- For rational stability, check if $q_1 x + 1$ has the same sign across the entire range of $x$ in the segment.

### 3.2 Result Management
- Move existing `piecewise_fit_comparison.png` and `residual_improvement.png` to the `baseline/` folder.
- Generate new high-resolution comparison plots and residual grids in the `rational_exploration/` folder.

## 4. Evaluation
- Comparison of $R^2$ and RMSE across all 5 models (Linear, Quadratic, Exponential, Rational [1/1], Rational [2/1]).
- Quantification of the marginal error reduction achieved by the rational approach.
