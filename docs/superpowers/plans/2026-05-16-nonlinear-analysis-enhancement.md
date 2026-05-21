# Nonlinear Analysis Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RMSE metrics and LaTeX-formatted equation annotations to the nonlinear analysis output and plots.

**Architecture:**
1. Update core fitting functions to calculate and return RMSE.
2. Implement a LaTeX equation generator for each model type.
3. Update plotting functions to display equations and RMSE.
4. Update main summary report.

**Tech Stack:** Python (NumPy, SciPy, Matplotlib, Scikit-learn)

---

### Task 1: Update Fitting Logic

**Files:**
- Modify: `nonlinear_analysis.py`

- [ ] **Step 1: Import `mean_squared_error`**
Add `from sklearn.metrics import mean_squared_error` to imports.

- [ ] **Step 2: Update `fit_models`**
Calculate RMSE and update return signature.
```python
def fit_models(x_data, y_data, model_name):
    # ... fitting logic ...
    if popt is not None:
        y_pred = func(x_data, *popt)
        r2 = r2_score(y_data, y_pred)
        rmse = np.sqrt(mean_squared_error(y_data, y_pred))
    else:
        rmse = np.inf
    return func, popt, r2, y_pred, rmse
```

- [ ] **Step 3: Update `get_best_fit`**
Update to track and return RMSE.
```python
def get_best_fit(x_data, y_data):
    # ...
    best_rmse = np.inf
    # ... in loop ...
    func, popt, r2, _, rmse = fit_models(x_data, y_data, model_name)
    if r2 > best_r2:
        best_rmse = rmse
        # ...
    return best_model, best_func, best_popt, best_r2, best_rmse
```

- [ ] **Step 4: Commit**
`git add nonlinear_analysis.py && git commit -m "feat: update fitting logic to include RMSE"`

### Task 2: Implement Equation Generation and Plotting

**Files:**
- Modify: `nonlinear_analysis.py`

- [ ] **Step 1: Add `generate_equation_string` function**
```python
def generate_equation_string(model_name, popt):
    if model_name == "linear":
        return f"$y = {popt[0]:.4f}x + {popt[1]:.4f}$"
    elif model_name == "quadratic":
        return f"$y = {popt[0]:.4f}x^2 + {popt[1]:.4f}x + {popt[2]:.4f}$"
    elif model_name == "exponential":
        return f"$y = {popt[0]:.4f}e^{{{popt[1]:.4f}(x - {popt[3]:.4f})}} + {popt[2]:.4f}$"
    return ""
```

- [ ] **Step 2: Update `plot_piecewise_comparison`**
Update signature to include RMSEs and add annotations.
```python
def plot_piecewise_comparison(s1_data, s2_data, s1_best, s2_best, breakpoint_year=1990):
    # ...
    name1, func1, popt1, r2_1, rmse1 = s1_best
    # legend: label=f'Seg 1: {name1.capitalize()} (R²={r2_1:.4f}, RMSE={rmse1:.4f})'
    # annotation:
    eq1 = generate_equation_string(name1, popt1)
    plt.text(co2_1.min(), temp_1.max(), eq1, color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    # same for s2 ...
```

- [ ] **Step 3: Commit**
`git add nonlinear_analysis.py && git commit -m "feat: add equation strings and RMSE to piecewise plot"`

### Task 3: Update Main Execution and Summary

**Files:**
- Modify: `nonlinear_analysis.py`

- [ ] **Step 1: Update `main` block**
Update calls to `fit_models` and `get_best_fit`. Calculate combined RMSE. Update print statements.

- [ ] **Step 2: Run and Verify**
`python3 nonlinear_analysis.py`
Check `nonlinear_results/piecewise_fit_comparison.png` for equations and RMSE in legend.
Check console output for RMSE values.

- [ ] **Step 3: Commit**
`git add nonlinear_analysis.py && git commit -m "refactor: add RMSE metrics and equation annotations to nonlinear analysis"`
