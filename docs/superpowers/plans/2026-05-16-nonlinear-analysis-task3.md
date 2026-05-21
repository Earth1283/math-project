# Piecewise Nonlinear Analysis Task 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify the best models for each segment and create a high-definition piecewise comparison plot.

**Architecture:** 
- Implement `get_best_fit` to automatically select the model with the highest $R^2$.
- Implement `plot_piecewise_comparison` using Matplotlib for visualization.
- Update the main block to integrate these functions and produce the output.

**Tech Stack:** NumPy, SciPy, Matplotlib, Scikit-learn.

---

### Task 1: Implement `get_best_fit`

**Files:**
- Modify: `nonlinear_analysis.py`

- [ ] **Step 1: Add `get_best_fit` function**

```python
def get_best_fit(x_data: np.ndarray, y_data: np.ndarray) -> tuple[str, Callable, np.ndarray, float]:
    """
    Identifies the model with the highest R^2 for the given data.
    
    Args:
        x_data: Independent variable data.
        y_data: Dependent variable data.
        
    Returns:
        A tuple (model_name, function, parameters, r2).
    """
    best_model = None
    best_func = None
    best_popt = None
    best_r2 = -np.inf
    
    for model_name in ["linear", "quadratic", "exponential"]:
        func, popt, r2, _ = fit_models(x_data, y_data, model_name)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name
            best_func = func
            best_popt = popt
            
    return best_model, best_func, best_popt, best_r2
```

- [ ] **Step 2: Verify `get_best_fit` with a quick print in `main`**

### Task 2: Implement `plot_piecewise_comparison`

**Files:**
- Modify: `nonlinear_analysis.py`

- [ ] **Step 1: Import Matplotlib**

```python
import matplotlib.pyplot as plt
```

- [ ] **Step 2: Add `plot_piecewise_comparison` function**

```python
def plot_piecewise_comparison(
    s1_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    s2_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    s1_best: tuple[str, Callable, np.ndarray, float],
    s2_best: tuple[str, Callable, np.ndarray, float],
    breakpoint_year: int = 1990
):
    """
    Creates and saves a high-definition piecewise comparison plot.
    
    Args:
        s1_data: Segment 1 (years, co2, temp).
        s2_data: Segment 2 (years, co2, temp).
        s1_best: Best fit info for Segment 1.
        s2_best: Best fit info for Segment 2.
        breakpoint_year: The year where segments split.
    """
    os.makedirs("nonlinear_results", exist_ok=True)
    
    plt.figure(figsize=(14, 8), dpi=300)
    
    # Extract data
    years1, co2_1, temp_1 = s1_data
    years2, co2_2, temp_2 = s2_data
    
    # Scatter actual data
    plt.scatter(co2_1, temp_1, color='lightskyblue', alpha=0.5, label='Actual Data (Seg 1)')
    plt.scatter(co2_2, temp_2, color='salmon', alpha=0.5, label='Actual Data (Seg 2)')
    
    # Plot Segment 1 Best Fit
    name1, func1, popt1, r2_1 = s1_best
    x_range1 = np.linspace(co2_1.min(), co2_1.max(), 100)
    plt.plot(x_range1, func1(x_range1, *popt1), color='blue', linewidth=2, 
             label=f'Seg 1: {name1.capitalize()} (R²={r2_1:.4f})')
    
    # Plot Segment 2 Best Fit
    name2, func2, popt2, r2_2 = s2_best
    x_range2 = np.linspace(co2_2.min(), co2_2.max(), 100)
    plt.plot(x_range2, func2(x_range2, *popt2), color='red', linewidth=2, 
             label=f'Seg 2: {name2.capitalize()} (R²={r2_2:.4f})')
    
    # Vertical line for breakpoint
    breakpoint_co2 = co2_1[-1]
    plt.axvline(breakpoint_co2, color='gray', linestyle='--', alpha=0.7, label=f'Breakpoint ({breakpoint_year})')
    
    # HD Labels and Formatting
    plt.xlabel('CO2 Concentration (ppm)', fontsize=12)
    plt.ylabel('Surface Air Temperature Change (°C)', fontsize=12)
    plt.title('Piecewise Nonlinear Fit Comparison: CO2 vs Temperature', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = "nonlinear_results/piecewise_fit_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")
```

### Task 3: Update `main` block and Verify

**Files:**
- Modify: `nonlinear_analysis.py`

- [ ] **Step 1: Update `main` block to use new functions**

```python
if __name__ == "__main__":
    co2_path = 'data/co2-ppm.csv'
    temp_path = 'data/surface-air-temp-change.csv'
    
    try:
        years, co2, temp = load_and_align_data(co2_path, temp_path)
        s1, s2 = segment_data(years, co2, temp)
        
        print("Data segmented successfully.")
        
        # Identify best fits
        s1_best = get_best_fit(s1[1], s1[2])
        s2_best = get_best_fit(s2[1], s2[2])
        
        print(f"Segment 1 Best Fit: {s1_best[0].capitalize()} (R^2 = {s1_best[3]:.4f})")
        print(f"Segment 2 Best Fit: {s2_best[0].capitalize()} (R^2 = {s2_best[3]:.4f})")
        
        # Generate plot
        plot_piecewise_comparison(s1, s2, s1_best, s2_best)

    except Exception as e:
        print(f"Failed to execute nonlinear analysis: {e}")
        raise
```

- [ ] **Step 2: Run script and verify output**

Run: `python nonlinear_analysis.py`
Expected: Print best fits and save plot to `nonlinear_results/piecewise_fit_comparison.png`.

- [ ] **Step 3: Commit**

```bash
git add nonlinear_analysis.py
git commit -m "feat: implement best fit identification and comparison plot"
```
