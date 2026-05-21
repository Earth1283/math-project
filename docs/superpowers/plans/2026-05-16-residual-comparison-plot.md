# Residual Comparison Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a visualization that proves the piecewise nonlinear approach is superior by comparing residuals against a global linear model.

**Architecture:** Calculate residuals for a global linear model and the piecewise nonlinear models, then scatter plot them to show the reduction in error.

**Tech Stack:** Python, NumPy, Matplotlib, Scikit-learn (r2_score).

---

### Task 1: Implement `plot_residual_comparison`

**Files:**
- Modify: `nonlinear_analysis.py`

- [ ] **Step 1: Add the plotting function**

```python
def plot_residual_comparison(
    co2: np.ndarray,
    global_residuals: np.ndarray,
    piecewise_residuals: np.ndarray,
    global_r2: float,
    piecewise_r2: float
):
    """
    Creates and saves a high-definition residual comparison plot.
    
    Args:
        co2: CO2 concentration data (X-axis).
        global_residuals: Residuals from the global linear model.
        piecewise_residuals: Residuals from the piecewise nonlinear model.
        global_r2: R^2 score of the global linear model.
        piecewise_r2: R^2 score of the piecewise nonlinear model.
    """
    os.makedirs("nonlinear_results", exist_ok=True)
    
    plt.figure(figsize=(14, 8), dpi=300)
    
    # Plot residuals
    plt.scatter(co2, global_residuals, color='gray', alpha=0.5, label=f'Global Linear Residuals (R²={global_r2:.4f})')
    plt.scatter(co2, piecewise_residuals, color='blue', alpha=0.7, label=f'Piecewise Nonlinear Residuals (R²={piecewise_r2:.4f})')
    
    # Reference line
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)
    
    # HD Labels and Formatting
    plt.xlabel('CO2 Concentration (ppm)', fontsize=12)
    plt.ylabel('Temperature Residual (°C)', fontsize=12)
    plt.title('Residual Analysis: Global Linear vs. Piecewise Nonlinear Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = "nonlinear_results/residual_improvement.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Residual plot saved to {output_path}")
```

- [ ] **Step 2: Commit**

```bash
git add nonlinear_analysis.py
git commit -m "feat: add plot_residual_comparison function"
```

---

### Task 2: Update `main` block and Calculate Residuals

**Files:**
- Modify: `nonlinear_analysis.py`

- [ ] **Step 1: Update the main execution flow**

```python
if __name__ == "__main__":
    co2_path = 'data/co2-ppm.csv'
    temp_path = 'data/surface-air-temp-change.csv'
    
    try:
        years, co2, temp = load_and_align_data(co2_path, temp_path)
        s1, s2 = segment_data(years, co2, temp)
        
        print("Data segmented successfully.")
        
        # 1. Global Linear Model
        _, g_popt, g_r2, g_pred = fit_models(co2, temp, "linear")
        global_residuals = temp - g_pred
        
        # 2. Piecewise Nonlinear Model
        # Identify best fits
        s1_best = get_best_fit(s1[1], s1[2])
        s2_best = get_best_fit(s2[1], s2[2])
        
        print(f"Segment 1 Best Fit: {s1_best[0].capitalize()} (R^2 = {s1_best[3]:.4f})")
        print(f"Segment 2 Best Fit: {s2_best[0].capitalize()} (R^2 = {s2_best[3]:.4f})")
        
        # Calculate Piecewise Residuals
        # Seg 1 residuals
        _, f1, p1, _ = s1_best
        res1 = s1[2] - f1(s1[1], *p1)
        
        # Seg 2 residuals
        _, f2, p2, _ = s2_best
        res2 = s2[2] - f2(s2[1], *p2)
        
        piecewise_residuals = np.concatenate([res1, res2])
        
        # Combined R^2 for Piecewise
        piecewise_pred = np.concatenate([f1(s1[1], *p1), f2(s2[1], *p2)])
        combined_r2 = r2_score(temp, piecewise_pred)
        
        # Generate plots
        plot_piecewise_comparison(s1, s2, s1_best, s2_best)
        plot_residual_comparison(co2, global_residuals, piecewise_residuals, g_r2, combined_r2)
        
        # Final Summary
        print("-" * 30)
        print("ANALYSIS SUMMARY")
        print(f"Global Linear R^2:    {g_r2:.4f}")
        print(f"Piecewise Nonlinear R^2: {combined_r2:.4f}")
        print(f"Total Improvement:    {combined_r2 - g_r2:.4f}")
        print("-" * 30)

    except Exception as e:
        print(f"Failed to execute nonlinear analysis: {e}")
        raise
```

- [ ] **Step 2: Run the script to verify**

Run: `python3 nonlinear_analysis.py`
Expected: 
- Console output shows "Total Improvement" (positive value).
- `nonlinear_results/piecewise_fit_comparison.png` is generated.
- `nonlinear_results/residual_improvement.png` is generated.

- [ ] **Step 3: Commit**

```bash
git add nonlinear_analysis.py
git commit -m "feat: implement residual comparison plot and project wrap-up"
```
