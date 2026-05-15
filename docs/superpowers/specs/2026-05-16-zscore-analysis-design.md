# Z-Score Analysis: Standardized Trend Comparison

## 1. Overview
Create a Python script (`zscore_analysis.py`) to standardize atmospheric CO2 and temperature change data using Z-score transformations. This allows for a direct visual comparison of their trends over time on a single scale, which is essential for the Manim explainer video narrative.

## 2. Requirements
- **Data Source**: 
    - `data/co2-ppm.csv`
    - `data/surface-air-temp-change.csv`
- **Timeframe**: 1959 to the latest common year.
- **Transformation**: $z = \frac{x - \mu}{\sigma}$ for both variables.
- **Visualization**:
    - A single time-series plot with both Z-scores.
    - Legend identifying CO2 and Temperature lines.
    - Horizontal line at $y=0$ (the mean for both).
    - Save as `zscore_comparison.png`.

## 3. Technical Strategy

### 3.1 Data Pipeline
1.  **Loading**: Re-use the robust NumPy loading logic from `linear.py`.
2.  **Standardization**:
    - Calculate `mean` and `std` for CO2.
    - Calculate `mean` and `std` for Temperature.
    - Compute Z-scores for both.
3.  **Verification**: Confirm both Z-scored arrays have a mean of approximately 0 and a standard deviation of 1.

### 3.2 Visualization
- **Library**: `matplotlib`.
- **Plot**: Two lines on the same Y-axis.
- **Aesthetics**: Blue for CO2, Red for Temperature. Grid lines and 0-reference line.

## 4. Manim Integration (Forward Looking)
- The Manim script will use this logic to show the lines "normalizing" into each other.
- The visual will demonstrate how correlation is easier to see when data is standardized.
