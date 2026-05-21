# Manim Explainer: The Nonlinear Odyssey

## 1. Overview
Create a cinematic Manim animation (`nonlinear-explainer.py`) that narrates the discovery of the best-fitting piecewise model. The story follows a path of trial and error, moving from a simple global linear fit to a complex (but failing) rational fit, and finally to the superior "Baseline" piecewise nonlinear model.

## 2. Requirements
- **Library**: `manim`
- **Output**: 1080p high-quality render.
- **Narrative Arc**:
    1.  **Global Linear**: The initial benchmark.
    2.  **Failed Rational**: A single complex equation struggling to fit the whole series.
    3.  **Breakthrough**: Snapping the data at 1990.
    4.  **Final Baseline**: The Quadratic + Exponential piecewise model winning the competition.
- **Visuals**: Dynamic $R^2$ counter, "snapping" line geometry, and "morphing" equations.

## 3. Scene Breakdown

### Scene 1: The Global Baseline
- **Visuals**: Full axes with $CO_2$ vs. Temperature scatter data.
- **Action**: Draw the global linear regression line.
- **Display**: $R^2 = 0.9241$.

### Scene 2: Attempt 1 - The Global Rational Fit
- **Action**: Fade out the linear line. Attempt to fit a single [2/1] Rational curve to the entire dataset.
- **Result**: The curve should look "forced" (e.g., dipping unnaturally or missing key trends).
- **Text**: "Attempt 1: One size fits all? (Failed)"

### Scene 3: The Piecewise Breakthrough
- **Action**: Draw a vertical "Regime Change" line at 1990.
- **Geometry**: The global line reappears and **snaps** into two segments at the 1990 point.
- **Trial**: Briefly show two complex rational curves attempting to fit, then vibrating/disappearing as they "fail" or look unstable.

### Scene 4: The Final Revelation (Winning Baseline)
- **Action**: The two line segments **bend** into their final forms:
    - **Segment 1**: Quadratic.
    - **Segment 2**: Exponential.
- **Metrics**: The $R^2$ counter climbs visibly to **0.9291**.
- **Display**: Both winning equations fade in over their respective curves using LaTeX formatting.

## 4. Technical Implementation Notes
- Import `load_and_align_data` and `fit_models` from `nonlinear_analysis.py` to ensure the animation values are 100% accurate.
- Use `ValueTracker` for the climbing $R^2$ score.
- Ensure all "2" subscripts in $CO_2$ are properly rendered.
