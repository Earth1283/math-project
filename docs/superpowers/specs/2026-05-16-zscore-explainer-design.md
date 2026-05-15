# Manim Explainer: Z-Score Normalization

## 1. Overview
Create a Manim animation (`zscore-explainer.py`) that visually explains why and how we use Z-score standardization to compare datasets with vastly different units (CO2 ppm vs. Temperature Change °C).

## 2. Requirements
- **Library**: `manim`
- **Output**: 1080p high-quality render.
- **Narrative Focus**: 
    1. The problem of mismatched scales.
    2. The step-by-step mathematical transformation ($z = \frac{x-\mu}{\sigma}$).
    3. The revelation of correlation on a unified scale.
- **Humor**: Include the specific "well that's awkward 😅" moment when scales collide.

## 3. Scene Breakdown

### Scene 1: The Scale Problem
- **Visuals**: Two separate graphs side-by-side. 
    - Left: CO2 (300-420 range).
    - Right: Temp (-0.5 to 1.5 range).
- **Action**: Merge the two axes into one.
- **Result**: The CO2 line shoots off the top of the screen; the Temp line looks like a horizontal flat line at the bottom.
- **Punchline**: Text "well that's awkward 😅" fades in.

### Scene 2: The Math of Standardization
- **Visuals**: Show formula $z = \frac{x - \mu}{\sigma}$.
- **Step 1: Shift**: Animate the CO2 line shifting down by its mean ($\mu_{CO2}$).
- **Step 2: Stretch**: Animate the line resizing as it's divided by its standard deviation ($\sigma_{CO2}$).
- **Result**: CO2 is now centered at 0 with a unit scale.

### Scene 3: The Revelation
- **Action**: Repeat the transformation for Temperature simultaneously.
- **Result**: Both lines now oscillate on the same scale (-3 to 3 Z-score range).
- **Analysis**: Fade in a conclusion about how their trends are almost identical once standardized.

## 4. Technical Implementation Notes
- Use `zscore_analysis.py` logic for calculating exact means and standard deviations.
- Use manually constructed math symbols (VGroups) to ensure proper rendering without a full LaTeX environment.
- Use `Transform` animations for the "shift and stretch" effect to make the math feel physical.
