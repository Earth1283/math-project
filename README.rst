CO\ :sub:`2` & Global Temperature — A Regression Analysis
==========================================================

*A math project exploring whether rising CO*\ :sub:`2` *actually explains rising temperatures,
and how far we can push regression modeling before it breaks.*

.. image:: regression_analysis.png
   :width: 640
   :alt: Linear Regression: CO₂ vs. Temperature Change

----

What This Project Is
--------------------

We started with a simple question: **can you model the relationship between atmospheric
CO**\ :sub:`2` **and global temperature using regression?**

Turns out yes — but the story gets more interesting when you ask *which* model fits best,
*why* a single linear fit isn't enough, and *when* we might hit 685 ppm (roughly
2.4× pre-industrial levels). This project works through all of that, layer by layer,
using real NOAA data from 1959 to 2024.

The analysis covers:

- Basic linear regression and what R² actually means
- Z-score standardization to compare CO\ :sub:`2` (ppm) and temperature (°C) on the same scale
- Piecewise nonlinear modeling because the pre- and post-1990 eras behave differently
- A hybrid ensemble model (quadratic + exponential) that fits the full 1959–2024 arc
- Long-range forecasting and threshold projections out to 2050 and beyond

Each major concept has both a static matplotlib figure (print-ready, 300 dpi)
and a Manim animation that walks through the math step by step.

----

The Data
--------

Both datasets live in ``data/`` and are aligned on common years (1959–2020 overlap,
with 2022–2024 validation points added manually).

+--------------------------------------+---------------------------------------------+
| File                                 | Contents                                    |
+======================================+=============================================+
| ``co2-ppm.csv``                      | Yearly atmospheric CO\ :sub:`2` (ppm), NOAA |
+--------------------------------------+---------------------------------------------+
| ``surface-air-temp-change.csv``      | Yearly global surface temp anomaly (°C)     |
+--------------------------------------+---------------------------------------------+

----

Key Findings
------------

**Linear regression** gives a solid baseline:
   R² = 0.9241 — about 92% of temperature variance is explained by CO\ :sub:`2` alone.
   Slope ≈ 0.0106 °C per ppm.

**Piecewise nonlinear modeling** (splitting at 1990) bumps that to R² ≈ 0.9298.
   Pre-1990: quadratic growth. Post-1990: exponential acceleration.
   The 1990 structural break corresponds to a documented surge in industrial emissions.

**Hybrid ensemble model** (quadratic + exponential, fitted jointly on 1959–2024):
   R² = 0.9996, RMSE = 0.65 ppm. Basically perfect in-sample.
   The exponential term captures the post-2000 acceleration that a pure quadratic misses.

**685 ppm threshold** (≈ 2.4× pre-industrial):

   - Global linear model: ~2195
   - Piecewise exponential: ~2078
   - Piecewise rational: ~2079

   The spread between models is over a century — which is itself an important finding
   about how sensitive long-range forecasts are to model choice.

----

Analysis Scripts
----------------

Run any of these from the project root:

.. code-block:: bash

    python linear.py                    # linear regression + residuals → regression_analysis.png
    python zscore_analysis.py           # z-score standardization → zscore_comparison.png
    python nonlinear_analysis.py        # piecewise nonlinear fits → nonlinear_results/
    python co2_prediction.py            # 3-model projections to 2050 → co2_projections/
    python co2_ensemble_analysis.py     # hybrid ensemble fit + residuals → co2_projections/
    python co2_model_validation.py      # backtest against 2022–2024 data → co2_projections/
    python co2_model_diagnostics.py     # 2×2 diagnostic panel → co2_projections/
    python co2_threshold_visualization.py  # 685 ppm timeline → co2_projections/

Output figures are saved at 300 dpi with a clean print-ready style.

----

Manim Explainer Videos
-----------------------

Five animated explainers, each building on the last. Render at low quality (``-ql``)
for a quick preview, or high quality (``-qh``) for the final 1080p60 version.

**1. Z-Score Standardization**

.. code-block:: bash

    manim -qh zscore-explainer.py ZScoreExplainer

Explains *why* you can't just plot ppm and °C on the same axis, walks through the
z-score formula step by step, and reveals the Pearson r = 0.97 correlation.

**2. Linear Regression**

.. code-block:: bash

    manim -qh linear-explainer.py LinearRegressionExplainer

Builds the regression line visually, derives SS\ :sub:`res` and R² from scratch,
and shows the residual plot.

**3. Piecewise Nonlinear Modeling**

.. code-block:: bash

    manim -qh nonlinear-explainer.py NonlinearOdyssey

Shows why a global linear fit misses the curvature, introduces the 1990 structural
breakpoint, and fits quadratic + exponential segments with a live R² counter.

**4. CO**\ :sub:`2` **Prediction & Threshold Forecasting**

.. code-block:: bash

    manim -qh co2_prediction_video.py CO2PredictionExplainer

Animates three model curves growing toward 2050, then extends to the 685 ppm
threshold to show how far apart the model crossing-years actually land.

**5. Hybrid Ensemble Model**

.. code-block:: bash

    manim -qh co2_ensemble_video.py EnsembleExplainer

The capstone video. Shows how adding an exponential term to the quadratic baseline
closes the "acceleration gap," displays both fitted equations with coefficients,
and ends with a residual validation panel.

----

Output Files
------------

+--------------------------------------------------------+----------------------------------------+
| Path                                                   | What it is                             |
+========================================================+========================================+
| ``regression_analysis.png``                            | Linear fit + residuals                 |
+--------------------------------------------------------+----------------------------------------+
| ``zscore_comparison.png``                              | Standardized CO\ :sub:`2` vs. temp     |
+--------------------------------------------------------+----------------------------------------+
| ``co2_projections/co2_projections_2050.png``           | 3-model forecast to 2050               |
+--------------------------------------------------------+----------------------------------------+
| ``co2_projections/co2_threshold_685ppm.png``           | Threshold crossing timeline            |
+--------------------------------------------------------+----------------------------------------+
| ``co2_projections/hybrid_ensemble_fit.png``            | Ensemble fit + residuals               |
+--------------------------------------------------------+----------------------------------------+
| ``co2_projections/model_validation_2024.png``          | Backtest: 2022–2024                    |
+--------------------------------------------------------+----------------------------------------+
| ``co2_projections/model_diagnostics_testing.png``      | 2×2 diagnostic panel                  |
+--------------------------------------------------------+----------------------------------------+
| ``nonlinear_results/baseline_analysis.png``            | Piecewise baseline fit                 |
+--------------------------------------------------------+----------------------------------------+
| ``nonlinear_results/rational_exploration_analysis.png``| Piecewise rational fit                 |
+--------------------------------------------------------+----------------------------------------+
| ``media/videos/``                                      | Rendered Manim animations (mp4)        |
+--------------------------------------------------------+----------------------------------------+

----

Dependencies
------------

.. code-block:: bash

    pip install numpy matplotlib scikit-learn scipy manim

LaTeX is required for Manim's math rendering. On macOS with Homebrew:

.. code-block:: bash

    brew install basictex
    tlmgr --usermode install standalone preview
    brew install dvisvgm

All Manim files use ``config.tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")``
to avoid PostScript conflicts with Homebrew's dvisvgm.
