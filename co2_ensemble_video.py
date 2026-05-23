from manim import *
import numpy as np
from scipy.optimize import curve_fit
import calendar
from co2_prediction import load_csv

# --- Constants & Colors ---
BLUE_C = "#0072B2"
ORANGE_C = "#D55E00"
GREEN_C = "#009E73"
RED_C = "#CC79A7"
GRAY_C = "#999999"

def hybrid_model(x, a, b, c, d, e):
    t = x - 1959
    quad_part = a * t**2 + b * t + c
    exp_part = d * np.exp(e * t)
    return quad_part + exp_part

def quad_only(x, a, b, c):
    t = x - 1959
    return a * t**2 + b * t + c

def exp_only(x, d, e):
    t = x - 1959
    return d * np.exp(e * t)

class EnsembleExplainer(Scene):
    def construct(self):
        # 1. Setup Data & Fit
        data = load_csv('data/co2-ppm.csv')
        years_hist, co2_hist = data[:, 0], data[:, 1]
        years_val = np.array([2022, 2023, 2024])
        co2_val = np.array([418.53, 421.08, 424.61])
        
        x_full = np.concatenate([years_hist, years_val])
        y_full = np.concatenate([co2_hist, co2_val])
        
        p0 = [0.005, 0.8, 310, 5.0, 0.02]
        popt, _ = curve_fit(hybrid_model, x_full, y_full, p0=p0, maxfev=50000)

        # Helpers
        def format_date(decimal_year):
            year = int(decimal_year)
            remainder = decimal_year - year
            month_idx = int(remainder * 12) + 1
            month_idx = max(1, min(12, month_idx))
            return f"{calendar.month_name[month_idx]} {year}"

        def get_co2_label(font_size=24, color=WHITE):
            main = Text("CO", font_size=font_size, color=color)
            sub = Text("2", font_size=font_size * 0.7, color=color)
            sub.next_to(main, DR, buff=0.03).shift(UP * 0.08)
            return VGroup(main, sub)

        # 2. Scene 1: The Data Landscape
        axes = Axes(
            x_range=[1950, 2060, 20],
            y_range=[300, 550, 50],
            x_length=10,
            y_length=6,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 20},
            tips=False
        ).shift(DOWN * 0.3)
        
        labels = axes.get_axis_labels(
            x_label=Text("Year", font_size=20),
            y_label=VGroup(get_co2_label(font_size=20), Text(" (ppm)", font_size=18)).arrange(RIGHT, buff=0.1)
        )
        
        title = VGroup(Text("Constructing the ", font_size=36), get_co2_label(font_size=36), Text(" Ensemble", font_size=36)).arrange(RIGHT, buff=0.1).to_edge(UP)
        
        self.play(Write(title), Create(axes), FadeIn(labels))
        
        hist_dots = VGroup(*[Dot(axes.c2p(x, y), color=GRAY_C, radius=0.03, fill_opacity=0.4) for x, y in zip(years_hist, co2_hist)])
        val_dots = VGroup(*[Dot(axes.c2p(x, y), color=RED_C, radius=0.06) for x, y in zip(years_val, co2_val)])
        
        self.play(FadeIn(hist_dots, lag_ratio=0.005))
        self.play(FadeIn(val_dots, lag_ratio=0.2))
        self.wait(1)

        # Metrics - Redrawable for live changes
        r2_tracker = ValueTracker(0.0)
        def get_r2_text():
            val = r2_tracker.get_value()
            main = Text("R", font_size=24)
            sup = Text("2", font_size=16).next_to(main, UR, buff=0.03).shift(DOWN*0.05)
            eq = Text(f" = {val:.4f}", font_size=24)
            return VGroup(main, sup, eq).arrange(RIGHT, buff=0.05)
            
        r2_display = always_redraw(lambda: get_r2_text().to_corner(UL, buff=1.0).shift(DOWN*1.2))
        self.add(r2_display)
        
        # 3. Scene 2: The Foundation (Quadratic)
        quad_curve = axes.plot(lambda x: quad_only(x, *popt[:3]), x_range=[1959, 2024], color=BLUE_C, stroke_width=4)
        quad_label = Text("Foundation: Quadratic Growth", font_size=24, color=BLUE_C).to_corner(UL, buff=1.0).shift(DOWN*0.5)
        
        # Calculate Quad R2
        from sklearn.metrics import r2_score
        y_quad_pred = quad_only(x_full, *popt[:3])
        r2_quad = r2_score(y_full, y_quad_pred)
        
        self.play(Create(quad_curve), Write(quad_label), r2_tracker.animate.set_value(max(0, r2_quad)), run_time=2)
        self.wait(1)
        
        # Highlight the gap at the end
        gap_arrow = Arrow(start=axes.c2p(2024, quad_only(2024, *popt[:3])), end=axes.c2p(2024, 424.61), color=YELLOW, buff=0.1)
        gap_text = Text("The Acceleration Gap", font_size=20, color=YELLOW).next_to(gap_arrow, LEFT)
        
        self.play(GrowArrow(gap_arrow), Write(gap_text))
        self.wait(2)
        self.play(FadeOut(gap_arrow), FadeOut(gap_text))

        # 4. Scene 3: The Accelerator (Exponential)
        exp_curve = axes.plot(lambda x: exp_only(x, *popt[3:]), x_range=[1959, 2024], color=ORANGE_C, stroke_width=2, stroke_opacity=0.5)
        exp_label = Text("Accelerator: Exponential Component", font_size=24, color=ORANGE_C).next_to(quad_label, DOWN, aligned_edge=LEFT)
        
        self.play(Create(exp_curve), Write(exp_label))
        self.wait(2)
        
        # 5. Scene 4: Synthesis (Morph)
        full_hybrid_curve = axes.plot(lambda x: hybrid_model(x, *popt), x_range=[1959, 2024], color=GREEN_C, stroke_width=5)
        
        formula = VGroup(
            Text("Integrated Model = ", font_size=24),
            Text("Quadratic", font_size=24, color=BLUE_C),
            Text(" + ", font_size=24),
            Text("Exponential", font_size=24, color=ORANGE_C)
        ).arrange(RIGHT, buff=0.1).to_corner(DL, buff=1.0).shift(UP*0.5)
        
        # Final R2
        r2_final = r2_score(y_full, hybrid_model(x_full, *popt))

        self.play(
            ReplacementTransform(VGroup(quad_curve, exp_curve), full_hybrid_curve),
            ReplacementTransform(VGroup(quad_label, exp_label), formula),
            r2_tracker.animate.set_value(r2_final),
            run_time=2
        )
        self.wait(2)

        # 6. Scene 5: Residual Analysis (New Ending)
        # Shift current scene to top half - REDUCED HEIGHT to avoid overlap
        new_axes_top = Axes(
            x_range=[1950, 2030, 20],
            y_range=[300, 450, 50],
            x_length=10,
            y_length=2.2,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 14},
            tips=False
        ).to_edge(UP, buff=1.0)
        
        # Bottom Axes for Residuals - REDUCED HEIGHT and repositioned
        axes_res = Axes(
            x_range=[300, 450, 50],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=2.2,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 14},
            tips=False
        ).to_edge(DOWN, buff=1.2)
        
        res_label = Text("Residuals (ppm)", font_size=16).next_to(axes_res, LEFT, buff=0.2).rotate(90*DEGREES)
        res_x_label = Text("Predicted CO2 (ppm)", font_size=16).next_to(axes_res, DOWN, buff=0.2)
        res_zero_line = DashedLine(axes_res.c2p(300, 0), axes_res.c2p(450, 0), color=WHITE, stroke_opacity=0.5)

        # Transition to two-panel
        self.play(
            FadeOut(title), FadeOut(labels), FadeOut(formula), FadeOut(r2_display),
            ReplacementTransform(axes, new_axes_top),
            *[d.animate.move_to(new_axes_top.c2p(x_full[i], y_full[i])) for i, d in enumerate(VGroup(*hist_dots, *val_dots))],
            full_hybrid_curve.animate.become(new_axes_top.plot(lambda x: hybrid_model(x, *popt), x_range=[1959, 2024], color=GREEN_C, stroke_width=3)),
            run_time=2
        )
        
        # Draw residual panel components
        res_title = Text("Residual Plot: Accuracy Validation", font_size=24, weight=BOLD).next_to(axes_res, UP, buff=0.3)
        self.play(Create(axes_res), Create(res_zero_line), Write(res_label), Write(res_x_label), Write(res_title))
        
        # Generate and plot residuals live
        y_preds = hybrid_model(x_full, *popt)
        residuals = y_full - y_preds
        res_dots = VGroup(*[Dot(axes_res.c2p(p, r), color=GREEN_C, radius=0.03) for p, r in zip(y_preds, residuals)])
        
        self.play(FadeIn(res_dots, lag_ratio=0.01))
        
        conclusion = Text("Model captures entire historical trend with minimal bias.", font_size=20, color=GOLD).to_edge(DOWN, buff=0.15)
        self.play(Write(conclusion))

        self.wait(5)
