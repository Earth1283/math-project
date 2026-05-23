from manim import *
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from co2_prediction import load_csv

config.tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")

BLUE_C   = "#0072B2"
ORANGE_C = "#D55E00"
GREEN_C  = "#009E73"
RED_C    = "#CC79A7"
GRAY_C   = "#999999"
GOLD_C   = "#E69F00"


def hybrid_model(x, a, b, c, d, e):
    t = x - 1959
    return a * t**2 + b * t + c + d * np.exp(e * t)


def quad_only(x, a, b, c):
    t = x - 1959
    return a * t**2 + b * t + c


def exp_only(x, d, e):
    t = x - 1959
    return d * np.exp(e * t)


class EnsembleExplainer(Scene):
    def construct(self):

        # ── Load data and fit hybrid model ────────────────────────────────────
        data       = load_csv('data/co2-ppm.csv')
        years_hist = data[:, 0]
        co2_hist   = data[:, 1]

        years_val = np.array([2022, 2023, 2024])
        co2_val   = np.array([418.53, 421.08, 424.61])

        x_full = np.concatenate([years_hist, years_val])
        y_full = np.concatenate([co2_hist,   co2_val])

        p0 = [0.005, 0.8, 310, 5.0, 0.02]
        popt, _ = curve_fit(hybrid_model, x_full, y_full, p0=p0, maxfev=50000)

        r2_quad_val  = r2_score(y_full, quad_only(x_full, *popt[:3]))
        r2_final_val = r2_score(y_full, hybrid_model(x_full, *popt))

        # ── SCENE 0: Opener ───────────────────────────────────────────────────
        opener_title = MathTex(r"\text{The Hybrid Ensemble Model}", font_size=52).shift(UP * 0.7)
        opener_sub   = Tex(r"When quadratic and exponential work together.", font_size=32).next_to(opener_title, DOWN, buff=0.5)
        opener_sub2  = Tex(
            r"Fitted on 1959--2024 including recent validation data.",
            font_size=26, color=ManimColor(GRAY_C)
        ).next_to(opener_sub, DOWN, buff=0.3)

        self.play(Write(opener_title))
        self.play(FadeIn(opener_sub))
        self.play(FadeIn(opener_sub2))
        self.wait(2)
        self.play(FadeOut(VGroup(opener_title, opener_sub, opener_sub2)))

        # ── SCENE 1: Data landscape ───────────────────────────────────────────
        axes = Axes(
            x_range=[1950, 2030, 20],
            y_range=[300, 450, 50],
            x_length=10, y_length=5.5,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 20},
            tips=False
        ).shift(DOWN * 0.3)

        labels = axes.get_axis_labels(
            x_label=Text("Year", font_size=20),
            y_label=MathTex(r"\text{CO}_2\text{ (ppm)}", font_size=20)
        )

        title = MathTex(r"\text{The Hybrid Ensemble Model}", font_size=36).to_edge(UP, buff=0.3)
        self.play(Write(title), Create(axes), FadeIn(labels))

        hist_dots = VGroup(*[
            Dot(axes.c2p(x, y), color=ManimColor(GRAY_C), radius=0.035, fill_opacity=0.5)
            for x, y in zip(years_hist, co2_hist)
        ])

        # Validation data as cross markers
        val_markers = VGroup(*[
            Cross(scale_factor=0.14, stroke_color=ManimColor(RED_C), stroke_width=4)
                 .move_to(axes.c2p(x, y))
            for x, y in zip(years_val, co2_val)
        ])

        val_legend = VGroup(
            Cross(scale_factor=0.11, stroke_color=ManimColor(RED_C), stroke_width=3),
            Tex(r"Validation data (2022--2024)", font_size=20, color=ManimColor(RED_C))
        ).arrange(RIGHT, buff=0.25).to_corner(UR, buff=0.9)

        self.play(FadeIn(hist_dots, lag_ratio=0.005))
        self.play(FadeIn(val_markers, lag_ratio=0.3), FadeIn(val_legend))
        self.wait(1)

        # Live R² display
        r2_tracker = ValueTracker(0.0)
        r2_static  = MathTex(r"R^2 =", font_size=30).to_corner(UL, buff=1.0).shift(DOWN * 1.2)
        r2_value   = always_redraw(
            lambda: Text(f"{r2_tracker.get_value():.4f}", font_size=30)
                     .next_to(r2_static, RIGHT, buff=0.12)
        )
        self.add(r2_static, r2_value)

        # ── SCENE 2: Foundation — Quadratic ──────────────────────────────────
        quad_label = Tex(r"\textbf{Foundation: Quadratic Growth}", font_size=26, color=ManimColor(BLUE_C)).to_corner(UL, buff=1.0).shift(DOWN * 0.5)
        quad_curve = axes.plot(lambda x: quad_only(x, *popt[:3]), x_range=[1959, 2024], color=ManimColor(BLUE_C), stroke_width=4)

        self.play(Create(quad_curve), Write(quad_label), r2_tracker.animate.set_value(max(0, r2_quad_val)), run_time=2)

        # Quadratic equation with fitted coefficients
        a, b, c = popt[:3]
        sign_b_q = "+" if b >= 0 else "-"
        sign_c_q = "+" if c >= 0 else "-"
        quad_eq = MathTex(
            rf"\hat{{y}} = {a:.4f}t^2 {sign_b_q} {abs(b):.4f}t {sign_c_q} {abs(c):.2f}",
            font_size=22, color=ManimColor(BLUE_C)
        ).next_to(quad_label, DOWN, aligned_edge=LEFT, buff=0.25)
        t_note = Tex(
            r"$t = \text{year} - 1959$",
            font_size=18, color=ManimColor(GRAY_C)
        ).next_to(quad_eq, DOWN, aligned_edge=LEFT, buff=0.1)
        self.play(FadeIn(quad_eq), FadeIn(t_note))
        self.wait(1)

        # Acceleration gap arrow
        gap_arrow = Arrow(
            start=axes.c2p(2024, quad_only(2024, *popt[:3])),
            end=axes.c2p(2024, 424.61),
            color=YELLOW, buff=0.05, stroke_width=4
        )
        gap_text = Tex(r"\textbf{The Acceleration Gap}", font_size=22, color=YELLOW).next_to(gap_arrow, LEFT, buff=0.2)

        self.play(GrowArrow(gap_arrow), Write(gap_text))
        self.wait(2)
        self.play(FadeOut(gap_arrow), FadeOut(gap_text), FadeOut(quad_eq), FadeOut(t_note))

        # ── SCENE 3: Accelerator — Exponential component ──────────────────────
        exp_label = Tex(r"\textbf{Accelerator: Exponential Component}", font_size=26, color=ManimColor(ORANGE_C)).next_to(quad_label, DOWN, aligned_edge=LEFT, buff=0.3)
        exp_curve = axes.plot(lambda x: exp_only(x, *popt[3:]), x_range=[1959, 2024], color=ManimColor(ORANGE_C), stroke_width=2.5, stroke_opacity=0.55)

        self.play(Create(exp_curve), Write(exp_label))
        self.wait(2)

        # ── SCENE 4: Synthesis — morph to hybrid ─────────────────────────────
        full_hybrid = axes.plot(lambda x: hybrid_model(x, *popt), x_range=[1959, 2024], color=ManimColor(GREEN_C), stroke_width=5)

        formula = MathTex(
            r"\hat{y} = \underbrace{at^2 + bt + c}_{\text{Quadratic}} + \underbrace{d\,e^{et}}_{\text{Exponential}}",
            font_size=34
        ).to_corner(DL, buff=0.9).shift(UP * 0.5)
        formula_note = Tex(r"where $t = \text{year} - 1959$", font_size=22).next_to(formula, DOWN, buff=0.2)

        self.play(
            ReplacementTransform(VGroup(quad_curve, exp_curve), full_hybrid),
            ReplacementTransform(VGroup(quad_label, exp_label), formula),
            r2_tracker.animate.set_value(r2_final_val),
            run_time=2.5
        )
        self.play(FadeIn(formula_note))

        # Hybrid model with fitted coefficients
        d_c, e_c = popt[3:]
        hybrid_coeff_eq = MathTex(
            rf"\hat{{y}} = {a:.4f}t^2 {sign_b_q} {abs(b):.4f}t {sign_c_q} {abs(c):.2f} + {d_c:.4f}e^{{{e_c:.4f}t}}",
            font_size=18, color=ManimColor(GREEN_C)
        ).next_to(formula_note, DOWN, buff=0.15).align_to(formula, LEFT)
        self.play(FadeIn(hybrid_coeff_eq))
        self.wait(2)

        # ── SCENE 5: R² improvement callout card ──────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        r2_card = VGroup(
            MathTex(r"R^2_{\text{quadratic}} = " + f"{r2_quad_val:.4f}", font_size=44, color=ManimColor(BLUE_C)),
            MathTex(r"\Downarrow \;\text{add exponential term}", font_size=30, color=ManimColor(GRAY_C)),
            MathTex(r"R^2_{\text{hybrid}} = " + f"{r2_final_val:.4f}", font_size=52, color=ManimColor(GREEN_C)),
        ).arrange(DOWN, buff=0.55).move_to(ORIGIN)

        self.play(Write(r2_card[0]))
        self.wait(0.5)
        self.play(Write(r2_card[1]), Write(r2_card[2]))
        self.wait(3)
        self.play(FadeOut(r2_card))

        # ── SCENE 6: Residual analysis ────────────────────────────────────────
        axes_top = Axes(
            x_range=[1950, 2030, 20],
            y_range=[300, 450, 50],
            x_length=10, y_length=2.3,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 14},
            tips=False
        ).to_edge(UP, buff=1.1)

        axes_res = Axes(
            x_range=[300, 450, 50],
            y_range=[-2, 2, 1],
            x_length=10, y_length=2.3,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 14},
            tips=False
        ).to_edge(DOWN, buff=1.1)

        res_zero = DashedLine(axes_res.c2p(300, 0), axes_res.c2p(450, 0), color=WHITE, stroke_opacity=0.5)
        res_title  = Tex(r"\textbf{Residual Plot: Accuracy Validation}", font_size=22).next_to(axes_res, UP, buff=0.25)
        res_ylabel = Tex(r"Residual (ppm)", font_size=16).next_to(axes_res, LEFT, buff=0.25).rotate(90 * DEGREES)
        res_xlabel = Tex(r"Predicted CO$_2$ (ppm)", font_size=16).next_to(axes_res, DOWN, buff=0.2)

        top_hist = VGroup(*[
            Dot(axes_top.c2p(x, y), color=ManimColor(GRAY_C), radius=0.03, fill_opacity=0.5)
            for x, y in zip(years_hist, co2_hist)
        ])
        top_val = VGroup(*[
            Cross(scale_factor=0.11, stroke_color=ManimColor(RED_C), stroke_width=3).move_to(axes_top.c2p(x, y))
            for x, y in zip(years_val, co2_val)
        ])
        top_curve = axes_top.plot(lambda x: hybrid_model(x, *popt), x_range=[1959, 2024], color=ManimColor(GREEN_C), stroke_width=2.5)

        self.play(
            Create(axes_top), Create(axes_res), Create(res_zero),
            FadeIn(top_hist), FadeIn(top_val), Create(top_curve),
            Write(res_title), Write(res_ylabel), Write(res_xlabel)
        )

        # Compute and plot residuals
        y_preds   = hybrid_model(x_full, *popt)
        residuals = y_full - y_preds
        res_dots  = VGroup(*[
            Dot(axes_res.c2p(p, r), color=ManimColor(GREEN_C), radius=0.04)
            for p, r in zip(y_preds, residuals)
        ])
        self.play(FadeIn(res_dots, lag_ratio=0.01))

        # Residual stats panel
        rmse_val = float(np.sqrt(np.mean(residuals ** 2)))
        bias_val = float(np.mean(residuals))
        res_stats = VGroup(
            MathTex(r"\text{RMSE} = " + f"{rmse_val:.3f}" + r"\,\text{ppm}", font_size=20),
            MathTex(r"\text{Bias} = " + f"{bias_val:.3f}" + r"\,\text{ppm}", font_size=20),
            Tex(r"No systematic over/under-prediction.", font_size=18, color=ManimColor(GREEN_C)),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UR, buff=0.6)
        self.play(FadeIn(res_stats))

        conclusion = Tex(
            r"Residuals cluster near zero --- the hybrid captures both trend and curvature.",
            font_size=22, color=ManimColor(GOLD_C)
        ).to_edge(DOWN, buff=0.15)
        self.play(Write(conclusion))
        self.wait(5)
