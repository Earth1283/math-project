from manim import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from nonlinear_analysis import load_and_align_data, segment_data, fit_models, quadratic_func, exponential_func, linear_func

config.tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")

BLUE_C   = "#0072B2"
ORANGE_C = "#D55E00"
GREEN_C  = "#009E73"
RED_C    = "#CC79A7"
GRAY_C   = "#999999"
GOLD_C   = "#E69F00"

class NonlinearOdyssey(Scene):
    def construct(self):

        # ── Load REAL data and fit models ─────────────────────────────────────
        years, co2, temp = load_and_align_data('data/co2-ppm.csv', 'data/surface-air-temp-change.csv')
        s1, s2 = segment_data(years, co2, temp, 1990)

        _, popt_lin, r2_lin, y_pred_lin, _ = fit_models(co2, temp, 'linear')
        _, popt_s1,  r2_s1,  y_pred_s1,  _ = fit_models(s1[1], s1[2], 'quadratic')
        _, popt_s2,  r2_s2,  y_pred_s2,  _ = fit_models(s2[1], s2[2], 'exponential')

        bp_year = 1990
        idx     = np.where(years <= bp_year)[0][-1]
        bp_co2  = co2[idx]

        y_pred_total = np.concatenate([y_pred_s1, y_pred_s2])
        total_r2     = r2_score(temp, y_pred_total)

        # ── SCENE 0: Opener ───────────────────────────────────────────────────
        opener_title = Tex(r"\textbf{Piecewise Nonlinear Modeling}", font_size=52).shift(UP * 0.6)
        opener_sub   = Tex(
            r"What if the CO$_2$--temperature relationship changes over time?",
            font_size=30
        ).next_to(opener_title, DOWN, buff=0.5)
        self.play(Write(opener_title))
        self.play(FadeIn(opener_sub))
        self.wait(2)
        self.play(FadeOut(opener_title), FadeOut(opener_sub))

        # ── Setup axes ────────────────────────────────────────────────────────
        title = Tex(r"\textbf{Piecewise Nonlinear Modeling}", font_size=36).to_edge(UP, buff=0.3)
        self.play(Write(title))

        axes = Axes(
            x_range=[310, 420, 20],
            y_range=[-0.5, 1.5, 0.5],
            x_length=9, y_length=4.8,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 20},
            tips=False
        ).shift(DOWN * 0.5)

        co2_lab  = VGroup(MathTex(r"\text{CO}_2", font_size=28), Text(" (ppm)", font_size=20)).arrange(RIGHT, buff=0.1)
        temp_lab = Text("Temp Change (°C)", font_size=20)
        labels   = axes.get_axis_labels(x_label=co2_lab, y_label=temp_lab)

        dots = VGroup(*[Dot(axes.c2p(x, y), color=ManimColor(BLUE_C), radius=0.045, fill_opacity=0.6) for x, y in zip(co2, temp)])

        self.play(Create(axes), FadeIn(labels))
        self.play(FadeIn(dots, lag_ratio=0.01))
        self.wait(1)

        # Live R² display
        r2_tracker = ValueTracker(0.0)
        r2_static  = MathTex(r"R^2 =", font_size=36, color=YELLOW).to_edge(RIGHT, buff=1.8).shift(UP * 2.0)
        r2_value   = always_redraw(
            lambda: Text(f"{max(0, r2_tracker.get_value()):.4f}", font_size=36, color=YELLOW)
                     .next_to(r2_static, RIGHT, buff=0.12)
        )
        self.add(r2_static, r2_value)

        # ── SCENE 1: Global Linear Baseline ──────────────────────────────────
        lin_label   = Tex(r"\textbf{Baseline: Global Linear Fit}", font_size=26).to_corner(UL).shift(DOWN * 0.5)
        a_lin, b_lin = popt_lin
        sign_lin     = "+" if b_lin >= 0 else "-"
        lin_eq       = MathTex(
            rf"\hat{{y}} = {a_lin:.4f}x \; {sign_lin} \; {abs(b_lin):.4f}",
            font_size=22
        ).next_to(lin_label, DOWN, aligned_edge=LEFT)
        lin_comment  = Tex(r"\textit{Captures the trend, but misses curvature.}", font_size=20).next_to(lin_eq, DOWN, aligned_edge=LEFT)

        line_lin = axes.plot(lambda x: linear_func(x, *popt_lin), color=WHITE)

        self.play(Write(lin_label), Write(lin_eq))
        self.play(Create(line_lin))
        self.play(r2_tracker.animate.set_value(r2_lin), run_time=1.5)
        self.play(FadeIn(lin_comment))
        self.wait(2)

        # ── SCENE 2: Global Rational — illustrative failure ───────────────────
        rat_label   = Tex(r"\textbf{Attempt: Global Rational Fit}", font_size=26, color=ManimColor(RED_C)).to_corner(UL).shift(DOWN * 0.5)
        rat_comment = Tex(r"\textit{Numerically unstable --- poles near the data range.}", font_size=20, color=ManimColor(RED_C)).next_to(rat_label, DOWN, aligned_edge=LEFT)

        line_fail = axes.plot(
            lambda x: (0.5 * (x - 310)**2 - 10 * (x - 310) + 5) / (2 * (x - 310) + 1) - 0.2,
            x_range=[315, 415], color=ManimColor(RED_C)
        )

        self.play(
            FadeOut(VGroup(lin_label, lin_eq, lin_comment)),
            Write(rat_label)
        )
        self.play(Create(line_fail), r2_tracker.animate.set_value(0.15), run_time=2)
        self.play(Write(rat_comment))

        # Vibration effect — the model is unstable
        for _ in range(3):
            self.play(line_fail.animate.shift(UP * 0.05 + RIGHT * 0.02), run_time=0.05)
            self.play(line_fail.animate.shift(DOWN * 0.1 + LEFT * 0.04), run_time=0.05)
            self.play(line_fail.animate.shift(UP * 0.05 + RIGHT * 0.02), run_time=0.05)

        self.wait(2)
        self.play(FadeOut(VGroup(rat_label, rat_comment, line_fail)))

        # ── SCENE 3: Breakpoint insight — full-screen callout ─────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob not in []])

        insight_card = VGroup(
            Tex(r"\textbf{Key Insight: Structural Breakpoint at 1990}", font_size=38, color=ManimColor(GOLD_C)),
            Tex(r"Post-1990 industrial acceleration drove a regime change\\in CO$_2$ growth rate.", font_size=30),
            Tex(r"Two regimes $\;\Rightarrow\;$ two different models.", font_size=28, color=ManimColor(GRAY_C))
        ).arrange(DOWN, buff=0.5).move_to(ORIGIN)

        self.play(Write(insight_card[0]))
        self.play(FadeIn(insight_card[1]))
        self.play(FadeIn(insight_card[2]))
        self.wait(3)
        self.play(FadeOut(insight_card))

        # ── Rebuild axes for piecewise scene ──────────────────────────────────
        title2  = Tex(r"\textbf{Piecewise Nonlinear Modeling}", font_size=36).to_edge(UP, buff=0.3)
        axes2   = axes.copy()
        labels2 = labels.copy()
        dots2   = dots.copy()

        self.play(Write(title2), Create(axes2), FadeIn(labels2))
        self.play(FadeIn(dots2, lag_ratio=0.01))

        r2_tracker2 = ValueTracker(r2_lin)
        r2_static2  = MathTex(r"R^2 =", font_size=36, color=YELLOW).to_edge(RIGHT, buff=1.8).shift(UP * 2.0)
        r2_value2   = always_redraw(
            lambda: Text(f"{max(0, r2_tracker2.get_value()):.4f}", font_size=36, color=YELLOW)
                     .next_to(r2_static2, RIGHT, buff=0.12)
        )
        # Show linear line as starting point
        line_lin2 = axes2.plot(lambda x: linear_func(x, *popt_lin), color=WHITE)
        self.add(r2_static2, r2_value2)
        self.play(Create(line_lin2))
        self.wait(0.5)

        # Breakpoint line
        break_text = Tex(r"\textbf{Split at 1990}", font_size=28, color=ManimColor(GOLD_C)).to_edge(UP, buff=0.3)
        self.play(Transform(title2, break_text))

        bp_line  = DashedLine(axes2.c2p(bp_co2, -0.5), axes2.c2p(bp_co2, 1.5), color=ManimColor(GOLD_C), stroke_width=2)
        bp_label = MathTex(
            rf"1990:\ {bp_co2:.1f}\,\text{{ppm}}",
            font_size=20, color=ManimColor(GOLD_C)
        ).next_to(bp_line, UP, buff=0.1)

        self.play(Create(bp_line), Write(bp_label))
        self.wait(1)

        # ── SCENE 4: Piecewise fit — quad + exp ───────────────────────────────
        win_label = Tex(r"\textbf{Best fit per segment:}", font_size=26, color=ManimColor(GREEN_C)).to_corner(UL).shift(DOWN * 0.5)

        # Split linear line at breakpoint
        line_lin_1 = axes2.plot(lambda x: linear_func(x, *popt_lin), x_range=[co2.min(), bp_co2], color=WHITE)
        line_lin_2 = axes2.plot(lambda x: linear_func(x, *popt_lin), x_range=[bp_co2, co2.max()], color=WHITE)
        self.add(line_lin_1, line_lin_2)
        self.remove(line_lin2)

        # Target piecewise curves
        s1_curve = axes2.plot(lambda x: quadratic_func(x, *popt_s1), x_range=[co2.min(), bp_co2], color=ManimColor(BLUE_C), stroke_width=3)
        s2_curve = axes2.plot(lambda x: exponential_func(x, *popt_s2), x_range=[bp_co2, co2.max()], color=ManimColor(ORANGE_C), stroke_width=3)

        a, b, c = popt_s1
        eq1 = MathTex(
            rf"\hat{{y}} = {a:.5f}x^2 {'+' if b >= 0 else '-'} {abs(b):.4f}x {'+' if c >= 0 else '-'} {abs(c):.4f}",
            font_size=18, color=ManimColor(BLUE_C)
        ).move_to(axes2.c2p(330, 1.25))

        a2, b2, c2, x0 = popt_s2
        eq2 = MathTex(
            rf"\hat{{y}} = {a2:.2e}\,e^{{{b2:.4f}(x - {x0:.1f})}} {'+' if c2 >= 0 else '-'} {abs(c2):.4f}",
            font_size=18, color=ManimColor(ORANGE_C)
        ).move_to(axes2.c2p(385, 0.25))

        self.play(FadeIn(win_label))
        self.play(
            Transform(line_lin_1, s1_curve),
            Transform(line_lin_2, s2_curve),
            r2_tracker2.animate.set_value(total_r2),
            run_time=3
        )
        self.play(Write(eq1), Write(eq2))
        self.wait(2)

        # ── SCENE 5: R² comparison callout card ───────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        compare_card = VGroup(
            MathTex(r"R^2_{\text{global}} = " + f"{r2_lin:.4f}", font_size=48, color=WHITE),
            MathTex(r"\Downarrow \;\text{split at 1990}", font_size=32, color=ManimColor(GOLD_C)),
            MathTex(r"R^2_{\text{piecewise}} = " + f"{total_r2:.4f}", font_size=48, color=ManimColor(GREEN_C)),
            MathTex(r"\Delta R^2 = +" + f"{total_r2 - r2_lin:.4f}", font_size=52, color=ManimColor(GOLD_C)),
        ).arrange(DOWN, buff=0.5).move_to(ORIGIN)

        self.play(Write(compare_card[0]))
        self.wait(0.5)
        self.play(Write(compare_card[1]), Write(compare_card[2]))
        self.wait(0.5)
        self.play(Write(compare_card[3]))
        self.wait(4)
