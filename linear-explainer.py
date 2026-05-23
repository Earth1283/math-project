from manim import *
import numpy as np
from sklearn.linear_model import LinearRegression
from linear import load_and_align_data

config.tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")

BLUE_C   = "#0072B2"
ORANGE_C = "#D55E00"
GREEN_C  = "#009E73"
RED_C    = "#CC79A7"
GRAY_C   = "#999999"
GOLD_C   = "#E69F00"

class LinearRegressionExplainer(Scene):
    def construct(self):

        # ── SCENE 0: Opener ──────────────────────────────────────────────────
        opener_title = Tex(r"\textbf{Linear Regression}", font_size=56).shift(UP * 0.9)
        opener_sub1  = MathTex(r"\text{CO}_2 \text{ vs.\ Temperature Change}", font_size=38).next_to(opener_title, DOWN, buff=0.4)
        opener_sub2  = Tex(r"1959--2020 · 62 annual measurements", font_size=28, color=ManimColor(GRAY_C)).next_to(opener_sub1, DOWN, buff=0.3)
        self.play(Write(opener_title))
        self.play(FadeIn(opener_sub1))
        self.play(FadeIn(opener_sub2))
        self.wait(2)
        self.play(FadeOut(VGroup(opener_title, opener_sub1, opener_sub2)))

        # ── Load REAL data ────────────────────────────────────────────────────
        try:
            years, co2_vals, temp_vals = load_and_align_data(
                'data/co2-ppm.csv', 'data/surface-air-temp-change.csv'
            )
            step = max(1, len(years) // 30)
            years_sampled = years[::step]
            X = co2_vals[::step]
            y = temp_vals[::step]

            model = LinearRegression()
            model.fit(X.reshape(-1, 1), y)
            m_opt = model.coef_[0]
            b_opt = model.intercept_
        except Exception:
            years_sampled = np.linspace(1959, 2020, 30)
            X = np.linspace(315, 415, 30)
            y = 0.01 * X - 3.2
            m_opt, b_opt = 0.01, -3.2

        y_pred_opt  = m_opt * X + b_opt
        ss_res_val  = float(np.sum((y - y_pred_opt) ** 2))
        ss_tot_val  = float(np.sum((y - np.mean(y)) ** 2))
        r2_val      = 1 - ss_res_val / ss_tot_val

        # ── SCENE 1: Data landscape ───────────────────────────────────────────
        axes = Axes(
            x_range=[310, 420, 20],
            y_range=[-0.5, 1.5, 0.5],
            x_length=7.5, y_length=4.5,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 20},
            tips=False
        ).to_edge(LEFT, buff=0.6).shift(DOWN * 0.2)

        co2_x_label = VGroup(
            MathTex(r"\text{CO}_2", font_size=24),
            Text(" Concentration (ppm)", font_size=18)
        ).arrange(RIGHT, buff=0.1)
        x_label = axes.get_x_axis_label(co2_x_label).shift(UP * 0.2)
        y_label = axes.get_y_axis_label(Text("Temp Change (°C)", font_size=18)).shift(RIGHT * 0.3)

        # Year-gradient dots: early years = blue, recent = orange
        year_min, year_max = years_sampled.min(), years_sampled.max()
        dots = VGroup(*[
            Dot(
                axes.c2p(x_val, y_val),
                color=interpolate_color(ManimColor(BLUE_C), ManimColor(ORANGE_C),
                                        (yr - year_min) / (year_max - year_min)),
                radius=0.07
            )
            for x_val, y_val, yr in zip(X, y, years_sampled)
        ])

        dot_legend = VGroup(
            VGroup(Dot(color=ManimColor(BLUE_C),   radius=0.07), Tex(r"1959", font_size=18)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(color=ManimColor(ORANGE_C), radius=0.07), Tex(r"2020", font_size=18)).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.6).next_to(axes, DOWN, buff=0.25)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
        self.play(FadeIn(dots, lag_ratio=0.05))
        self.play(FadeIn(dot_legend))
        self.wait(1.5)

        # ── SCENE 2: Equation + guess line + residuals ────────────────────────
        equation = MathTex(r"\hat{y} = mx + b", font_size=48).to_edge(RIGHT, buff=0.8).shift(UP * 1.8)
        m_desc = MathTex(r"m \text{ — slope}", font_size=30, color=YELLOW)
        b_desc = MathTex(r"b \text{ — intercept}", font_size=30, color=GREEN)
        eq_notes = VGroup(m_desc, b_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(equation, DOWN, buff=0.5).align_to(equation, LEFT)

        m_guess, b_guess = 0.005, -1.5
        guess_line = axes.plot(lambda x: m_guess * x + b_guess, color=YELLOW, x_range=[310, 420])

        self.play(Write(equation))
        self.play(Write(eq_notes))
        self.wait(0.5)
        self.play(Create(guess_line))
        self.wait(0.5)

        # Residuals
        residuals_lines = VGroup(*[
            DashedLine(
                start=dots[i].get_center(),
                end=axes.c2p(X[i], m_guess * X[i] + b_guess),
                color=ManimColor(RED_C), stroke_width=1.5
            )
            for i in range(len(X))
        ])
        residual_text = MathTex(
            r"\text{Residuals} = y_i - \hat{y}_i",
            font_size=30, color=ManimColor(RED_C)
        ).next_to(eq_notes, DOWN, buff=0.5).align_to(equation, LEFT)

        self.play(Create(residuals_lines, lag_ratio=0.02), Write(residual_text))
        self.wait(2)

        # ── SCENE 3: Optimal fit via sklearn ──────────────────────────────────
        code_snippet = "model = LinearRegression()\nmodel.fit(X, y)"
        code = Code(
            code_string=code_snippet,
            language="python",
            formatter_style="monokai",
            add_line_numbers=False,
            background="window",
        ).scale(0.55).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

        opt_text = Tex(r"Finding optimal fit\ldots", font_size=26, color=GREEN).next_to(code, DOWN, buff=0.4).align_to(code, LEFT)

        optimal_line = axes.plot(lambda x: m_opt * x + b_opt, color=ManimColor(GREEN_C), x_range=[310, 420])

        self.play(
            FadeOut(equation), FadeOut(eq_notes), FadeOut(residual_text),
            FadeIn(code)
        )
        self.play(Write(opt_text))
        self.wait(0.5)
        self.play(
            Transform(guess_line, optimal_line),
            FadeOut(residuals_lines),
            run_time=2.5
        )
        self.wait(1)
        self.play(FadeOut(opt_text))

        # ── SCENE 4: SS_res derivation ────────────────────────────────────────
        ss_label    = Tex(r"Sum of Squared Residuals:", font_size=28, color=YELLOW).to_edge(RIGHT, buff=0.8).shift(UP * 2.0)
        ss_eq_start = MathTex(
            r"SS_{\text{res}} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
            font_size=30, color=YELLOW
        ).next_to(ss_label, DOWN, buff=0.3).align_to(ss_label, LEFT)
        ss_eq_final = MathTex(
            rf"SS_{{\text{{res}}}} = {ss_res_val:.4f}",
            font_size=30, color=YELLOW
        ).move_to(ss_eq_start, aligned_edge=LEFT)

        self.play(FadeOut(code))
        self.play(Write(ss_label), Write(ss_eq_start))
        self.wait(1.5)
        self.play(Transform(ss_eq_start, ss_eq_final))
        self.wait(1.5)

        # R² derivation
        r2_label   = Tex(r"Coefficient of Determination:", font_size=28, color=ManimColor(GREEN_C)).next_to(ss_eq_start, DOWN, buff=0.7).align_to(ss_label, LEFT)
        r2_stage_a = MathTex(
            r"R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}",
            font_size=32, color=ManimColor(GREEN_C)
        ).next_to(r2_label, DOWN, buff=0.35).align_to(ss_label, LEFT)
        r2_stage_b = MathTex(
            rf"R^2 = 1 - \frac{{{ss_res_val:.2f}}}{{{ss_tot_val:.2f}}}",
            font_size=32, color=ManimColor(GREEN_C)
        ).move_to(r2_stage_a, aligned_edge=LEFT)
        r2_stage_c = MathTex(
            r"R^2 = " + f"{r2_val:.4f}",
            font_size=32, color=ManimColor(GREEN_C)
        ).move_to(r2_stage_a, aligned_edge=LEFT)

        self.play(Write(r2_label))
        self.play(Write(r2_stage_a))
        self.wait(1.5)
        self.play(Transform(r2_stage_a, r2_stage_b))
        self.wait(1.5)
        self.play(Transform(r2_stage_a, r2_stage_c))
        self.wait(2)

        # ── SCENE 5: R² callout card ──────────────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        r2_card = MathTex(
            r"R^2 = " + f"{r2_val:.4f}",
            font_size=80, color=ManimColor(GREEN_C)
        ).shift(UP * 1.0)
        interp = Tex(
            r"CO$_2$ explains " + f"{r2_val * 100:.1f}" + r"\% of temperature variance.",
            font_size=36
        ).next_to(r2_card, DOWN, buff=0.6)
        slope_line = Tex(
            r"Slope $= " + f"{m_opt:.4f}" + r"\,°\text{C/ppm}$",
            font_size=30, color=YELLOW
        ).next_to(interp, DOWN, buff=0.35)
        slope_note = Tex(
            r"Each 1 ppm $\uparrow$ in CO$_2$ $\Rightarrow$ " + f"{m_opt:.4f}" + r"°C $\uparrow$ in temperature.",
            font_size=26, color=ManimColor(GRAY_C)
        ).next_to(slope_line, DOWN, buff=0.25)

        self.play(Write(r2_card))
        self.play(FadeIn(interp))
        self.play(FadeIn(slope_line))
        self.play(FadeIn(slope_note))
        self.wait(5)
