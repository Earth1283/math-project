from manim import *
import numpy as np
from co2_prediction import load_csv, linear_func, exponential_func, rational_2_1, fit_models

config.tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")

BLUE_C   = "#0072B2"
ORANGE_C = "#D55E00"
GREEN_C  = "#009E73"
RED_C    = "#CC79A7"
GRAY_C   = "#999999"
GOLD_C   = "#E69F00"

class CO2PredictionExplainer(Scene):
    def construct(self):

        # ── Load data and fit models ──────────────────────────────────────────
        data = load_csv('data/co2-ppm.csv')
        years_hist, co2_hist = data[:, 0], data[:, 1]

        _, p_lin, _ = fit_models(years_hist, co2_hist, "linear")

        mask2   = years_hist > 1990
        _, p_exp, _ = fit_models(years_hist[mask2], co2_hist[mask2], "exponential")
        _, p_rat, _ = fit_models(years_hist[mask2], co2_hist[mask2], "rational_2_1")

        # ── SCENE 0: Opener ───────────────────────────────────────────────────
        opener_title = MathTex(r"\text{Forecasting the CO}_2\text{ Future}", font_size=52).shift(UP * 0.7)
        opener_sub   = Tex(
            r"3 models · Historical data 1959--2020 · Projecting to 2050 and beyond",
            font_size=30
        ).next_to(opener_title, DOWN, buff=0.5)
        opener_sub2  = Tex(
            r"Which model best captures the acceleration of emissions?",
            font_size=26, color=ManimColor(GRAY_C)
        ).next_to(opener_sub, DOWN, buff=0.3)

        self.play(Write(opener_title))
        self.play(FadeIn(opener_sub))
        self.play(FadeIn(opener_sub2))
        self.wait(2)
        self.play(FadeOut(VGroup(opener_title, opener_sub, opener_sub2)))

        # ── SCENE 1: Axes + historical data ──────────────────────────────────
        axes = Axes(
            x_range=[1950, 2060, 20],
            y_range=[300, 750, 100],
            x_length=10, y_length=5.5,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 20},
            tips=False
        ).shift(DOWN * 0.5)

        labels = axes.get_axis_labels(
            x_label=Text("Year", font_size=20),
            y_label=MathTex(r"\text{CO}_2\text{ (ppm)}", font_size=20)
        )

        title = MathTex(r"\text{Forecasting the CO}_2\text{ Future}", font_size=36).to_edge(UP, buff=0.3)

        dots = always_redraw(lambda: VGroup(*[
            Dot(axes.c2p(x, y), color=ManimColor(GRAY_C), radius=0.035, fill_opacity=0.5)
            for x, y in zip(years_hist, co2_hist)
        ]))

        self.play(Write(title), Create(axes), FadeIn(labels))
        self.play(FadeIn(dots))
        self.wait(0.8)

        # ── SCENE 2: Three growing model lines ───────────────────────────────
        time_tracker = ValueTracker(2023)

        line_lin = always_redraw(lambda: axes.plot(
            lambda x: linear_func(x, *p_lin),
            x_range=[1959, time_tracker.get_value()],
            color=ManimColor(BLUE_C), stroke_width=3
        ))
        line_exp = always_redraw(lambda: axes.plot(
            lambda x: exponential_func(x, *p_exp),
            x_range=[1990, time_tracker.get_value()],
            color=ManimColor(ORANGE_C), stroke_width=3
        ))
        line_rat = always_redraw(lambda: axes.plot(
            lambda x: rational_2_1(x, *p_rat),
            x_range=[1990, time_tracker.get_value()],
            color=ManimColor(GREEN_C), stroke_width=3
        ))

        legend = VGroup(
            VGroup(
                Line(ORIGIN, RIGHT * 0.55, color=ManimColor(BLUE_C), stroke_width=3),
                Tex(r"Global Linear", font_size=20, color=ManimColor(BLUE_C))
            ).arrange(RIGHT, buff=0.25),
            VGroup(
                Line(ORIGIN, RIGHT * 0.55, color=ManimColor(ORANGE_C), stroke_width=3),
                Tex(r"Post-1990 Exponential", font_size=20, color=ManimColor(ORANGE_C))
            ).arrange(RIGHT, buff=0.25),
            VGroup(
                Line(ORIGIN, RIGHT * 0.55, color=ManimColor(GREEN_C), stroke_width=3),
                Tex(r"Post-1990 Rational", font_size=20, color=ManimColor(GREEN_C))
            ).arrange(RIGHT, buff=0.25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).to_corner(UL, buff=0.9)

        date_text = always_redraw(lambda: MathTex(
            r"\text{Year: } " + str(int(time_tracker.get_value())),
            font_size=30, color=YELLOW
        ).to_corner(UR, buff=1.0))

        self.add(line_lin, line_exp, line_rat)
        self.play(FadeIn(legend), Write(date_text))
        self.play(time_tracker.animate.set_value(2050), run_time=5, rate_func=linear)
        self.wait(2)

        # 2050 projections panel
        val_lin_2050 = linear_func(2050, *p_lin)
        val_exp_2050 = exponential_func(2050, *p_exp)
        val_rat_2050 = rational_2_1(2050, *p_rat)

        label_2050 = VGroup(
            Tex(r"\textbf{2050 Projections}", font_size=26),
            MathTex(r"\text{Linear: } " + f"{val_lin_2050:.1f}" + r"\,\text{ppm}", font_size=22, color=ManimColor(BLUE_C)),
            MathTex(r"\text{Exponential: } " + f"{val_exp_2050:.1f}" + r"\,\text{ppm}", font_size=22, color=ManimColor(ORANGE_C)),
            MathTex(r"\text{Rational: } " + f"{val_rat_2050:.1f}" + r"\,\text{ppm}", font_size=22, color=ManimColor(GREEN_C)),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_corner(DR, buff=1.0)

        self.play(FadeIn(label_2050))
        self.wait(3)
        self.play(FadeOut(label_2050))

        # ── SCENE 3: Threshold callout card ──────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        thresh_card = VGroup(
            MathTex(r"685\,\text{ppm}", font_size=80, color=ManimColor(RED_C)),
            Tex(r"$\approx 2.4\times$ pre-industrial CO$_2$ (280 ppm)", font_size=34),
            Tex(r"A benchmark for severe long-term climate impact.", font_size=28, color=ManimColor(GRAY_C)),
        ).arrange(DOWN, buff=0.5).move_to(ORIGIN)

        self.play(Write(thresh_card[0]))
        self.play(FadeIn(thresh_card[1]), FadeIn(thresh_card[2]))
        self.wait(3)
        self.play(FadeOut(thresh_card))

        # ── SCENE 4: Extended axes to 685 ppm ────────────────────────────────
        new_axes = Axes(
            x_range=[1950, 2220, 40],
            y_range=[300, 750, 100],
            x_length=10, y_length=5.5,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 20},
            tips=False
        ).shift(DOWN * 0.5)

        new_labels = new_axes.get_axis_labels(
            x_label=Text("Year", font_size=20),
            y_label=MathTex(r"\text{CO}_2\text{ (ppm)}", font_size=20)
        )

        title2 = MathTex(r"\text{Forecasting the CO}_2\text{ Future}", font_size=36).to_edge(UP, buff=0.3)

        self.play(Create(new_axes), FadeIn(new_labels), Write(title2))

        dots2 = always_redraw(lambda: VGroup(*[
            Dot(new_axes.c2p(x, y), color=ManimColor(GRAY_C), radius=0.03, fill_opacity=0.4)
            for x, y in zip(years_hist, co2_hist)
        ]))

        threshold_line  = DashedLine(
            new_axes.c2p(1950, 685), new_axes.c2p(2220, 685),
            color=ManimColor(RED_C), stroke_width=2
        )
        threshold_label = MathTex(
            r"685\,\text{ppm threshold}", font_size=20, color=ManimColor(RED_C)
        ).next_to(threshold_line, UP, buff=0.15).to_edge(RIGHT, buff=0.6)

        target_years = {"lin": 2194.9, "exp": 2077.8, "rat": 2079.3}
        max_target   = max(target_years.values())

        line_lin2 = always_redraw(lambda: new_axes.plot(
            lambda x: linear_func(x, *p_lin),
            x_range=[1959, time_tracker.get_value()],
            color=ManimColor(BLUE_C), stroke_width=3
        ))
        line_exp2 = always_redraw(lambda: new_axes.plot(
            lambda x: exponential_func(x, *p_exp),
            x_range=[1990, time_tracker.get_value()],
            color=ManimColor(ORANGE_C), stroke_width=3
        ))
        line_rat2 = always_redraw(lambda: new_axes.plot(
            lambda x: rational_2_1(x, *p_rat),
            x_range=[1990, time_tracker.get_value()],
            color=ManimColor(GREEN_C), stroke_width=3
        ))

        legend2 = VGroup(
            VGroup(Line(ORIGIN, RIGHT*0.55, color=ManimColor(BLUE_C),   stroke_width=3), Tex(r"Global Linear",         font_size=18, color=ManimColor(BLUE_C))).arrange(RIGHT, buff=0.2),
            VGroup(Line(ORIGIN, RIGHT*0.55, color=ManimColor(ORANGE_C), stroke_width=3), Tex(r"Post-1990 Exponential", font_size=18, color=ManimColor(ORANGE_C))).arrange(RIGHT, buff=0.2),
            VGroup(Line(ORIGIN, RIGHT*0.55, color=ManimColor(GREEN_C),  stroke_width=3), Tex(r"Post-1990 Rational",    font_size=18, color=ManimColor(GREEN_C))).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_corner(UL, buff=0.9)

        date_text2 = always_redraw(lambda: MathTex(
            r"\text{Year: } " + str(int(time_tracker.get_value())),
            font_size=28, color=YELLOW
        ).to_corner(UR, buff=1.0))

        time_tracker.set_value(2050)
        self.add(dots2, line_lin2, line_exp2, line_rat2)
        self.play(FadeIn(legend2), Write(date_text2), Write(threshold_line), Write(threshold_label))
        self.wait(1)

        # Animate to 685 ppm crossing
        self.play(time_tracker.animate.set_value(max_target), run_time=8, rate_func=linear)
        self.wait(1)

        # ── SCENE 5: Crossing years callout card ──────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        cross_card = VGroup(
            Tex(r"\textbf{685 ppm Crossing Years}", font_size=44),
            MathTex(r"\text{Global Linear:} \quad " + f"{int(target_years['lin'])}",         font_size=36, color=ManimColor(BLUE_C)),
            MathTex(r"\text{Post-1990 Exponential:} \quad " + f"{int(target_years['exp'])}", font_size=36, color=ManimColor(ORANGE_C)),
            MathTex(r"\text{Post-1990 Rational:} \quad " + f"{int(target_years['rat'])}",   font_size=36, color=ManimColor(GREEN_C)),
        ).arrange(DOWN, buff=0.5).move_to(ORIGIN)

        self.play(Write(cross_card[0]))
        self.wait(0.3)
        self.play(Write(cross_card[1]))
        self.play(Write(cross_card[2]))
        self.play(Write(cross_card[3]))
        self.wait(5)
