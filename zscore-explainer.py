from manim import *
import numpy as np
from scipy.stats import pearsonr
from linear import load_and_align_data

config.tex_template = TexTemplate(tex_compiler="xelatex", output_format=".xdv")

BLUE_C   = "#0072B2"
ORANGE_C = "#D55E00"
GREEN_C  = "#009E73"
RED_C    = "#CC79A7"
GRAY_C   = "#999999"
GOLD_C   = "#E69F00"

class ZScoreExplainer(Scene):
    def construct(self):
        # --- Load REAL data ---
        years, co2_raw, temp_raw = load_and_align_data(
            'data/co2-ppm.csv', 'data/surface-air-temp-change.csv'
        )
        X = years
        r_val, _ = pearsonr(co2_raw, temp_raw)

        # ── SCENE 0: Opener ──────────────────────────────────────────────────
        opener_title = Tex(r"\textbf{The Scaling Problem}", font_size=56).shift(UP * 0.6)
        opener_sub = Tex(
            r"CO$_2$ (ppm) and $\Delta T$ (°C) live on incompatible scales.",
            font_size=32
        ).next_to(opener_title, DOWN, buff=0.5)
        self.play(Write(opener_title))
        self.play(FadeIn(opener_sub))
        self.wait(2)
        self.play(FadeOut(opener_title), FadeOut(opener_sub))

        # ── SCENE 1: Two axes — mismatched scales ────────────────────────────
        ax_co2 = Axes(
            x_range=[1950, 2030, 20],
            y_range=[300, 420, 40],
            x_length=5.5, y_length=3.5,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 16},
            tips=False
        ).to_edge(LEFT, buff=0.7).shift(DOWN * 0.3)

        ax_co2_labels = ax_co2.get_axis_labels(
            x_label=Text("Year", font_size=16),
            y_label=Text("ppm", font_size=16)
        )

        ax_temp = Axes(
            x_range=[1950, 2030, 20],
            y_range=[-0.5, 1.5, 0.5],
            x_length=5.5, y_length=3.5,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 16},
            tips=False
        ).to_edge(RIGHT, buff=0.7).shift(DOWN * 0.3)

        ax_temp_labels = ax_temp.get_axis_labels(
            x_label=Text("Year", font_size=16),
            y_label=Text("°C", font_size=16)
        )

        label_co2 = VGroup(
            MathTex(r"\text{CO}_2", font_size=28, color=ManimColor(BLUE_C)),
            Text(" Concentration", font_size=20, color=ManimColor(BLUE_C))
        ).arrange(RIGHT, buff=0.1).next_to(ax_co2, UP, buff=0.25)

        label_temp = Text("Temperature Change", font_size=20, color=ManimColor(ORANGE_C)).next_to(ax_temp, UP, buff=0.25)

        line_co2  = ax_co2.plot_line_graph(X, co2_raw,  add_vertex_dots=False, line_color=ManimColor(BLUE_C),   stroke_width=2.5)
        line_temp = ax_temp.plot_line_graph(X, temp_raw, add_vertex_dots=False, line_color=ManimColor(ORANGE_C), stroke_width=2.5)

        # Squashed temp on CO2 axes — visually illustrates scale mismatch
        line_temp_squashed = ax_co2.plot_line_graph(
            X, 300 + temp_raw, add_vertex_dots=False,
            line_color=ManimColor(ORANGE_C), stroke_width=2.5
        )

        self.play(
            Create(ax_co2), Create(ax_temp),
            FadeIn(ax_co2_labels), FadeIn(ax_temp_labels),
            FadeIn(label_co2), FadeIn(label_temp)
        )
        self.play(Create(line_co2), Create(line_temp))
        self.wait(1.5)

        # Show the scale gap
        scale_note = MathTex(
            r"\Delta T \approx 1°\text{C} \quad\text{vs.}\quad \text{CO}_2 \approx 100\,\text{ppm}",
            font_size=26, color=YELLOW
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(scale_note))
        self.wait(1.5)
        self.play(FadeOut(scale_note))

        # Merge attempt
        self.play(
            FadeOut(ax_temp), FadeOut(ax_temp_labels), FadeOut(label_temp),
            ReplacementTransform(line_temp, line_temp_squashed),
            line_co2.animate.set_stroke(opacity=0.3)
        )
        awkward_text = Text("well that's awkward...", color=YELLOW, font_size=30).next_to(ax_co2, DOWN, buff=0.4)
        self.play(Write(awkward_text))
        self.wait(2)

        self.play(FadeOut(VGroup(
            ax_co2, ax_co2_labels, line_co2, line_temp_squashed, label_co2, awkward_text
        )))

        # ── SCENE 2: Z-score formula — large & centered ──────────────────────
        math_title = Tex(r"\textbf{The Solution: Standardization}", font_size=44).to_edge(UP, buff=0.7)
        self.play(Write(math_title))

        formula = MathTex(r"z = \frac{x - \mu}{\sigma}", font_size=72).shift(UP * 0.3)
        formula.set_color_by_tex(r"\mu", YELLOW)
        formula.set_color_by_tex(r"\sigma", GREEN)
        self.play(Write(formula))
        self.wait(1)

        mu_desc    = MathTex(r"\mu : \text{Mean}", font_size=32, color=YELLOW)
        sigma_desc = MathTex(r"\sigma : \text{Std Dev}", font_size=32, color=GREEN)
        callouts = VGroup(mu_desc, sigma_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(formula, RIGHT, buff=1.2).shift(DOWN * 0.1)

        self.play(Write(mu_desc))
        self.wait(0.5)
        self.play(Write(sigma_desc))
        self.wait(1.5)

        # Slide formula to corner to make room for axes
        self.play(
            formula.animate.scale(0.6).to_corner(UR, buff=0.8),
            FadeOut(callouts), FadeOut(math_title)
        )

        # ── SCENE 3: Standardization animation ───────────────────────────────
        ax_z = Axes(
            x_range=[1950, 2030, 20],
            y_range=[-3, 3, 1],
            x_length=10, y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 16},
            tips=False
        ).shift(DOWN * 0.5)
        ax_z_labels = ax_z.get_axis_labels(
            x_label=Text("Year", font_size=18),
            y_label=Text("Z-score", font_size=18)
        )

        co2_mean = np.mean(co2_raw)
        co2_std  = np.std(co2_raw)
        # Hidden axes for coordinate-mapping the pre-scaled line
        co2_centered_max = np.max(np.abs(co2_raw - co2_mean))
        hidden_y_max = co2_centered_max + 40
        ax_hidden = Axes(
            x_range=[1950, 2030, 20],
            y_range=[-hidden_y_max, hidden_y_max, 20],
            x_length=10, y_length=5
        ).shift(DOWN * 0.5)

        # Start: line positioned above center (not yet standardized)
        co2_line_offset = ax_hidden.plot_line_graph(
            X, (co2_raw - co2_mean) + 30,
            add_vertex_dots=False, line_color=ManimColor(BLUE_C), stroke_width=2.5
        )

        self.play(Create(ax_z), FadeIn(ax_z_labels), Create(co2_line_offset))
        self.wait(0.5)

        # Step 1: Subtract mean
        step1 = Tex(r"\textbf{Step 1:} Subtract the mean", font_size=28, color=YELLOW).to_corner(UL, buff=1.0)
        self.play(Write(step1))
        co2_line_centered = ax_hidden.plot_line_graph(
            X, (co2_raw - co2_mean),
            add_vertex_dots=False, line_color=ManimColor(BLUE_C), stroke_width=2.5
        )
        self.play(Transform(co2_line_offset, co2_line_centered))
        self.wait(1)

        # Step 2: Divide by std dev
        step2 = Tex(r"\textbf{Step 2:} Divide by std dev", font_size=28, color=GREEN).next_to(step1, DOWN, aligned_edge=LEFT, buff=0.3)
        self.play(Write(step2))
        co2_z_line = ax_z.plot_line_graph(
            X, (co2_raw - co2_mean) / co2_std,
            add_vertex_dots=False, line_color=ManimColor(BLUE_C), stroke_width=2.5
        )
        self.play(Transform(co2_line_offset, co2_z_line))
        self.wait(2)
        self.play(FadeOut(step1), FadeOut(step2))

        # ── SCENE 4: Hidden correlation revealed ─────────────────────────────
        reveal_title = Tex(r"\textbf{The Hidden Correlation}", font_size=36, color=YELLOW).to_edge(UP, buff=0.5)
        self.play(Write(reveal_title))

        temp_mean = np.mean(temp_raw)
        temp_std  = np.std(temp_raw)
        temp_z_line = ax_z.plot_line_graph(
            X, (temp_raw - temp_mean) / temp_std,
            add_vertex_dots=False, line_color=ManimColor(ORANGE_C), stroke_width=2.5
        )

        legend_co2  = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=ManimColor(BLUE_C),   stroke_width=3.5),
            MathTex(r"\text{CO}_2\text{ (standardized)}", font_size=20, color=ManimColor(BLUE_C))
        ).arrange(RIGHT, buff=0.2)
        legend_temp = VGroup(
            Line(ORIGIN, RIGHT * 0.5, color=ManimColor(ORANGE_C), stroke_width=3.5),
            Tex(r"Temp (standardized)", font_size=20, color=ManimColor(ORANGE_C))
        ).arrange(RIGHT, buff=0.2)
        legend = VGroup(legend_co2, legend_temp).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_corner(UL, buff=1.0)

        self.play(Create(temp_z_line), FadeIn(legend))
        self.wait(2)

        # ── SCENE 5: Pearson r callout card ──────────────────────────────────
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        r_card  = MathTex(r"r = " + f"{r_val:.4f}", font_size=80, color=ManimColor(GOLD_C)).shift(UP * 0.8)
        r_label = Tex(
            r"Strong positive correlation between CO$_2$ and $\Delta T$",
            font_size=34
        ).next_to(r_card, DOWN, buff=0.6)
        r_sub = Tex(
            r"Standardizing reveals the relationship invisible at mismatched scales.",
            font_size=26, color=ManimColor(GRAY_C)
        ).next_to(r_label, DOWN, buff=0.35)

        self.play(Write(r_card))
        self.play(FadeIn(r_label))
        self.play(FadeIn(r_sub))
        self.wait(5)
