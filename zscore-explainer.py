from manim import *
import numpy as np

class ZScoreExplainer(Scene):
    def construct(self):
        # 1. Setup Data
        np.random.seed(42)
        X = np.linspace(1959, 2020, 60)
        co2_raw = 315 + 1.6 * (X - 1959) + np.random.normal(0, 1.5, 60)
        temp_raw = 0.1 * (X - 1959)/10 + np.random.normal(0, 0.1, 60)

        # 2. Scene 1: Mismatched Scales
        title = Text("The Scaling Problem", font_size=32).to_edge(UP)
        self.play(Write(title))
        
        ax_co2 = Axes(x_range=[1950, 2030, 10], y_range=[300, 420, 20], x_length=5, y_length=3).to_edge(LEFT, buff=0.5)
        ax_temp = Axes(x_range=[1950, 2030, 10], y_range=[-0.5, 1.5, 0.5], x_length=5, y_length=3).to_edge(RIGHT, buff=0.5)
        
        label_co2 = Text("CO2 (ppm)", font_size=20, color=BLUE).next_to(ax_co2, UP)
        label_temp = Text("Temp (C)", font_size=20, color=RED).next_to(ax_temp, UP)
        
        # Use plot for simple lines
        line_co2 = ax_co2.plot_line_graph(X, co2_raw, add_vertex_dots=False, line_color=BLUE)
        line_temp = ax_temp.plot_line_graph(X, temp_raw, add_vertex_dots=False, line_color=RED)
        
        self.play(Create(ax_co2), Create(ax_temp), FadeIn(label_co2), FadeIn(label_temp))
        self.play(Create(line_co2), Create(line_temp))
        self.wait(1)
        
        # Attempt to Merge (The Awkward Part)
        self.play(
            ax_temp.animate.move_to(ax_co2),
            label_temp.animate.move_to(label_co2),
            line_temp.animate.move_to(line_co2),
            FadeOut(ax_temp),
            FadeOut(label_temp)
        )
        
        awkward_text = Text("well that's awkward 😅", color=YELLOW, font_size=28).next_to(ax_co2, DOWN, buff=0.5)
        self.play(Write(awkward_text))
        self.wait(2)
        self.play(FadeOut(VGroup(ax_co2, line_co2, line_temp, label_co2, awkward_text, title)))

        # 3. Scene 2: Standardizing CO2
        math_title = Text("Standardization: Z-Score", font_size=32).to_edge(UP)
        
        # formula: z = (x - mu) / sigma
        z_prefix = Text("z = ", font_size=36)
        num = Text("x - \u03bc", font_size=36)
        den = Text("\u03c3", font_size=36)
        frac_line = Line(LEFT, RIGHT, color=WHITE).set_width(num.width * 1.1)
        
        # Center fraction parts relative to each other
        frac = VGroup(num, frac_line, den).arrange(DOWN, buff=0.1)
        formula = VGroup(z_prefix, frac).arrange(RIGHT, buff=0.2).shift(UP * 1.5)
        
        self.play(Write(math_title), Write(formula))
        self.wait(1)
        
        # Standard Axes (Z-score scale)
        ax_z = Axes(
            x_range=[1950, 2030, 10], 
            y_range=[-3, 3, 1], 
            x_length=8, 
            y_length=4,
            axis_config={"include_tip": True}
        ).shift(DOWN * 0.5)
        self.play(Create(ax_z))
        
        # Data logic - using noisy data for accuracy
        co2_mean = np.mean(co2_raw)
        co2_std = np.std(co2_raw)
        
        # Initial line: relative to its mean but offset by a visible amount (Issue 1)
        # We use the actual noisy data for consistency (Issue 2)
        offset = 1.5
        co2_line_raw = ax_z.plot_line_graph(
            X, 
            (co2_raw - co2_mean) + offset, 
            add_vertex_dots=False, 
            line_color=BLUE
        )
        
        # 1. Shift
        shift_text = Text("1. Shift (Subtract Mean)", font_size=24, color=BLUE).to_edge(DL, buff=1.0)
        self.play(Write(shift_text))
        # Transform to 0 offset (relative to its mean)
        co2_line_centered = ax_z.plot_line_graph(
            X, 
            (co2_raw - co2_mean), 
            add_vertex_dots=False, 
            line_color=BLUE
        )
        self.play(Transform(co2_line_raw, co2_line_centered))
        self.wait(1)
        
        # 2. Scale
        scale_text = Text("2. Scale (Divide by Std Dev)", font_size=24, color=BLUE).next_to(shift_text, DOWN, aligned_edge=LEFT)
        self.play(Write(scale_text))
        # Divide by standard deviation
        co2_z_line = ax_z.plot_line_graph(
            X, 
            (co2_raw - co2_mean) / co2_std, 
            add_vertex_dots=False, 
            line_color=BLUE
        )
        self.play(Transform(co2_line_raw, co2_z_line))
        self.wait(2)

