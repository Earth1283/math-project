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
        
        ax_co2 = Axes(
            x_range=[1950, 2030, 20], 
            y_range=[300, 420, 40], 
            x_length=5, 
            y_length=3,
            axis_config={"include_numbers": True, "label_constructor": Text}
        ).to_edge(LEFT, buff=0.5)
        
        # Adding formal labels
        ax_co2_labels = ax_co2.get_axis_labels(
            x_label=Text("Year", font_size=14), 
            y_label=Text("ppm", font_size=14)
        )
        
        ax_temp = Axes(
            x_range=[1950, 2030, 20], 
            y_range=[-0.5, 1.5, 0.5], 
            x_length=5, 
            y_length=3,
            axis_config={"include_numbers": True, "label_constructor": Text}
        ).to_edge(RIGHT, buff=0.5)
        
        ax_temp_labels = ax_temp.get_axis_labels(
            x_label=Text("Year", font_size=14), 
            y_label=Text("C", font_size=14)
        )
        
        label_co2 = Text("CO2 Concentration", font_size=20, color=BLUE).next_to(ax_co2, UP)
        label_temp = Text("Temperature Change", font_size=20, color=ORANGE).next_to(ax_temp, UP)
        
        # Use plot for simple lines
        line_co2 = ax_co2.plot_line_graph(X, co2_raw, add_vertex_dots=False, line_color=BLUE)
        line_temp = ax_temp.plot_line_graph(X, temp_raw, add_vertex_dots=False, line_color=ORANGE)
        
        # Calculate squashed temp line on ax_co2
        # FIX: Add 300 to the temp values so they appear at the bottom of the CO2 axis (300-420)
        # instead of hundreds of units off-screen.
        line_temp_squashed = ax_co2.plot_line_graph(X, 300 + temp_raw, add_vertex_dots=False, line_color=ORANGE)
        
        self.play(
            Create(ax_co2), Create(ax_temp), 
            FadeIn(ax_co2_labels), FadeIn(ax_temp_labels),
            FadeIn(label_co2), FadeIn(label_temp)
        )
        self.play(Create(line_co2), Create(line_temp))
        self.wait(1)
        
        # Attempt to Merge (The Awkward Part)
        self.play(
            FadeOut(ax_temp),
            FadeOut(ax_temp_labels),
            FadeOut(label_temp),
            ReplacementTransform(line_temp, line_temp_squashed),
            line_co2.animate.set_stroke(opacity=0.3)
        )
        
        # Use simple text to avoid encoding issues with emojis in some environments
        awkward_text = Text("well that's awkward...", color=YELLOW, font_size=28).next_to(ax_co2, DOWN, buff=0.5)
        self.play(Write(awkward_text))
        self.wait(2)
        
        # FIX: Ensure ALL Scene 1 elements are faded out, including labels
        self.play(FadeOut(VGroup(
            ax_co2, ax_co2_labels, 
            line_co2, line_temp_squashed, 
            label_co2, awkward_text, title
        )))

        # 3. Scene 2: Standardizing CO2
        math_title = Text("Standardization: Z-Score", font_size=32).to_edge(UP)
        
        # formula components
        z_prefix = Text("z", font_size=36)
        equals = Text(" = ", font_size=36)
        x_val = Text("x", font_size=36)
        minus = Text(" - ", font_size=36)
        mu = Text("\u03bc", font_size=36, color=YELLOW)
        num = VGroup(x_val, minus, mu).arrange(RIGHT, buff=0.1)
        sigma = Text("\u03c3", font_size=36, color=GREEN)
        frac_line = Line(LEFT, RIGHT, color=WHITE).set_width(num.width * 1.1)
        
        frac = VGroup(num, frac_line, sigma).arrange(DOWN, buff=0.1)
        formula = VGroup(z_prefix, equals, frac).arrange(RIGHT, buff=0.2).shift(UP * 1.5)
        
        self.play(Write(math_title))
        self.play(Write(formula))
        self.wait(1)

        # Explain Formula
        mu_desc = Text("\u03bc : Mean", font_size=24, color=YELLOW).next_to(formula, RIGHT, buff=1.0).shift(UP * 0.3)
        sigma_desc = Text("\u03c3 : Std Dev", font_size=24, color=GREEN).next_to(mu_desc, DOWN, aligned_edge=LEFT)
        
        self.play(Indicate(mu), Write(mu_desc))
        self.wait(0.5)
        self.play(Indicate(sigma), Write(sigma_desc))
        self.wait(1)
        
        # Standard Axes (Z-score scale)
        ax_z = Axes(
            x_range=[1950, 2030, 20], 
            y_range=[-3, 3, 1], 
            x_length=8, 
            y_length=4,
            axis_config={"include_numbers": True, "label_constructor": Text}
        ).shift(DOWN * 0.5)
        
        # Data logic - using noisy data for accuracy
        co2_mean = np.mean(co2_raw)
        co2_std = np.std(co2_raw)
        
        # Hidden axis to keep centered-but-unscaled data visible
        ax_hidden = Axes(
            x_range=[1950, 2030, 20], 
            y_range=[-60, 60, 20], 
            x_length=8, 
            y_length=4
        ).shift(DOWN * 0.5)
        
        # Start with uncentered line
        co2_line_offset = ax_hidden.plot_line_graph(
            X, 
            (co2_raw - co2_mean) + 30, 
            add_vertex_dots=False, 
            line_color=BLUE
        )
        
        self.play(Create(ax_z), Create(co2_line_offset))
        self.wait(1)
        
        # 1. Shift
        shift_text = Text("1. Shift (Subtract Mean)", font_size=24, color=YELLOW).to_edge(DL, buff=1.0).shift(UP * 1.5)
        self.play(Write(shift_text), Indicate(mu))
        # Transform to centered line
        co2_line_centered = ax_hidden.plot_line_graph(
            X, 
            (co2_raw - co2_mean), 
            add_vertex_dots=False, 
            line_color=BLUE
        )
        self.play(Transform(co2_line_offset, co2_line_centered))
        self.wait(1)
        
        # 2. Scale
        scale_text = Text("2. Scale (Divide by Std Dev)", font_size=24, color=GREEN).next_to(shift_text, UP, aligned_edge=LEFT)
        self.play(Write(scale_text), Indicate(sigma))
        # Transform to final Z-score line
        co2_z_line = ax_z.plot_line_graph(
            X, 
            (co2_raw - co2_mean) / co2_std, 
            add_vertex_dots=False, 
            line_color=BLUE
        )
        self.play(Transform(co2_line_offset, co2_z_line))
        self.wait(2)
        
        self.play(FadeOut(VGroup(shift_text, scale_text, mu_desc, sigma_desc)))

        # 4. Scene 3: Final Comparison
        revelation_title = Text("The Hidden Correlation", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Transform(math_title, revelation_title))
        
        # Standardize Temp line
        temp_mean = np.mean(temp_raw)
        temp_std = np.std(temp_raw)
        temp_z_line = ax_z.plot_line_graph(X, (temp_raw - temp_mean) / temp_std, add_vertex_dots=False, line_color=RED)
        
        label_z_co2 = Text("CO2 (Standardized)", font_size=18, color=BLUE).next_to(ax_z, UP, buff=0.2).shift(LEFT * 2)
        label_z_temp = Text("Temp (Standardized)", font_size=18, color=RED).next_to(ax_z, UP, buff=0.2).shift(RIGHT * 2)

        self.play(Create(temp_z_line), FadeIn(label_z_co2), FadeIn(label_z_temp))
        self.wait(1)
        
        conclusion = Text("Trends become comparable once scales are unified.", font_size=24).to_edge(DOWN, buff=0.5)
        self.play(Write(conclusion))
        self.wait(5)

