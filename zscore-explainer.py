from manim import *
import numpy as np

class ZScoreExplainer(Scene):
    def construct(self):
        # --- Helpers for Subscripts ---
        def get_subscript(main_text, sub_text, color=WHITE, font_size=24):
            main = Text(main_text, font_size=font_size, color=color)
            sub = Text(sub_text, font_size=font_size * 0.65, color=color)
            sub.next_to(main, DR, buff=0.03).shift(UP * 0.08)
            return VGroup(main, sub)

        def get_superscript(main_text, super_text, color=WHITE, font_size=24):
            main = Text(main_text, font_size=font_size, color=color)
            sup = Text(super_text, font_size=font_size * 0.6, color=color)
            sup.next_to(main, UR, buff=0.02).shift(DOWN * 0.05)
            return VGroup(main, sup)

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
        
        label_co2 = VGroup(
            MathTex(r"CO_2", font_size=30, color=BLUE), 
            Text(" Concentration", font_size=20, color=BLUE)
        ).arrange(RIGHT, buff=0.1).next_to(ax_co2, UP)
        
        label_temp = Text("Temperature Change", font_size=20, color=ORANGE).next_to(ax_temp, UP)
        
        line_co2 = ax_co2.plot_line_graph(X, co2_raw, add_vertex_dots=False, line_color=BLUE)
        line_temp = ax_temp.plot_line_graph(X, temp_raw, add_vertex_dots=False, line_color=ORANGE)
        
        # Calculate squashed temp line on ax_co2
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
        
        awkward_text = Text("well that's awkward...", color=YELLOW, font_size=28).next_to(ax_co2, DOWN, buff=0.5)
        self.play(Write(awkward_text))
        self.wait(2)
        
        # Ensure ALL Scene 1 elements are faded out
        self.play(FadeOut(VGroup(
            ax_co2, ax_co2_labels, 
            line_co2, line_temp_squashed, 
            label_co2, awkward_text, title
        )))

        # 3. Scene 2: Standardizing CO2
        math_title = VGroup(
            Text("Standardization: ", font_size=32), 
            MathTex(r"CO_2", font_size=48)
        ).arrange(RIGHT, buff=0.2).to_edge(UP)
        
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
        
        # Standard Axes (Z-score scale) - ENLARGED for more visibility
        ax_z = Axes(
            x_range=[1950, 2030, 20], 
            y_range=[-3, 3, 1], 
            x_length=10, 
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text}
        ).shift(DOWN * 0.5)
        
        co2_mean = np.mean(co2_raw)
        co2_std = np.std(co2_raw)
        
        # Hidden axis for centered-but-unscaled data - ENLARGED to match ax_z
        ax_hidden = Axes(
            x_range=[1950, 2030, 20], 
            y_range=[-60, 60, 20], 
            x_length=10, 
            y_length=5
        ).shift(DOWN * 0.5)
        
        # Start line
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
        
        # Move formula to top right to clear space
        self.play(
            Transform(math_title, revelation_title),
            formula.animate.scale(0.7).to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)
        )
        
        temp_mean = np.mean(temp_raw)
        temp_std = np.std(temp_raw)
        temp_z_line = ax_z.plot_line_graph(X, (temp_raw - temp_mean) / temp_std, add_vertex_dots=False, line_color=ORANGE)
        
        # Reposition labels
        label_z_co2 = VGroup(
            MathTex(r"CO_2", font_size=28, color=BLUE), 
            Text(" (Standardized)", font_size=18, color=BLUE)
        ).arrange(RIGHT, buff=0.1).next_to(ax_z, UP, buff=0.2).to_edge(LEFT, buff=1.5)
        
        label_z_temp = Text("Temp (Standardized)", font_size=18, color=ORANGE).next_to(ax_z, UP, buff=0.2).to_edge(RIGHT, buff=1.5)

        self.play(Create(temp_z_line), FadeIn(label_z_co2), FadeIn(label_z_temp))
        self.wait(1)
        
        conclusion = Text("Trends become comparable once scales are unified.", font_size=24).to_edge(DOWN, buff=0.5)
        self.play(Write(conclusion))
        self.wait(5)
