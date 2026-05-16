from manim import *
import numpy as np

class ZScoreExplainer(Scene):
    def construct(self):
        # --- Helpers ---
        def get_subscript(main_text, sub_text, color=WHITE, font_size=24):
            main = Text(main_text, font_size=font_size, color=color)
            sub = Text(sub_text, font_size=font_size * 0.65, color=color)
            sub.next_to(main, DR, buff=0.03).shift(UP * 0.08)
            return VGroup(main, sub)

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
