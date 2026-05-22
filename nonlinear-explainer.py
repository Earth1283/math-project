from manim import *
import numpy as np
from sklearn.linear_model import LinearRegression
from nonlinear_analysis import load_and_align_data, fit_models

class NonlinearOdyssey(Scene):
    def construct(self):
        # 1. Load REAL Data
        years, co2, temp = load_and_align_data('data/co2-ppm.csv', 'data/surface-air-temp-change.csv')
        step = len(years) // 30
        X_sample = co2[::step]
        y_sample = temp[::step]

        # 2. Scene 1: The Global Baseline
        title = Text("The Nonlinear Odyssey", font_size=40).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(
            x_range=[310, 420, 20],
            y_range=[-0.5, 1.5, 0.5],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text}
        ).shift(DOWN * 0.5)
        
        # Subscript helper for labels
        def get_subscript(main_text, sub_text, color=WHITE, font_size=24):
            main = Text(main_text, font_size=font_size, color=color)
            sub = Text(sub_text, font_size=font_size * 0.65, color=color)
            sub.next_to(main, DR, buff=0.03).shift(UP * 0.08)
            return VGroup(main, sub)

        co2_lab = VGroup(get_subscript("CO", "2", font_size=20), Text(" (ppm)", font_size=20)).arrange(RIGHT, buff=0.05)
        labels = axes.get_axis_labels(x_label=co2_lab, y_label=Text("Temp (C)", font_size=20))
        
        dots = VGroup(*[Dot(axes.c2p(x, y), color=BLUE, radius=0.06) for x, y in zip(X_sample, y_sample)])
        
        self.play(Create(axes), FadeIn(labels))
        self.play(FadeIn(dots, lag_ratio=0.1))
        self.wait(1)
        
        # Fit global linear
        m_lin, b_lin = np.polyfit(X_sample, y_sample, 1)
        line_lin = axes.plot(lambda x: m_lin * x + b_lin, color=WHITE)
        lin_label = Text("Attempt 0: Global Linear", font_size=24).to_corner(UL).shift(DOWN*0.5)
        lin_comment = Text("\"It's honest work...\"", font_size=20, slant=ITALIC).next_to(lin_label, DOWN, aligned_edge=LEFT)
        
        self.play(Create(line_lin), Write(lin_label))
        self.play(FadeIn(lin_comment))
        self.wait(2)
        self.play(FadeOut(VGroup(line_lin, lin_label, lin_comment)))
