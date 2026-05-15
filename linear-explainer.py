from manim import *
import numpy as np

class LinearRegressionExplainer(Scene):
    def construct(self):
        # 1. Introduction
        title = Text("CO2 vs. Temperature Change")
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # Simulated Data for Visualization
        np.random.seed(42)
        X = np.linspace(315, 415, 30)
        y = 0.01 * X - 3.2 + np.random.normal(0, 0.2, 30)
        
        # Create Axes
        axes = Axes(
            x_range=[310, 420, 20],
            y_range=[-0.5, 1.5, 0.5],
            x_length=7,
            y_length=4,
            axis_config={
                "include_numbers": True,
                "label_constructor": Text,
            }
        ).shift(DOWN * 0.5)
        
        x_label = axes.get_x_axis_label(Text("CO2 (ppm)", font_size=24))
        y_label = axes.get_y_axis_label(Text("Temp (C)", font_size=24))
        
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
        
        # Plot Scatter Points
        dots = VGroup(*[Dot(axes.c2p(x_val, y_val), color=BLUE) for x_val, y_val in zip(X, y)])
        self.play(FadeIn(dots, lag_ratio=0.1))
        self.wait(1)
        
        # 3. Best Fit Math and Residuals
        m_guess, b_guess = 0.005, -1.5
        equation = Text("y = mx + b").to_corner(UR).shift(LEFT * 0.5)
        
        guess_line = axes.plot(
            lambda x: m_guess * x + b_guess,
            color=YELLOW,
            x_range=[310, 420]
        )
        
        self.play(Write(equation))
        self.play(Create(guess_line))
        self.wait(1)
        
        # Visualize Residuals
        residuals = VGroup(*[
            DashedLine(
                start=dots[i].get_center(),
                end=axes.c2p(X[i], m_guess * X[i] + b_guess),
                color=RED,
                stroke_width=2
            )
            for i in range(len(X))
        ])
        
        self.play(Create(residuals, lag_ratio=0.05))
        self.wait(2)
