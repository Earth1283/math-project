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

        # 3. Scene 2: Attempting Global Rational
        rat_label = Text("Attempt 1: Global Rational Fit", font_size=24, color=RED).to_corner(UL).shift(DOWN*0.5)
        rat_comment = Text("Equation is having an existential crisis...", font_size=20, color=RED, slant=ITALIC).next_to(rat_label, DOWN, aligned_edge=LEFT)
        emoji = Text("😵‍💫", font_size=40).next_to(rat_comment, RIGHT) # 😵‍💫

        # Create a "forced" looking rational curve that fits poorly
        line_fail = axes.plot(lambda x: (0.5 * (x-310)**2 - 10*(x-310) + 5) / (2*(x-310) + 1) - 0.2, x_range=[315, 415], color=RED)

        self.play(Write(rat_label))
        self.play(Create(line_fail), run_time=2)
        self.play(Write(rat_comment), FadeIn(emoji))
        
        # Vibration
        self.play(line_fail.animate.shift(UP*0.1), run_time=0.1)
        self.play(line_fail.animate.shift(DOWN*0.1), run_time=0.1)
        self.play(line_fail.animate.shift(UP*0.1), run_time=0.1)
        
        self.wait(2)
        self.play(FadeOut(VGroup(rat_label, rat_comment, emoji, line_fail)))

        # 4. Scene 3: The Breakthrough
        break_text = Text("GALAXY BRAIN MOMENT: Segment at 1990", font_size=32, color=GOLD).to_edge(UP)
        self.play(Transform(title, break_text))
        
        bp_x = 354
        bp_line = DashedLine(axes.c2p(bp_x, -0.5), axes.c2p(bp_x, 1.5), color=GOLD)
        bp_label = Text("1990: The Math Hinge", font_size=18, color=GOLD).next_to(bp_line, UP)
        
        self.play(Create(bp_line), Write(bp_label))
        self.wait(1)

        # 5. Scene 4: The Win
        win_label = Text("Now we're cooking with science! \ud83d\udd25", font_size=32, color=GREEN).to_corner(UL).shift(DOWN*0.5)
        
        s1_line = axes.plot(lambda x: 0.00022 * x**2 - 0.1468 * x + 22.8051, x_range=[315.98, 354.35], color=BLUE)
        s2_line = axes.plot(lambda x: 2.33e-7 * np.exp(0.0470 * (x - 355.62)) - 0.2081, x_range=[355.62, 414.24], color=RED)

        self.play(FadeIn(win_label))
        self.play(Create(s1_line), Create(s2_line), run_time=3)
        
        r2_text = Text(f"Piecewise R\u00b2 = 0.9291", font_size=36, color=GREEN).to_edge(RIGHT, buff=1.0).shift(UP*1.5)
        self.play(Write(r2_text))
        self.wait(5)
