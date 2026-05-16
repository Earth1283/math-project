from manim import *
import numpy as np

class LinearRegressionExplainer(Scene):
    def construct(self):
        # --- Helper for Subscripts ---
        def get_subscript(main_text, sub_text, color=WHITE, font_size=24):
            main = Text(main_text, font_size=font_size, color=color)
            # Increased subscript font size ratio from 0.5 to 0.65 for legibility
            sub = Text(sub_text, font_size=font_size * 0.65, color=color)
            sub.next_to(main, DR, buff=0.03).shift(UP * 0.08)
            return VGroup(main, sub)

        def get_superscript(main_text, super_text, color=WHITE, font_size=24):
            main = Text(main_text, font_size=font_size, color=color)
            sup = Text(super_text, font_size=font_size * 0.6, color=color)
            sup.next_to(main, UR, buff=0.02).shift(DOWN * 0.05)
            return VGroup(main, sup)

        # 1. Introduction
        title = Text("CO2 vs. Temperature Change", font_size=32).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(1)
        
        # Simulated Data for Visualization
        np.random.seed(42)
        X = np.linspace(315, 415, 30)
        y = 0.01 * X - 3.2 + np.random.normal(0, 0.2, 30)
        
        # Create Axes - Compact and shifted to bottom-left
        axes = Axes(
            x_range=[310, 420, 20],
            y_range=[-0.5, 1.5, 0.5],
            x_length=6,
            y_length=4,
            axis_config={
                "include_numbers": True,
                "label_constructor": Text,
            }
        ).to_edge(DL, buff=1.0)
        
        x_label = axes.get_x_axis_label(Text("CO2 (ppm)", font_size=18)).shift(UP * 0.2)
        y_label = axes.get_y_axis_label(Text("Temp (C)", font_size=18)).shift(RIGHT * 0.2)
        
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
        
        # Plot Scatter Points
        dots = VGroup(*[Dot(axes.c2p(x_val, y_val), color=BLUE, radius=0.06) for x_val, y_val in zip(X, y)])
        self.play(FadeIn(dots, lag_ratio=0.1))
        self.wait(1)
        
        # 3. Best Fit Math and Residuals
        m_guess, b_guess = 0.005, -1.5
        equation = Text("y = mx + b", slant=ITALIC, font_size=28).to_edge(RIGHT, buff=2.5).shift(UP * 2.0)
        
        residual_label = Text("Residuals =", font_size=20, color=RED)
        residual_math = Text("Observed - Predicted", font_size=20, color=RED).next_to(residual_label, RIGHT, buff=0.2)
        # Moved further from the right edge to avoid cutoff
        residual_text = VGroup(residual_label, residual_math).next_to(equation, DOWN, buff=0.5).to_edge(RIGHT, buff=1.5)
        
        guess_line = axes.plot(
            lambda x: m_guess * x + b_guess,
            color=YELLOW,
            x_range=[310, 420]
        )
        
        self.play(Write(equation))
        self.wait(0.5)
        self.play(Create(guess_line))
        self.wait(0.5)
        
        # Visualize Residuals
        residuals = VGroup(*[
            DashedLine(
                start=dots[i].get_center(),
                end=axes.c2p(X[i], m_guess * X[i] + b_guess),
                color=RED,
                stroke_width=1.5
            )
            for i in range(len(X))
        ])
        
        self.play(Create(residuals, lag_ratio=0.02), Write(residual_text))
        self.wait(2)

        # 4. Python Code and Optimal Fit
        code_snippet = "model = LinearRegression()\nmodel.fit(X, y)"
        code = Code(
            code_string=code_snippet,
            language="python",
            formatter_style="monokai",
            add_line_numbers=False,
            background="window",
        ).scale(0.5).to_corner(UR, buff=0.8).shift(DOWN * 0.5)

        # Repositioned optimization text
        optimization_text = Text("Finding Optimal Fit...", font_size=22, color=GREEN).next_to(code, DOWN, buff=0.4).align_to(code, LEFT)

        # Optimal parameters
        m_opt, b_opt = 0.0106, -3.4445
        optimal_line = axes.plot(
            lambda x: m_opt * x + b_opt,
            color=GREEN,
            x_range=[310, 420]
        )
        
        # --- Advanced Statistics Section ---
        ss_res_val = np.sum((y - (m_opt * X + b_opt))**2)
        ss_tot_val = np.sum((y - np.mean(y))**2)
        r2_val = 1 - (ss_res_val / ss_tot_val)
        
        # 1. SS_res Morphing
        ss_label = Text("Sum of Squared Residuals:", font_size=20, color=YELLOW)
        ss_eq_start = VGroup(
            get_subscript("SS", "res", color=YELLOW, font_size=24),
            Text("= \u03a3(y - \u0177)\u00b2", font_size=24, color=YELLOW)
        ).arrange(RIGHT, buff=0.1)
        
        ss_eq_final = VGroup(
            get_subscript("SS", "res", color=YELLOW, font_size=24),
            Text(f"= {ss_res_val:.4f}", font_size=24, color=YELLOW)
        ).arrange(RIGHT, buff=0.1)

        # 2. R^2 Morphing
        r2_label = Text("Coefficient of Determination:", font_size=20, color=GREEN)
        
        # Stage A: General Formula
        r2_prefix = Text("R\u00b2 = 1 - ", font_size=24, color=GREEN)
        num_a = get_subscript("SS", "res", color=GREEN, font_size=16)
        den_a = get_subscript("SS", "tot", color=GREEN, font_size=16)
        line_a = Line(LEFT, RIGHT, color=GREEN, stroke_width=1.5).scale(0.35).next_to(num_a, DOWN, buff=0.1)
        den_a.next_to(line_a, DOWN, buff=0.1)
        frac_a = VGroup(num_a, line_a, den_a).next_to(r2_prefix, RIGHT, buff=0.15).shift(UP * 0.05)
        r2_stage_a = VGroup(r2_prefix, frac_a)

        # Stage B: Plugged in Values
        r2_prefix_b = r2_prefix.copy()
        num_b = Text(f"{ss_res_val:.2f}", font_size=16, color=GREEN)
        den_b = Text(f"{ss_tot_val:.2f}", font_size=16, color=GREEN)
        line_b = Line(LEFT, RIGHT, color=GREEN, stroke_width=1.5).scale(0.35).next_to(num_b, DOWN, buff=0.1)
        den_b.next_to(line_b, DOWN, buff=0.1)
        frac_b = VGroup(num_b, line_b, den_b).next_to(r2_prefix_b, RIGHT, buff=0.15).shift(UP * 0.05)
        r2_stage_b = VGroup(r2_prefix_b, frac_b)

        # Stage C: Final Value
        r2_stage_c = Text(f"R\u00b2 = {r2_val:.4f}", font_size=24, color=GREEN)

        # Positioning
        stats_group_y = 1.0
        ss_label.to_edge(RIGHT, buff=1.0).shift(UP * stats_group_y)
        ss_eq_start.next_to(ss_label, DOWN, buff=0.3).align_to(ss_label, LEFT)
        ss_eq_final.move_to(ss_eq_start, aligned_edge=LEFT)

        r2_label.next_to(ss_eq_start, DOWN, buff=0.8).align_to(ss_label, LEFT)
        r2_stage_a.next_to(r2_label, DOWN, buff=0.4).align_to(ss_label, LEFT)
        r2_stage_b.move_to(r2_stage_a, aligned_edge=LEFT)
        r2_stage_c.move_to(r2_stage_a, aligned_edge=LEFT)

        # Animation Sequence
        self.play(
            FadeOut(equation),
            FadeOut(residual_text),
            FadeIn(code),
        )
        self.wait(1)

        self.play(Write(optimization_text))
        self.wait(0.5)

        self.play(
            Transform(guess_line, optimal_line),
            FadeOut(residuals),
            run_time=3
        )
        self.wait(1)

        # SS_res sequence
        self.play(Write(ss_label))
        self.play(Write(ss_eq_start))
        self.wait(1.5)
        self.play(Transform(ss_eq_start, ss_eq_final))
        self.wait(1.5)

        # R^2 sequence
        self.play(Write(r2_label))
        self.play(Write(r2_stage_a))
        self.wait(1.5)
        self.play(Transform(r2_stage_a, r2_stage_b))
        self.wait(1.5)
        self.play(Transform(r2_stage_a, r2_stage_c))
        
        self.wait(5)
