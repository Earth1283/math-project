from manim import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from nonlinear_analysis import load_and_align_data, segment_data, fit_models, quadratic_func, exponential_func, linear_func

class NonlinearOdyssey(Scene):
    def construct(self):
        # --- Helpers ---
        def get_subscript(main_text, sub_text, color=WHITE, font_size=24):
            main = Text(main_text, font_size=font_size, color=color)
            sub = Text(sub_text, font_size=font_size * 0.65, color=color)
            sub.next_to(main, DR, buff=0.02).shift(UP * 0.08)
            return VGroup(main, sub)

        def get_superscript(main_text, super_text, color=WHITE, font_size=24):
            main = Text(main_text, font_size=font_size, color=color)
            sup = Text(super_text, font_size=font_size * 0.6, color=color)
            sup.next_to(main, UR, buff=0.02).shift(DOWN * 0.05)
            return VGroup(main, sup)

        # 1. Load REAL Data and Fit Models
        years, co2, temp = load_and_align_data('data/co2-ppm.csv', 'data/surface-air-temp-change.csv')
        s1, s2 = segment_data(years, co2, temp, 1990)
        
        # Get actual coefficients and stats
        _, popt_lin, r2_lin, y_pred_lin, _ = fit_models(co2, temp, 'linear')
        _, popt_s1, r2_s1, y_pred_s1, _ = fit_models(s1[1], s1[2], 'quadratic')
        _, popt_s2, r2_s2, y_pred_s2, _ = fit_models(s2[1], s2[2], 'exponential')
        
        bp_year = 1990
        idx = np.where(years <= bp_year)[0][-1]
        bp_co2 = co2[idx]
        
        # Calculate piecewise R2
        y_pred_total = np.concatenate([y_pred_s1, y_pred_s2])
        total_r2 = r2_score(temp, y_pred_total)

        # 2. Setup Axes
        title = Text("The Nonlinear Odyssey", font_size=40).to_edge(UP)
        self.play(Write(title))
        
        axes = Axes(
            x_range=[310, 420, 20],
            y_range=[-0.5, 1.5, 0.5],
            x_length=9,
            y_length=5,
            axis_config={
                "include_numbers": True,
                "label_constructor": Text,
                "font_size": 20
            },
            tips=False
        ).shift(DOWN * 0.5)
        
        co2_lab = VGroup(MathTex(r"CO_2", font_size=30), Text(" (ppm)", font_size=20)).arrange(RIGHT, buff=0.1)
        temp_lab = Text("Temp Change (C)", font_size=20)
        labels = axes.get_axis_labels(x_label=co2_lab, y_label=temp_lab)
        
        dots = VGroup(*[Dot(axes.c2p(x, y), color=BLUE, radius=0.04, fill_opacity=0.6) for x, y in zip(co2, temp)])
        
        self.play(Create(axes), FadeIn(labels))
        self.play(FadeIn(dots, lag_ratio=0.01))
        self.wait(1)
        
        # --- R2 Display Setup ---
        r2_tracker = ValueTracker(0)
        # Use proper subscript/superscript formatting for R^2
        r2_main = Text("R", font_size=30, color=YELLOW)
        r2_sup = Text("2", font_size=20, color=YELLOW).next_to(r2_main, UR, buff=0.03).shift(DOWN * 0.05)
        r2_label_text = VGroup(r2_main, r2_sup)
        
        r2_equals = Text(" = ", font_size=30, color=YELLOW).next_to(r2_label_text, RIGHT, buff=0.1)
        
        def r2_val_func():
            return Text(f"{max(0, r2_tracker.get_value()):.4f}", font_size=30, color=YELLOW).next_to(r2_equals, RIGHT, buff=0.1)
        
        val_text_mob = always_redraw(r2_val_func)
        r2_display = VGroup(r2_label_text, r2_equals, val_text_mob).to_edge(RIGHT, buff=1.0).shift(UP*1.5)
        
        # 3. Scene 1: Global Linear Baseline
        lin_label = Text("Attempt 0: Global Linear", font_size=24).to_corner(UL).shift(DOWN*0.5)
        lin_eq_str = f"y = {popt_lin[0]:.4f}x {'+' if popt_lin[1]>=0 else '-'} {abs(popt_lin[1]):.4f}"
        lin_eq = Text(lin_eq_str, font_size=20).next_to(lin_label, DOWN, aligned_edge=LEFT)
        lin_comment = Text("\"It's honest work...\"", font_size=20, slant=ITALIC).next_to(lin_eq, DOWN, aligned_edge=LEFT)
        
        line_lin = axes.plot(lambda x: linear_func(x, *popt_lin), color=WHITE)
        
        self.play(Write(lin_label), Write(lin_eq), FadeIn(lin_comment))
        self.play(Create(line_lin), FadeIn(r2_display))
        self.play(r2_tracker.animate.set_value(r2_lin), run_time=1.5)
        self.wait(2)

        # 4. Scene 2: The Existential Crisis
        rat_label = Text("Attempt 1: Global Rational Fit", font_size=24, color=RED).to_corner(UL).shift(DOWN*0.5)
        rat_comment = Text("Equation is having an existential crisis...", font_size=20, color=RED, slant=ITALIC).next_to(rat_label, DOWN, aligned_edge=LEFT)
        
        # Fake a wildly unstable rational curve
        line_fail = axes.plot(lambda x: (0.5 * (x-310)**2 - 10*(x-310) + 5) / (2*(x-310) + 1) - 0.2, x_range=[315, 415], color=RED)

        self.play(FadeOut(VGroup(lin_label, lin_eq, lin_comment)), Write(rat_label))
        # Animate R2 dropping during failure
        self.play(Create(line_fail), r2_tracker.animate.set_value(0.15), run_time=2)
        self.play(Write(rat_comment))
        
        # Vibration effect
        for _ in range(3):
            self.play(line_fail.animate.shift(UP*0.05 + RIGHT*0.02), run_time=0.05)
            self.play(line_fail.animate.shift(DOWN*0.1 + LEFT*0.04), run_time=0.05)
            self.play(line_fail.animate.shift(UP*0.05 + RIGHT*0.02), run_time=0.05)
            
        self.wait(2)
        self.play(FadeOut(VGroup(rat_label, rat_comment, line_fail)))

        # 5. Scene 3: The Snap (Splitting the Linear Line)
        break_text = Text("GALAXY BRAIN MOMENT: Segment at 1990", font_size=32, color=GOLD).to_edge(UP)
        self.play(Transform(title, break_text))
        
        # Correctly split linear line using CO2 ranges
        line_lin_1 = axes.plot(lambda x: linear_func(x, *popt_lin), x_range=[co2.min(), bp_co2], color=WHITE)
        line_lin_2 = axes.plot(lambda x: linear_func(x, *popt_lin), x_range=[bp_co2, co2.max()], color=WHITE)
        
        self.add(line_lin_1, line_lin_2)
        self.remove(line_lin)
        
        bp_line = DashedLine(axes.c2p(bp_co2, -0.5), axes.c2p(bp_co2, 1.5), color=GOLD)
        bp_label = Text("1990: The Math Hinge", font_size=18, color=GOLD).next_to(bp_line, UP)
        
        self.play(Create(bp_line), Write(bp_label))
        
        # Trial: Two Rationals (Funny)
        trial_label = Text("Trial: Two Rational Fits", font_size=24, color=YELLOW).to_corner(UL).shift(DOWN*0.5)
        trial_comment = Text("Calculations are screaming...", font_size=20, color=YELLOW, slant=ITALIC).next_to(trial_label, DOWN, aligned_edge=LEFT)
        
        # S1: Over-oscillating
        trial_1 = axes.plot(lambda x: 0.1 * (x-330) + 0.2 * np.sin(0.5 * x), x_range=[co2.min(), bp_co2], color=YELLOW)
        # S2: Wild Asymptote
        trial_2 = axes.plot(lambda x: -10/(x-430) + 0.5, x_range=[bp_co2, co2.max()], color=YELLOW)
        
        self.play(Write(trial_label))
        # Animate R2 changing during trials
        self.play(Create(trial_1), Create(trial_2), r2_tracker.animate.set_value(0.08), run_time=2)
        self.play(Write(trial_comment))
        self.wait(2)
        self.play(FadeOut(VGroup(trial_1, trial_2, trial_label, trial_comment)))

        # 6. Scene 4: The Bend (Transform segments into Quad and Exp)
        win_label = Text("Now we're cooking with science!", font_size=32, color=GREEN).to_corner(UL).shift(DOWN*0.5)
        
        s1_curve = axes.plot(lambda x: quadratic_func(x, *popt_s1), x_range=[co2.min(), bp_co2], color=BLUE)
        s2_curve = axes.plot(lambda x: exponential_func(x, *popt_s2), x_range=[bp_co2, co2.max()], color=RED)
        
        # --- Polished Equation Formatting ---
        def get_quad_eq(popt):
            a, b, c = popt
            base = Text(f"y = {a:.5f}x", font_size=16, color=BLUE)
            sup = Text("2", font_size=10, color=BLUE).next_to(base, UR, buff=0.02).shift(DOWN*0.05)
            rest = Text(f" {'+' if b>=0 else '-'} {abs(b):.4f}x {'+' if c>=0 else '-'} {abs(c):.4f}", font_size=16, color=BLUE)
            return VGroup(VGroup(base, sup), rest).arrange(RIGHT, buff=0.1)

        def get_exp_eq(popt):
            a, b, c, x0 = popt
            base = Text(f"y = {a:.2e}e", font_size=16, color=RED)
            expo = Text(f"{b:.4f}(x-{x0:.1f})", font_size=10, color=RED).next_to(base, UR, buff=0.02).shift(DOWN*0.05)
            const = Text(f" {'+' if c>=0 else '-'} {abs(c):.4f}", font_size=16, color=RED)
            return VGroup(VGroup(base, expo), const).arrange(RIGHT, buff=0.1)

        # Repositioned equations for more room
        eq1 = get_quad_eq(popt_s1).move_to(axes.c2p(325, 1.2))
        eq2 = get_exp_eq(popt_s2).move_to(axes.c2p(380, 0.2))

        self.play(FadeIn(win_label))
        # Animate R2 climbing to the ultimate score
        self.play(
            Transform(line_lin_1, s1_curve),
            Transform(line_lin_2, s2_curve),
            r2_tracker.animate.set_value(total_r2),
            run_time=3
        )
        self.play(Write(eq1), Write(eq2))
        
        final_msg = Text("Rigor > Linear Simplicity", font_size=32, color=GOLD).next_to(r2_display, DOWN, buff=0.5)
        self.play(Write(final_msg))
        
        self.wait(5)
