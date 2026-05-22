from manim import *
import numpy as np
from co2_prediction import load_csv, linear_func, exponential_func, rational_2_1, fit_models
import calendar

class CO2PredictionExplainer(Scene):
    def construct(self):
        # 1. Load Data and Fit Models (Re-fitting here for precision)
        data = load_csv('data/co2-ppm.csv')
        years_hist, co2_hist = data[:, 0], data[:, 1]
        
        # Fit Global Linear
        _, p_lin, _ = fit_models(years_hist, co2_hist, "linear")
        
        # Segment 2 Models (Post-1990)
        mask2 = years_hist > 1990
        _, p_exp, _ = fit_models(years_hist[mask2], co2_hist[mask2], "exponential")
        _, p_rat, _ = fit_models(years_hist[mask2], co2_hist[mask2], "rational_2_1")

        # Helpers for date formatting
        def format_date(decimal_year):
            year = int(decimal_year)
            remainder = decimal_year - year
            month_idx = int(remainder * 12) + 1
            month_idx = max(1, min(12, month_idx))
            return f"{calendar.month_name[month_idx]} {year}"

        # Colors from project standards
        BLUE_C = "#0072B2"
        ORANGE_C = "#D55E00"
        GREEN_C = "#009E73"
        GRAY_C = "#999999"
        RED_C = "#CC79A7"

        # Helpers for subscripts without LaTeX
        def get_co2_label(font_size=24, color=WHITE):
            main = Text("CO", font_size=font_size, color=color)
            sub = Text("2", font_size=font_size * 0.7, color=color)
            sub.next_to(main, DR, buff=0.03).shift(UP * 0.08)
            return VGroup(main, sub)

        # 2. Setup initial Axes (1950 - 2060)
        axes = Axes(
            x_range=[1950, 2060, 20],
            y_range=[300, 750, 100],
            x_length=10,
            y_length=6,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 20},
            tips=False
        ).shift(DOWN * 0.5)
        
        y_label_group = VGroup(
            get_co2_label(font_size=20),
            Text(" (ppm)", font_size=18)
        ).arrange(RIGHT, buff=0.1)

        labels = axes.get_axis_labels(
            x_label=Text("Year", font_size=20),
            y_label=y_label_group
        )
        
        title_co2 = get_co2_label(font_size=36)
        title_rest = Text(" Future", font_size=36)
        title = VGroup(Text("Forecasting the ", font_size=36), title_co2, title_rest).arrange(RIGHT, buff=0.1).to_edge(UP)
        
        # Plot Historical Data - Redrawable to stay synced with axes
        dots = always_redraw(lambda: VGroup(*[Dot(axes.c2p(x, y), color=GRAY_C, radius=0.03, fill_opacity=0.4) for x, y in zip(years_hist, co2_hist)]))
        
        self.play(Write(title), Create(axes), FadeIn(labels))
        self.play(FadeIn(dots))
        self.wait(1)

        # 3. Predict to 2050
        # Dynamic value tracker for the current year
        time_tracker = ValueTracker(2023)
        
        def get_line(model_fn, params, start_year, end_year_tracker, color):
            return always_redraw(lambda: axes.plot(
                model_fn,
                x_range=[start_year, end_year_tracker.get_value()],
                color=color,
                stroke_width=3
            ))

        line_lin = get_line(lambda x: linear_func(x, *p_lin), p_lin, 1959, time_tracker, BLUE_C)
        line_exp = get_line(lambda x: exponential_func(x, *p_exp), p_exp, 1990, time_tracker, ORANGE_C)
        line_rat = get_line(lambda x: rational_2_1(x, *p_rat), p_rat, 1990, time_tracker, GREEN_C)

        self.add(line_lin, line_exp, line_rat)
        
        # Counter for Current Date
        date_text = always_redraw(lambda: Text(
            format_date(time_tracker.get_value()),
            font_size=24, color=YELLOW
        ).to_corner(UR, buff=1.0))
        
        self.play(Write(date_text))
        
        # Growth to 2050
        self.play(time_tracker.animate.set_value(2050), run_time=5, rate_func=linear)
        self.wait(2)

        # Show 2050 Values
        val_lin_2050 = linear_func(2050, *p_lin)
        val_exp_2050 = exponential_func(2050, *p_exp)
        val_rat_2050 = rational_2_1(2050, *p_rat)

        label_2050 = VGroup(
            Text(f"2050 Projections:", font_size=20),
            Text(f"Linear: {val_lin_2050:.1f} ppm", font_size=18, color=BLUE_C),
            Text(f"Baseline: {val_exp_2050:.1f} ppm", font_size=18, color=ORANGE_C),
            Text(f"Rational: {val_rat_2050:.1f} ppm", font_size=18, color=GREEN_C)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(date_text, DOWN, buff=0.5).align_to(date_text, LEFT)
        
        self.play(FadeIn(label_2050))
        self.wait(3)
        self.play(FadeOut(label_2050))

        # 4. Rescale and Growth to 685 ppm
        # Target years
        target_years = {
            "lin": 2194.9,
            "exp": 2077.8,
            "rat": 2079.3
        }
        max_target = max(target_years.values())

        # Animate axis range change
        new_axes = Axes(
            x_range=[1950, 2220, 40],
            y_range=[300, 750, 100],
            x_length=10,
            y_length=6,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 20},
            tips=False
        ).shift(DOWN * 0.5)
        
        new_labels = new_axes.get_axis_labels(
            x_label=Text("Year", font_size=20),
            y_label=VGroup(get_co2_label(font_size=20), Text(" (ppm)", font_size=18)).arrange(RIGHT, buff=0.1)
        )

        # Remove old redrawables before transform to avoid ghosting or scale issues
        self.remove(dots, line_lin, line_exp, line_rat)
        
        # Threshold Line (tied to new_axes)
        threshold_line = DashedLine(
            new_axes.c2p(1950, 685), new_axes.c2p(2220, 685),
            color=RED_C, stroke_width=2
        )
        threshold_label = Text("Threshold: 685 ppm", font_size=18, color=RED_C).next_to(threshold_line, UP, buff=0.1).to_edge(RIGHT, buff=1.0)

        self.play(
            ReplacementTransform(axes, new_axes),
            ReplacementTransform(labels, new_labels),
            run_time=2
        )
        axes = new_axes # Update reference for updaters
        
        # Re-add redrawables with updated axes reference
        dots = always_redraw(lambda: VGroup(*[Dot(axes.c2p(x, y), color=GRAY_C, radius=0.03, fill_opacity=0.4) for x, y in zip(years_hist, co2_hist)]))
        line_lin = get_line(lambda x: linear_func(x, *p_lin), p_lin, 1959, time_tracker, BLUE_C)
        line_exp = get_line(lambda x: exponential_func(x, *p_exp), p_exp, 1990, time_tracker, ORANGE_C)
        line_rat = get_line(lambda x: rational_2_1(x, *p_rat), p_rat, 1990, time_tracker, GREEN_C)
        
        self.add(dots, line_lin, line_exp, line_rat)
        self.play(Write(threshold_line), Write(threshold_label))
        self.wait(1)

        # Continue growth to 685 ppm
        self.play(time_tracker.animate.set_value(max_target), run_time=8, rate_func=linear)
        self.wait(1)

        # Mark individual crossing points - STAGGERED to avoid clipping
        stagger = [UP, DOWN, UP]
        for i, (key, yr) in enumerate(target_years.items()):
            color = BLUE_C if key == "lin" else (ORANGE_C if key == "exp" else GREEN_C)
            direction = stagger[i % 3]
            # Use closures carefully in lambdas
            dot = always_redraw(lambda yr=yr, color=color: Dot(axes.c2p(yr, 685), color=color))
            lbl = always_redraw(lambda yr=yr, color=color, dot=dot, direction=direction: 
                                Text(format_date(yr), font_size=16, color=color).next_to(dot, direction, buff=0.3))
            self.play(Create(dot), Write(lbl))

        self.wait(5)
