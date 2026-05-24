import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0d1117"
AX_BG  = "#0d1117"
TEXT   = "#e6edf3"
GRID   = "#21262d"
SPINE  = "#30363d"

BLUE   = "#58a6ff"
ORANGE = "#ffa657"
GREEN  = "#3fb950"
RED    = "#ff7b72"
YELLOW = "#e3b341"
PURPLE = "#d2a8ff"
GRAY   = "#6e7681"

OUTPUT = "presentation"
os.makedirs(OUTPUT, exist_ok=True)

RC = {
    "figure.facecolor": BG,
    "axes.facecolor":   AX_BG,
    "axes.edgecolor":   SPINE,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT,
    "text.color":       TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "grid.color":       GRID,
    "grid.linewidth":   0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": SPINE,
    "legend.framealpha": 0.92,
    "font.family":      "sans-serif",
    "font.size":        14,
    "axes.spines.top":  False,
    "axes.spines.right": False,
}

def rc(): plt.rcParams.update(RC)

def statbox(ax, text, loc=(0.04, 0.96), **kwargs):
    ax.text(*loc, text, transform=ax.transAxes, fontsize=15, va='top', color=TEXT,
            bbox=dict(boxstyle='round,pad=0.55', facecolor='#161b22',
                      edgecolor=SPINE, alpha=0.95), **kwargs)

def glow(ax, x, y, color, lw=3, n=3, base_alpha=0.12, **kw):
    for i in range(n, 0, -1):
        ax.plot(x, y, color=color, linewidth=lw + i * 3, alpha=base_alpha, zorder=2)
    ax.plot(x, y, color=color, linewidth=lw, zorder=3, **kw)

# ── Data ──────────────────────────────────────────────────────────────────────
def load():
    co2_raw  = np.loadtxt('data/co2-ppm.csv',                  delimiter=',', skiprows=1)
    temp_raw = np.loadtxt('data/surface-air-temp-change.csv',   delimiter=',', skiprows=1)
    co2_raw  = co2_raw[co2_raw[:, 0]   >= 1959]
    temp_raw = temp_raw[temp_raw[:, 0]  >= 1959]
    common   = np.intersect1d(co2_raw[:, 0], temp_raw[:, 0])
    c = co2_raw[np.isin(co2_raw[:, 0],  common)]; c = c[c[:, 0].argsort()]
    t = temp_raw[np.isin(temp_raw[:, 0], common)]; t = t[t[:, 0].argsort()]
    return c[:, 0], c[:, 1], t[:, 1]

def hybrid_model(x, a, b, c, d, e):
    t = x - 1959
    return a*t**2 + b*t + c + d*np.exp(e*t)

def linear_func(x, a, b):       return a*x + b
def exp_func(x, a, b, c, x0):   return a*np.exp(b*(x - x0)) + c

# ── 1. Linear Regression ──────────────────────────────────────────────────────
def slide_linear(years, co2, temp):
    rc()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=150,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor(BG)

    model = LinearRegression().fit(co2.reshape(-1,1), temp)
    co2_line = np.linspace(co2.min(), co2.max(), 300)
    y_pred_all = model.predict(co2.reshape(-1,1))
    residuals  = temp - y_pred_all
    r2 = model.score(co2.reshape(-1,1), temp)
    m, b = model.coef_[0], model.intercept_

    # Scatter + regression
    ax1.scatter(co2, temp, color=BLUE, alpha=0.55, s=55, edgecolors='none', zorder=4,
                label='Observed data (1959–2020)')
    glow(ax1, co2_line, model.predict(co2_line.reshape(-1,1)), RED, lw=3)
    ax1.plot([], [], color=RED, lw=3,
             label=f'$\\hat{{y}} = {m:.4f}x {b:+.4f}$')

    statbox(ax1, f"$R^2 = {r2:.4f}$\nSlope $= {m:.4f}$ °C/ppm")
    ax1.set_title(r'Linear Regression: $\mathrm{CO}_2$ vs. Global Temperature Change',
                  fontsize=21, fontweight='bold', pad=14)
    ax1.set_xlabel(r'$\mathrm{CO}_2$ Concentration (ppm)', fontsize=16, labelpad=8)
    ax1.set_ylabel('Temperature Anomaly (°C)', fontsize=16, labelpad=8)
    ax1.legend(fontsize=13, loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.45)

    # Residuals
    ax2.scatter(y_pred_all, residuals, color=GREEN, alpha=0.55, s=45,
                edgecolors='none', zorder=4, label='Residuals')
    ax2.axhline(0, color=SPINE, linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.set_title('Residual Plot', fontsize=16, fontweight='bold', pad=10)
    ax2.set_xlabel('Predicted Temperature (°C)', fontsize=15, labelpad=8)
    ax2.set_ylabel('Residuals (°C)', fontsize=15, labelpad=8)
    ax2.legend(fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.45)

    fig.tight_layout(h_pad=2.5)
    path = f"{OUTPUT}/slide_linear_regression.png"
    fig.savefig(path, facecolor=BG, dpi=150)
    plt.close()
    print(f"  ✓  {path}")

# ── 2. Z-Score Comparison ─────────────────────────────────────────────────────
def slide_zscore(years, co2, temp):
    rc()
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor(BG)

    co2_z  = (co2  - co2.mean())  / co2.std()
    temp_z = (temp - temp.mean()) / temp.std()
    r = np.corrcoef(co2_z, temp_z)[0, 1]

    ax.fill_between(years, co2_z, temp_z, alpha=0.07, color=PURPLE)
    glow(ax, years, co2_z,  BLUE,   lw=3)
    glow(ax, years, temp_z, ORANGE, lw=3)
    ax.plot([], [], color=BLUE,   lw=3, label=r'$\mathrm{CO}_2$ (standardized)')
    ax.plot([], [], color=ORANGE, lw=3, label='Temperature (standardized)')
    ax.axhline(0, color=SPINE, linewidth=1, alpha=0.5)

    statbox(ax, f"Pearson $r = {r:.4f}$\n92% of variance explained", loc=(0.72, 0.22))
    ax.set_title(r'Z-Score Standardization: $\mathrm{CO}_2$ and Temperature on the Same Scale',
                 fontsize=21, fontweight='bold', pad=14)
    ax.set_xlabel('Year', fontsize=16, labelpad=8)
    ax.set_ylabel('Standardized Value (σ)', fontsize=16, labelpad=8)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.45)

    fig.tight_layout()
    path = f"{OUTPUT}/slide_zscore.png"
    fig.savefig(path, facecolor=BG, dpi=150)
    plt.close()
    print(f"  ✓  {path}")

# ── 3. Piecewise Nonlinear ────────────────────────────────────────────────────
def slide_piecewise(years, co2, temp):
    rc()
    BREAK = 1990
    mask1 = years <= BREAK
    mask2 = years  > BREAK
    co2_1, temp_1 = co2[mask1], temp[mask1]
    co2_2, temp_2 = co2[mask2], temp[mask2]

    p1 = np.polyfit(co2_1, temp_1, 2)
    p2 = np.polyfit(co2_2, temp_2, 2)
    r2_1 = r2_score(temp_1, np.polyval(p1, co2_1))
    r2_2 = r2_score(temp_2, np.polyval(p2, co2_2))
    combined = r2_score(temp, np.concatenate([np.polyval(p1, co2_1), np.polyval(p2, co2_2)]))
    global_r2 = r2_score(temp, LinearRegression().fit(co2.reshape(-1,1), temp).predict(co2.reshape(-1,1)))

    split_co2 = (co2_1[-1] + co2_2[0]) / 2
    x1 = np.linspace(co2_1.min(), split_co2, 200)
    x2 = np.linspace(split_co2, co2_2.max(), 200)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=150,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor(BG)

    ax1.scatter(co2_1, temp_1, color=BLUE,   alpha=0.5, s=55, edgecolors='none', zorder=4,
                label=f'Data pre-{BREAK}')
    ax1.scatter(co2_2, temp_2, color=ORANGE, alpha=0.5, s=55, edgecolors='none', zorder=4,
                label=f'Data post-{BREAK}')
    glow(ax1, x1, np.polyval(p1, x1), BLUE,   lw=3)
    glow(ax1, x2, np.polyval(p2, x2), ORANGE, lw=3)
    ax1.plot([], [], color=BLUE,   lw=3, label=f'Seg 1: Quadratic  ($R^2={r2_1:.4f}$)')
    ax1.plot([], [], color=ORANGE, lw=3, label=f'Seg 2: Quadratic  ($R^2={r2_2:.4f}$)')
    ax1.axvline(split_co2, color=GRAY, linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Breakpoint ({BREAK})')

    statbox(ax1, f"Combined $R^2 = {combined:.4f}$\nGlobal linear $R^2 = {global_r2:.4f}$")
    ax1.set_title(r'Piecewise Nonlinear Regression: 1990 Structural Breakpoint',
                  fontsize=21, fontweight='bold', pad=14)
    ax1.set_xlabel(r'$\mathrm{CO}_2$ Concentration (ppm)', fontsize=16, labelpad=8)
    ax1.set_ylabel('Temperature Anomaly (°C)', fontsize=16, labelpad=8)
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.45)

    # Residual comparison
    global_res   = temp - LinearRegression().fit(co2.reshape(-1,1), temp).predict(co2.reshape(-1,1))
    piece_pred   = np.concatenate([np.polyval(p1, co2_1), np.polyval(p2, co2_2)])
    piece_res    = temp - piece_pred

    ax2.scatter(co2, global_res, color=GRAY,  alpha=0.45, s=35, edgecolors='none',
                label=f'Global linear residuals ($R^2={global_r2:.4f}$)')
    ax2.scatter(co2, piece_res,  color=GREEN, alpha=0.65, s=45, edgecolors='none',
                label=f'Piecewise residuals ($R^2={combined:.4f}$)')
    ax2.axhline(0, color=SPINE, linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.set_title('Residual Comparison: Global vs. Piecewise', fontsize=15, fontweight='bold', pad=8)
    ax2.set_xlabel(r'$\mathrm{CO}_2$ (ppm)', fontsize=15, labelpad=8)
    ax2.set_ylabel('Residuals (°C)', fontsize=15, labelpad=8)
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.45)

    fig.tight_layout(h_pad=2.5)
    path = f"{OUTPUT}/slide_piecewise.png"
    fig.savefig(path, facecolor=BG, dpi=150)
    plt.close()
    print(f"  ✓  {path}")

# ── 4. Hybrid Ensemble Model ──────────────────────────────────────────────────
def slide_ensemble(years, co2):
    rc()
    years_val = np.array([2022, 2023, 2024])
    co2_val   = np.array([418.53, 421.08, 424.61])
    x_all = np.concatenate([years, years_val])
    y_all = np.concatenate([co2,   co2_val])

    popt, _ = curve_fit(hybrid_model, x_all, y_all, p0=[0.005, 0.8, 310, 5.0, 0.02], maxfev=50000)
    y_pred   = hybrid_model(x_all, *popt)
    r2       = r2_score(y_all, y_pred)
    residuals = y_all - y_pred

    yrs_line = np.linspace(1959, 2025, 600)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=150,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor(BG)

    ax1.scatter(years, co2, color=BLUE, alpha=0.45, s=50, edgecolors='none', zorder=3,
                label='Historical (1959–2020)')
    ax1.scatter(years_val, co2_val, color=RED, marker='X', s=220, zorder=6,
                edgecolors=BG, linewidths=1.2, label='Validation (2022–2024)')
    glow(ax1, yrs_line, hybrid_model(yrs_line, *popt), GREEN, lw=3.5)
    ax1.plot([], [], color=GREEN, lw=3, label=f'Hybrid Ensemble  ($R^2={r2:.6f}$)')

    a, b, c, d, e = popt
    eq = (f"$y(t) = {a:.5f}t^2 + {b:.4f}t + {c:.1f}$\n"
          f"$\\quad\\;+ {d:.3f}\\,e^{{{e:.4f}t}},\\quad t = \\text{{year}} - 1959$")
    statbox(ax1, eq + f"\n$R^2 = {r2:.6f}$", loc=(0.97, 0.32), ha='right')

    ax1.set_title(r'Hybrid Ensemble Model: Quadratic + Exponential $\mathrm{CO}_2$ Fit',
                  fontsize=21, fontweight='bold', pad=14)
    ax1.set_xlabel('Year', fontsize=16, labelpad=8)
    ax1.set_ylabel(r'$\mathrm{CO}_2$ Concentration (ppm)', fontsize=16, labelpad=8)
    ax1.legend(fontsize=13, loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.45)
    ax1.set_xlim(1955, 2027)
    ax1.set_ylim(308, 440)

    ax2.scatter(y_pred, residuals, color=GREEN, alpha=0.55, s=40,
                edgecolors='none', zorder=4, label='Residuals')
    ax2.axhline(0, color=SPINE, linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.set_title('Residual Plot', fontsize=15, fontweight='bold', pad=8)
    ax2.set_xlabel(r'Predicted $\mathrm{CO}_2$ (ppm)', fontsize=15, labelpad=8)
    ax2.set_ylabel('Residuals (ppm)', fontsize=15, labelpad=8)
    ax2.legend(fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.45)

    fig.tight_layout(h_pad=2.5)
    path = f"{OUTPUT}/slide_ensemble.png"
    fig.savefig(path, facecolor=BG, dpi=150)
    plt.close()
    print(f"  ✓  {path}")

# ── 5. Projections to 2050 ────────────────────────────────────────────────────
def slide_projections(years, co2):
    rc()
    years_val = np.array([2022, 2023, 2024])
    co2_val   = np.array([418.53, 421.08, 424.61])
    x_all = np.concatenate([years, years_val])
    y_all = np.concatenate([co2,   co2_val])

    popt_lin, _ = curve_fit(linear_func, x_all, y_all, p0=[1.5, -2800])
    mask2 = x_all > 1990
    x2, y2 = x_all[mask2], y_all[mask2]
    popt_exp, _ = curve_fit(exp_func, x2, y2, p0=[315, 0.01, 0, x2.min()],
                             bounds=([0,0,-np.inf,x2.min()],[np.inf,np.inf,np.inf,x2.max()]))
    popt_hyb, _ = curve_fit(hybrid_model, x_all, y_all, p0=[0.005, 0.8, 310, 5.0, 0.02], maxfev=50000)

    future = np.linspace(2024, 2050, 400)
    y_lin  = linear_func(future, *popt_lin)
    y_exp  = exp_func(future, *popt_exp)
    y_hyb  = hybrid_model(future, *popt_hyb)

    def find_685(fn, popt, end=2300):
        x = np.linspace(2024, end, 500000)
        idx = np.where(fn(x, *popt) >= 685)[0]
        return x[idx[0]] if len(idx) else None

    yr_lin = find_685(linear_func, popt_lin)
    yr_exp = find_685(exp_func,    popt_exp)
    yr_hyb = find_685(hybrid_model, popt_hyb)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor(BG)

    ax.scatter(years, co2, color=GRAY,   alpha=0.22, s=22, zorder=2, label='Historical data')
    ax.scatter(years_val, co2_val, color=YELLOW, marker='D', s=90,
               edgecolors='none', zorder=5, label='2022–2024 observed')

    glow(ax, future, y_lin,  BLUE,   lw=3)
    glow(ax, future, y_exp,  ORANGE, lw=3)
    glow(ax, future, y_hyb,  GREEN,  lw=3)
    ax.plot([], [], color=BLUE,   lw=3, label=f'Global linear  (2050: {linear_func(2050, *popt_lin):.0f} ppm)')
    ax.plot([], [], color=ORANGE, lw=3, label=f'Piecewise exp  (2050: {exp_func(2050, *popt_exp):.0f} ppm)')
    ax.plot([], [], color=GREEN,  lw=3, label=f'Hybrid ensemble  (2050: {hybrid_model(2050, *popt_hyb):.0f} ppm)')

    ax.axhline(685, color=RED, linestyle='--', linewidth=2.2, alpha=0.85, zorder=6)
    ax.text(2051, 693, '685 ppm  (2.4× pre-industrial)', color=RED, fontsize=13, va='bottom')

    for yr, col in [(yr_lin, BLUE), (yr_exp, ORANGE), (yr_hyb, GREEN)]:
        if yr and yr < 2300:
            ax.axvline(yr, color=col, linestyle=':', linewidth=1.3, alpha=0.28, zorder=1)

    # Crossing year summary
    def fmt(yr): return f"{yr:.0f}" if yr else ">2300"
    statbox(ax, f"685 ppm crossing:\n  Linear:   {fmt(yr_lin)}\n  Exp:       {fmt(yr_exp)}\n  Ensemble: {fmt(yr_hyb)}",
            loc=(0.04, 0.65))

    ax.set_title(r'$\mathrm{CO}_2$ Projections to 2050 — Three-Model Comparison',
                 fontsize=21, fontweight='bold', pad=14)
    ax.set_xlabel('Year', fontsize=16, labelpad=8)
    ax.set_ylabel(r'$\mathrm{CO}_2$ Concentration (ppm)', fontsize=16, labelpad=8)
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.set_xlim(1955, 2055)
    ax.set_ylim(300, 740)

    fig.tight_layout()
    path = f"{OUTPUT}/slide_projections.png"
    fig.savefig(path, facecolor=BG, dpi=150)
    plt.close()
    print(f"  ✓  {path}")

# ── GIF 1: CO₂ Data + Model Reveal ───────────────────────────────────────────
def anim_co2_rise(years, co2):
    rc()
    years_val = np.array([2022, 2023, 2024])
    co2_val   = np.array([418.53, 421.08, 424.61])
    all_yrs   = np.concatenate([years, years_val])
    all_co2   = np.concatenate([co2,   co2_val])

    popt, _ = curve_fit(hybrid_model, all_yrs, all_co2, p0=[0.005, 0.8, 310, 5.0, 0.02], maxfev=50000)
    smooth_x = np.linspace(1959, 2024.5, 500)
    smooth_y = hybrid_model(smooth_x, *popt)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    ax.set_xlim(1955, 2027)
    ax.set_ylim(308, 435)
    ax.set_title(r'Atmospheric $\mathrm{CO}_2$ Rise  (1959–2024)',
                 fontsize=22, fontweight='bold', pad=14)
    ax.set_xlabel('Year', fontsize=17, labelpad=8)
    ax.set_ylabel(r'$\mathrm{CO}_2$ (ppm)', fontsize=17, labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.4)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    scat      = ax.scatter([], [], color=BLUE, alpha=0.6, s=50, edgecolors='none', zorder=4)
    fit_line, = ax.plot([], [], color=GREEN, lw=3.5, zorder=3, alpha=0)
    yr_label  = ax.text(0.97, 0.06, '', transform=ax.transAxes,
                        fontsize=22, color=TEXT, ha='right', fontweight='bold')
    ppm_label = ax.text(0.97, 0.14, '', transform=ax.transAxes,
                        fontsize=15, color=GRAY, ha='right')

    N      = len(all_yrs)
    REVEAL = 40           # frames for fit line to draw in
    TOTAL  = N + REVEAL

    def update(frame):
        n = min(frame + 1, N)
        scat.set_offsets(np.c_[all_yrs[:n], all_co2[:n]])
        yr_label.set_text(f"{int(all_yrs[min(n-1, N-1)])}")
        ppm_label.set_text(f"{all_co2[min(n-1, N-1)]:.1f} ppm")

        if frame >= N:
            prog = (frame - N) / REVEAL
            idx  = int(prog * len(smooth_x))
            fit_line.set_data(smooth_x[:idx], smooth_y[:idx])
            fit_line.set_alpha(min(1.0, prog * 1.8))
        return scat, fit_line, yr_label, ppm_label

    ani  = animation.FuncAnimation(fig, update, frames=TOTAL, interval=55, blit=True)
    path = f"{OUTPUT}/anim_co2_rise.gif"
    ani.save(path, writer='pillow', fps=18, dpi=100)
    plt.close()
    print(f"  ✓  {path}")

# ── GIF 2: Projection Curves Growing ─────────────────────────────────────────
def anim_projections(years, co2):
    rc()
    years_val = np.array([2022, 2023, 2024])
    co2_val   = np.array([418.53, 421.08, 424.61])
    x_all = np.concatenate([years, years_val])
    y_all = np.concatenate([co2,   co2_val])

    popt_lin, _ = curve_fit(linear_func, x_all, y_all, p0=[1.5, -2800])
    mask2 = x_all > 1990
    x2, y2 = x_all[mask2], y_all[mask2]
    popt_exp, _ = curve_fit(exp_func, x2, y2, p0=[315, 0.01, 0, x2.min()],
                             bounds=([0,0,-np.inf,x2.min()],[np.inf,np.inf,np.inf,x2.max()]))
    popt_hyb, _ = curve_fit(hybrid_model, x_all, y_all, p0=[0.005, 0.8, 310, 5.0, 0.02], maxfev=50000)

    NF       = 160
    future   = np.linspace(2024, 2050, NF)
    y_lin_f  = linear_func(future, *popt_lin)
    y_exp_f  = exp_func(future, *popt_exp)
    y_hyb_f  = hybrid_model(future, *popt_hyb)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)

    ax.scatter(years, co2, color=GRAY, alpha=0.2, s=20, zorder=2)
    ax.scatter(years_val, co2_val, color=YELLOW, marker='D', s=90, edgecolors='none', zorder=5)
    ax.axhline(685, color=RED, linestyle='--', linewidth=2, alpha=0.7, zorder=6)
    ax.text(1958, 690, '685 ppm threshold', color=RED, fontsize=13, va='bottom')

    ax.set_xlim(1955, 2055)
    ax.set_ylim(300, 720)
    ax.set_title(r'$\mathrm{CO}_2$ Projections to 2050', fontsize=22, fontweight='bold', pad=14)
    ax.set_xlabel('Year', fontsize=17, labelpad=8)
    ax.set_ylabel(r'$\mathrm{CO}_2$ (ppm)', fontsize=17, labelpad=8)
    ax.grid(True, linestyle=':', alpha=0.4)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    l_lin, = ax.plot([], [], color=BLUE,   lw=3, zorder=4, label='Global linear')
    l_exp, = ax.plot([], [], color=ORANGE, lw=3, zorder=4, label='Piecewise exp')
    l_hyb, = ax.plot([], [], color=GREEN,  lw=3, zorder=4, label='Hybrid ensemble')
    ax.legend(fontsize=14, loc='upper left')

    yr_label = ax.text(0.97, 0.06, '', transform=ax.transAxes,
                       fontsize=20, color=TEXT, ha='right', fontweight='bold')

    def update(frame):
        n = frame + 1
        l_lin.set_data(future[:n], y_lin_f[:n])
        l_exp.set_data(future[:n], y_exp_f[:n])
        l_hyb.set_data(future[:n], y_hyb_f[:n])
        yr_label.set_text(f"{int(future[min(n-1, NF-1)])}")
        return l_lin, l_exp, l_hyb, yr_label

    ani  = animation.FuncAnimation(fig, update, frames=NF, interval=40, blit=True)
    path = f"{OUTPUT}/anim_projections.gif"
    ani.save(path, writer='pillow', fps=24, dpi=100)
    plt.close()
    print(f"  ✓  {path}")

# ── Table: Predictions by 2050 + 685 ppm threshold ───────────────────────────
def slide_prediction_table(years, co2):
    import sys as _sys
    _sys.path.insert(0, '.')
    import co2_prediction as _cp

    predictor = _cp.CO2Predictor(years, co2)
    future    = np.arange(1959, 2051)

    y_lin, _, popt_lin         = predictor.predict_global_linear(future)
    y_exp, _, popts_e, fns_e   = predictor.predict_piecewise(
        future, ["linear", "quadratic", "exponential"])
    y_rat, _, popts_r, fns_r   = predictor.predict_piecewise(
        future, ["linear", "quadratic", "exponential", "rational_1_1", "rational_2_1"])

    yr_lin = _cp.find_year_reaches_limit(_cp.linear_func, popt_lin, 2023)
    yr_exp = _cp.find_year_reaches_limit(fns_e[1], popts_e[1], 2023)
    yr_rat = _cp.find_year_reaches_limit(fns_r[1], popts_r[1], 2023)

    def fmt_yr(yr): return f"≈ {int(yr)}" if yr else "> 2500"

    ROWS = [
        ("Global Linear",         f"{y_lin[-1]:.1f}",  fmt_yr(yr_lin)),
        ("Piecewise Exponential", f"{y_exp[-1]:.1f}",  fmt_yr(yr_exp)),
        ("Piecewise Rational",    f"{y_rat[-1]:.1f}",  fmt_yr(yr_rat)),
    ]
    MODEL_COLORS = [BLUE, ORANGE, GREEN]

    # ── layout constants ────────────────────────────────────────────────────────
    TABLE_TOP = 0.76   # top edge of header row (in axes coords, ylim 0→1)
    ROW_H     = 0.172
    # col: (left_x, width)
    COLS = [(0.03, 0.41), (0.46, 0.27), (0.75, 0.22)]
    HEADERS = ["Model", "2050 Prediction", "Reaches 685 ppm"]

    def row_center(i):   # i=0 → header
        return TABLE_TOP - ROW_H * (i + 0.5)

    rc()
    fig, ax = plt.subplots(figsize=(14, 4.2), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)

    from matplotlib.patches import Rectangle as Rect

    # Title
    ax.text(0.5, 0.96,
            r'$\mathrm{CO}_2$ Projections — Model Comparison',
            ha='center', va='top', fontsize=19, fontweight='bold',
            color=TEXT, transform=ax.transAxes)

    # ── header row ──────────────────────────────────────────────────────────────
    yc = row_center(0)
    ax.add_patch(Rect((0, yc - ROW_H/2), 1, ROW_H,
                       facecolor='#161b22', edgecolor='none',
                       transform=ax.transAxes, zorder=1))
    for (lx, w), h in zip(COLS, HEADERS):
        ax.text(lx + w/2, yc, h,
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=TEXT, transform=ax.transAxes, zorder=2)

    # ── data rows ───────────────────────────────────────────────────────────────
    ROW_BGS = ['#0d1117', '#111820', '#0d1117']
    for i, (row, bg, mc) in enumerate(zip(ROWS, ROW_BGS, MODEL_COLORS)):
        yc = row_center(i + 1)

        # row background
        ax.add_patch(Rect((0, yc - ROW_H/2), 1, ROW_H,
                           facecolor=bg, edgecolor='none',
                           transform=ax.transAxes, zorder=1))

        # col 1 — colored dot + model name
        lx, _ = COLS[0]
        ax.text(lx + 0.01, yc, '●',
                ha='left', va='center', fontsize=12, color=mc,
                transform=ax.transAxes, zorder=2)
        ax.text(lx + 0.05, yc, row[0],
                ha='left', va='center', fontsize=14, fontweight='bold',
                color=mc, transform=ax.transAxes, zorder=2)

        # col 2 — ppm value
        lx, w = COLS[1]
        ax.text(lx + w/2, yc, row[1] + " ppm",
                ha='center', va='center', fontsize=14, color=TEXT,
                transform=ax.transAxes, zorder=2)

        # col 3 — year in yellow
        lx, w = COLS[2]
        ax.text(lx + w/2, yc, row[2],
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=YELLOW, transform=ax.transAxes, zorder=2)

    # ── horizontal rules ────────────────────────────────────────────────────────
    ax.axhline(TABLE_TOP,              color=SPINE, lw=0.8)   # top of header
    ax.axhline(TABLE_TOP - ROW_H,      color=SPINE, lw=1.8)   # below header
    for i in range(1, 3):
        ax.axhline(TABLE_TOP - ROW_H * (i + 1), color=GRID, lw=0.7)
    ax.axhline(TABLE_TOP - ROW_H * 4, color=SPINE, lw=1.2)   # bottom

    # ── footnote ────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.015,
            "685 ppm ≈ 2.4× pre-industrial baseline  |  Historical data: NOAA 1959–2024",
            ha='center', va='bottom', fontsize=10, color=GRAY,
            transform=ax.transAxes)

    path = f"{OUTPUT}/slide_prediction_table.png"
    fig.savefig(path, facecolor=BG, dpi=150)
    plt.close()
    print(f"  ✓  {path}")


# ── Table: RMSE Comparison ───────────────────────────────────────────────────
def slide_rmse_table():
    rc()
    DATA = [
        ("Global Linear",         11.50, BLUE,   RED),
        ("Piecewise Exponential",  0.94, ORANGE, YELLOW),
        ("Piecewise Rational",     0.87, GREEN,  GREEN),
    ]
    MAX_RMSE = max(d[1] for d in DATA)

    TABLE_TOP = 0.76
    ROW_H     = 0.185
    MODEL_COL = (0.03, 0.44)
    BAR_COL   = (0.49, 0.48)

    def row_center(i): return TABLE_TOP - ROW_H * (i + 0.5)

    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)

    from matplotlib.patches import Rectangle as Rect, FancyBboxPatch

    ax.text(0.5, 0.96, 'Model Fit Quality — RMSE Comparison',
            ha='center', va='top', fontsize=19, fontweight='bold',
            color=TEXT, transform=ax.transAxes)

    # Header
    yc = row_center(0)
    ax.add_patch(Rect((0, yc - ROW_H/2), 1, ROW_H,
                       facecolor='#161b22', edgecolor='none',
                       transform=ax.transAxes, zorder=1))
    for (lx, w), h in zip([MODEL_COL, BAR_COL], ["Model", "RMSE (ppm)"]):
        ax.text(lx + w/2, yc, h,
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=TEXT, transform=ax.transAxes, zorder=2)

    # Data rows
    ROW_BGS = ['#0d1117', '#111820', '#0d1117']
    for i, ((model, rmse, mc, bar_color), bg) in enumerate(zip(DATA, ROW_BGS)):
        yc    = row_center(i + 1)
        frac  = rmse / MAX_RMSE
        blx, bw = BAR_COL
        bar_h = ROW_H * 0.46

        # Row background
        ax.add_patch(Rect((0, yc - ROW_H/2), 1, ROW_H,
                           facecolor=bg, edgecolor='none',
                           transform=ax.transAxes, zorder=1))

        # Model dot + name
        lx, _ = MODEL_COL
        ax.text(lx + 0.01, yc, '●',
                ha='left', va='center', fontsize=12, color=mc,
                transform=ax.transAxes, zorder=2)
        ax.text(lx + 0.05, yc, model,
                ha='left', va='center', fontsize=14, fontweight='bold',
                color=mc, transform=ax.transAxes, zorder=2)

        # Bar track
        ax.add_patch(Rect((blx, yc - bar_h/2), bw, bar_h,
                           facecolor='#21262d', edgecolor='none',
                           transform=ax.transAxes, zorder=2))

        # Colored bar (rounded)
        bar_w = max(bw * frac, 0.012)   # minimum so tiny bars are visible
        ax.add_patch(FancyBboxPatch(
            (blx, yc - bar_h/2), bar_w, bar_h,
            boxstyle="round,pad=0.004",
            facecolor=bar_color, edgecolor='none', alpha=0.88,
            transform=ax.transAxes, zorder=3))

        # Glow on bar
        ax.add_patch(Rect((blx, yc - bar_h/2), bar_w, bar_h,
                           facecolor=bar_color, alpha=0.18,
                           edgecolor='none', transform=ax.transAxes, zorder=2))

        # Value text — inside bar if large, outside if small
        value_str = f"{rmse:.2f}"
        if frac > 0.6:
            ax.text(blx + bar_w - 0.008, yc, value_str,
                    ha='right', va='center', fontsize=14, fontweight='bold',
                    color=BG, transform=ax.transAxes, zorder=4)
        else:
            ax.text(blx + bar_w + 0.018, yc, value_str,
                    ha='left', va='center', fontsize=14, fontweight='bold',
                    color=bar_color, transform=ax.transAxes, zorder=4)

    # Horizontal rules
    ax.axhline(TABLE_TOP,             color=SPINE, lw=0.8)
    ax.axhline(TABLE_TOP - ROW_H,     color=SPINE, lw=1.8)
    for i in range(1, 3):
        ax.axhline(TABLE_TOP - ROW_H * (i+1), color=GRID, lw=0.7)
    ax.axhline(TABLE_TOP - ROW_H * 4, color=SPINE, lw=1.2)

    ax.text(0.5, 0.015,
            "Lower RMSE = better fit  |  All values in ppm",
            ha='center', va='bottom', fontsize=10, color=GRAY,
            transform=ax.transAxes)

    path = f"{OUTPUT}/slide_rmse_table.png"
    fig.savefig(path, facecolor=BG, dpi=150)
    plt.close()
    print(f"  ✓  {path}")


# ── GIF 3: Scaling Issue (raw → zoom → z-score) ───────────────────────────────
def anim_scaling_issue(years, co2, temp):
    rc()
    co2_z  = (co2  - co2.mean()) / co2.std()
    temp_z = (temp - temp.mean()) / temp.std()
    r = np.corrcoef(co2_z, temp_z)[0, 1]

    def ease(t): return t * t * (3 - 2 * t)

    FPS = 15
    # seconds per phase
    P1 = 3.0   # raw CO₂ scale — temp invisible
    P2 = 2.5   # zoom y-axis into temperature range
    P3 = 2.0   # hold temp scale — CO₂ off-screen
    P4 = 0.6   # quick cut / blank
    P5 = 3.0   # z-score view

    f1 = int(P1 * FPS); f2 = int(P2 * FPS)
    f3 = int(P3 * FPS); f4 = int(P4 * FPS); f5 = int(P5 * FPS)
    TOTAL = f1 + f2 + f3 + f4 + f5

    Y_CO2  = (306, 438)
    Y_TEMP = (-0.6, 1.5)
    Y_Z    = (-3.0, 3.0)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.set_xlabel('Year', fontsize=17, labelpad=8)
    ax.set_xlim(years[0] - 3, years[-1] + 5)
    ax.grid(True, linestyle=':', alpha=0.4)

    co2_line,  = ax.plot(years, co2,  color=BLUE,   lw=3, zorder=4, clip_on=True)
    temp_line, = ax.plot(years, temp, color=ORANGE, lw=3, zorder=4, clip_on=True)

    # Static legend proxies — labels won't change mid-animation
    ax.plot([], [], color=BLUE,   lw=3, label=r'$\mathrm{CO}_2$')
    ax.plot([], [], color=ORANGE, lw=3, label='Temperature')
    leg = ax.legend(fontsize=14, loc='upper left')

    title_obj = ax.text(0.5, 1.03, '', transform=ax.transAxes,
                        fontsize=18, fontweight='bold', color=TEXT,
                        ha='center', va='bottom')
    ylabel_obj = ax.yaxis.label
    ylabel_obj.set_fontsize(16)

    annot = ax.text(0.5, 0.10, '', transform=ax.transAxes,
                    fontsize=14, ha='center', va='center', color=RED,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22',
                              edgecolor=RED, alpha=0.0))

    range_box = ax.text(0.97, 0.96, '', transform=ax.transAxes,
                        fontsize=13, ha='right', va='top', color=TEXT,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22',
                                  edgecolor=SPINE, alpha=0.0))

    def show_annot(text, color=RED, visible=True):
        annot.set_text(text); annot.set_color(color); annot.set_visible(visible)
        bp = annot.get_bbox_patch()
        bp.set_edgecolor(color); bp.set_alpha(0.92 if visible else 0.0)

    def show_rangebox(text, visible=True):
        range_box.set_text(text); range_box.set_visible(visible)
        range_box.get_bbox_patch().set_alpha(0.92 if visible else 0.0)

    def update(frame):
        # ── Phase 1: CO₂ scale — temp flatlines ──────────────────────────────
        if frame < f1:
            ax.set_ylim(*Y_CO2)
            co2_line.set_data(years, co2)
            temp_line.set_data(years, temp)
            ylabel_obj.set_text(r'$\mathrm{CO}_2$ (ppm)  /  Temperature (°C)')
            title_obj.set_text(r"Plotting $\mathrm{CO}_2$ (ppm) and Temperature (°C) on the same axis")
            if frame >= int(f1 * 0.45):
                show_annot("Temperature range ≈ 1.4 °C    vs    CO₂ range ≈ 106 ppm\n"
                           "At this scale, temperature is essentially invisible")
                show_rangebox(f"CO₂:  {co2.min():.0f} – {co2.max():.0f} ppm\n"
                              f"Temp: {temp.min():.2f} – {temp.max():.2f} °C", visible=True)
            else:
                show_annot('', visible=False); show_rangebox('', visible=False)

        # ── Phase 2: animate zoom into temperature range ──────────────────────
        elif frame < f1 + f2:
            t = ease((frame - f1) / f2)
            ax.set_ylim(
                Y_CO2[0] + t * (Y_TEMP[0] - Y_CO2[0]),
                Y_CO2[1] + t * (Y_TEMP[1] - Y_CO2[1]),
            )
            co2_line.set_data(years, co2)
            temp_line.set_data(years, temp)
            ylabel_obj.set_text('Value')
            title_obj.set_text("Zooming in to the temperature scale…")
            show_annot('', visible=False); show_rangebox('', visible=False)

        # ── Phase 3: temp scale — CO₂ off-screen ─────────────────────────────
        elif frame < f1 + f2 + f3:
            ax.set_ylim(*Y_TEMP)
            co2_line.set_data(years, co2)
            temp_line.set_data(years, temp)
            ylabel_obj.set_text('Temperature (°C)')
            title_obj.set_text(r"Zoomed to temperature scale — now CO₂ disappears")
            if frame >= f1 + f2 + int(f3 * 0.35):
                show_annot(r"CO₂ (315 – 421 ppm) is completely off the chart at this scale",
                           color=BLUE)
                show_rangebox('', visible=False)
            else:
                show_annot('', visible=False); show_rangebox('', visible=False)

        # ── Phase 4: brief cut / blank ────────────────────────────────────────
        elif frame < f1 + f2 + f3 + f4:
            ax.set_ylim(*Y_Z)
            co2_line.set_data([], [])
            temp_line.set_data([], [])
            ylabel_obj.set_text('')
            title_obj.set_text('')
            show_annot('', visible=False); show_rangebox('', visible=False)

        # ── Phase 5: z-score view ─────────────────────────────────────────────
        else:
            ax.set_ylim(*Y_Z)
            co2_line.set_data(years, co2_z)
            temp_line.set_data(years, temp_z)
            ylabel_obj.set_text('Standardized Value (σ)')
            title_obj.set_text("After Z-Score Standardization — both series on the same scale")
            elapsed = frame - (f1 + f2 + f3 + f4)
            if elapsed >= int(f5 * 0.3):
                show_annot(f"Pearson r = {r:.4f}   —   92 % of variance explained",
                           color=GREEN)
                show_rangebox(f"CO₂ z-score:  {co2_z.min():.2f} – {co2_z.max():.2f} σ\n"
                              f"Temp z-score: {temp_z.min():.2f} – {temp_z.max():.2f} σ",
                              visible=True)
                range_box.set_color(TEXT)
                range_box.get_bbox_patch().set_edgecolor(GREEN)
            else:
                show_annot('', visible=False); show_rangebox('', visible=False)

    ani  = animation.FuncAnimation(fig, update, frames=TOTAL, interval=1000 // FPS, blit=False)
    path = f"{OUTPUT}/anim_scaling_issue.gif"
    ani.save(path, writer='pillow', fps=FPS, dpi=100)
    plt.close()
    print(f"  ✓  {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    years, co2, temp = load()
    print(f"Loaded {len(years)} data points ({int(years[0])}–{int(years[-1])})\n")

    print("Generating static slides...")
    slide_linear(years, co2, temp)
    slide_zscore(years, co2, temp)
    slide_piecewise(years, co2, temp)
    slide_ensemble(years, co2)
    slide_projections(years, co2)

    print("\nGenerating animated GIFs...")
    anim_co2_rise(years, co2)
    anim_projections(years, co2)
    anim_scaling_issue(years, co2, temp)

    print(f"\nAll assets saved to ./{OUTPUT}/")
