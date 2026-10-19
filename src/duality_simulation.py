"""
EN3330 Assignment 03 - Lagrangian duality for  min x^2  s.t.  x >= 1.

Generates every figure used in the report (saved as vector PDFs in ../figures)
and prints numerical checks of the analytical results (saved in ../results).

Run:  python3 src/duality_simulation.py
Requires: numpy, matplotlib
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
FEASIBLE = "#1baf7a"
INFEASIBLE = "#e34948"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 10.5,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


# -------------------------------------------------------------- problem ----
def f(x):
    """Objective f(x) = x^2."""
    return x ** 2


def h(x, b=1.0):
    """Constraint h(x) = b - x <= 0  (i.e. x >= b)."""
    return b - x


def L(mu, x, b=1.0):
    """Lagrangian L(mu, x) = f(x) + mu h(x)."""
    return f(x) + mu * h(x, b)


def x_of_mu(mu):
    """Analytical minimiser of L(mu, .): dL/dx = 2x - mu = 0."""
    return mu / 2.0


def d_analytic(mu):
    """Dual function d(mu) = mu - mu^2/4."""
    return mu - mu ** 2 / 4.0


X_GRID = np.linspace(-10.0, 10.0, 400_001)  # fine grid for numerical minimisation


def d_numeric(mu, b=1.0):
    vals = L(mu, X_GRID, b)
    i = np.argmin(vals)
    return vals[i], X_GRID[i]


log_lines = []


def log(s=""):
    print(s)
    log_lines.append(s)


# =================================================================== Q1(a) ==
mus = [0, 1, 2, 3]
x = np.linspace(-1.5, 3.0, 600)
fig, ax = plt.subplots(figsize=(6.0, 3.9))
ax.axvspan(-1.5, 1.0, color=INFEASIBLE, alpha=0.06, lw=0)
ax.axvline(1.0, color=MUTED, lw=1.0, ls="--")
ax.text(0.97, 9.3, "infeasible  $x<1$", ha="right", color=INK_2, fontsize=8.5)
ax.text(1.05, 9.3, "feasible  $x\\geq 1$", ha="left", color=INK_2, fontsize=8.5)
for k, mu in enumerate(mus):
    c = SERIES[k]
    ax.plot(x, L(mu, x), color=c, label=f"$\\mu={mu}$")
    xm = x_of_mu(mu)
    ax.plot(xm, L(mu, xm), "o", ms=7, color=c, mec="white", mew=1.5, zorder=5)
    off = [(0, -15), (-16, -15), (8, -17), (16, -15)][k]
    ax.annotate(f"$x({mu})={xm:g}$", (xm, L(mu, xm)), textcoords="offset points",
                xytext=off, ha="center", fontsize=8.5, color=INK)
# trajectory of minimisers: (mu/2, mu - mu^2/4) lies on the curve y = 2x - x^2
mt = np.linspace(0, 3, 100)
ax.plot(x_of_mu(mt), d_analytic(mt), color=INK_2, lw=1.0, ls=":",
        label="path of $x(\\mu)$")
ax.set_xlim(-1.5, 3.0)
ax.set_ylim(-1.0, 10.0)
ax.set_xlabel("$x$")
ax.set_ylabel("$L(\\mu,x)=x^2+\\mu(1-x)$")
ax.legend(loc="upper left", ncol=1)
save(fig, "q1a_lagrangian_vs_x")

log("Q1(a): minimisers x(mu) = mu/2")
for mu in mus:
    Ln, xn = d_numeric(mu)
    log(f"  mu={mu}:  x(mu) analytic={x_of_mu(mu):.4f}  numeric={xn:.4f}   "
        f"min L analytic={d_analytic(mu):.4f}  numeric={Ln:.4f}")

# =================================================================== Q1(b) ==
mu = np.linspace(0, 6, 400)
fig, ax = plt.subplots(figsize=(6.0, 3.5))
ax.axhline(1.0, color=MUTED, lw=1.0, ls="--")
ax.text(5.95, 1.07, "$p^*=f(x^*)=1$", ha="right", color=INK_2, fontsize=8.5)
ax.plot(mu, d_analytic(mu), color=SERIES[0], label="$d(\\mu)=\\mu-\\mu^2/4$ (analytic)")
mu_s = np.arange(0, 6.01, 0.5)
ax.plot(mu_s, [d_numeric(m)[0] for m in mu_s], "s", ms=5, color=SERIES[1],
        mec="white", mew=1.0, label="numerical minimum of $L(\\mu,\\cdot)$ on a grid")
ax.plot(2, 1, "o", ms=9, color=INK, mec="white", mew=1.5, zorder=6)
ax.annotate("$\\mu^*=2,\\ d^*=1$", (2, 1), xytext=(2.5, 1.3),
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8), fontsize=9)
ax.set_xlim(0, 6)
ax.set_ylim(-3.2, 1.6)
ax.set_xlabel("$\\mu$")
ax.set_ylabel("$d(\\mu)$")
ax.legend(loc="lower left")
save(fig, "q1b_dual_function")

mu_dense = np.linspace(0, 10, 100_001)
i = np.argmax(d_analytic(mu_dense))
log(f"\nQ1(b): mu* (grid search) = {mu_dense[i]:.4f}, d* = {d_analytic(mu_dense)[i]:.4f}")

# =================================================================== Q1(c) ==
# g(x) = sup_{mu>=0} L(mu,x). Approximate the sup by mu in [0, M] for growing M.
xg = np.linspace(-1, 3, 801)
fig, ax = plt.subplots(figsize=(6.0, 3.9))
ax.axvspan(-1, 1, color=INFEASIBLE, alpha=0.06, lw=0)
ax.axvline(1.0, color=MUTED, lw=1.0, ls="--")
for k, M in enumerate([0, 2, 5, 10, 20]):
    gM = np.max([L(m, xg) for m in np.linspace(0, M, 201)], axis=0)
    ax.plot(xg, gM, color=SERIES[k], lw=1.4, alpha=0.9, label=f"$M={M}$")
# exact primal function
xf = xg[xg >= 1]
ax.plot(xf, f(xf), color=INK, lw=2.6, label="$g(x)=x^2$ for $x\\geq1$")
ax.plot(1, 1, "o", ms=8, color=INK, mec="white", mew=1.5, zorder=6)
ax.annotate("$\\min g = g(1)=1$", (1, 1), xytext=(1.6, 0.0),
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8), fontsize=9)
for xa in [-0.75, -0.25, 0.25, 0.75]:
    ax.annotate("", xy=(xa, 21.5), xytext=(xa, 15.5),
                arrowprops=dict(arrowstyle="-|>", color=INFEASIBLE, lw=1.2))
ax.text(0.0, 13.4, "$g(x)=+\\infty$ for $x<1$", ha="center", color=INFEASIBLE, fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"), zorder=7)
ax.set_xlim(-1, 3)
ax.set_ylim(-1.5, 22)
ax.set_xlabel("$x$")
ax.set_ylabel("$g(x)$")
ax.legend(loc="upper right", fontsize=8,
          title="$\\sup$ of $L(\\mu,x)$ over $\\mu\\in[0,M]$", title_fontsize=8)
save(fig, "q1c_primal_function")

log("\nQ1(c): truncated sup over mu in [0,M] at sample x")
log("   x    " + "".join(f"M={M:<9}" for M in [1, 10, 100, 1000]) + "x^2")
for xv in [-1.0, 0.0, 0.5, 0.99, 1.0, 2.0, 3.0]:
    row = "".join(f"{max(L(0, xv), L(M, xv)):<11.3f}" for M in [1, 10, 100, 1000])
    log(f"  {xv:5.2f}  {row}{f(xv):.3f}")

# =================================================================== Q2(a) ==
x_star, mu_star = 1.0, 2.0
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
ax = axes[0]
mm = np.linspace(0, 6, 100)
ax.plot(mm, L(mm, x_star), color=SERIES[0])
ax.plot(mu_star, L(mu_star, x_star), "o", ms=8, color=INK, mec="white", mew=1.5, zorder=5)
ax.annotate("$L(\\mu^*,x^*)=1$", (mu_star, 1), xytext=(2.5, 1.6),
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8), fontsize=9)
ax.set_xlim(0, 6)
ax.set_ylim(0, 3)
ax.set_xlabel("$\\mu$")
ax.set_ylabel("$L(\\mu,x^*)$")
ax.set_title("$L(\\mu,1)=1$ for all $\\mu\\geq 0$")
ax = axes[1]
xx = np.linspace(-1, 3, 300)
ax.plot(xx, L(mu_star, xx), color=SERIES[1])
ax.plot(x_star, L(mu_star, x_star), "o", ms=8, color=INK, mec="white", mew=1.5, zorder=5)
ax.axhline(1, color=MUTED, lw=1.0, ls="--")
ax.annotate("$L(\\mu^*,x^*)=1$", (1, 1), xytext=(1.5, 3.0),
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8), fontsize=9)
ax.set_xlim(-1, 3)
ax.set_ylim(0, 5.5)
ax.set_xlabel("$x$")
ax.set_ylabel("$L(\\mu^*,x)$")
ax.set_title("$L(2,x)=(x-1)^2+1$")
fig.tight_layout(w_pad=2.0)
save(fig, "q2a_saddle_slices")

# 3-D style contour view of the saddle point
MU, XX = np.meshgrid(np.linspace(0, 4, 200), np.linspace(-0.5, 2.5, 200))
fig, ax = plt.subplots(figsize=(5.2, 3.6))
cs = ax.contour(MU, XX, L(MU, XX), levels=np.arange(-2, 7, 0.5), cmap="Blues_r", linewidths=1.0)
ax.clabel(cs, cs.levels[::2], fontsize=7, fmt="%g")
ax.plot(mt := np.linspace(0, 4, 50), x_of_mu(mt), color=SERIES[1], lw=1.4,
        label="$x(\\mu)=\\mu/2$ (minimiser of $L$)")
ax.axhline(1, color=SERIES[2], lw=1.4, label="$x=x^*=1$")
ax.plot(mu_star, x_star, "o", ms=9, color=INK, mec="white", mew=1.5, zorder=6)
ax.annotate("saddle $(\\mu^*,x^*)=(2,1)$", (2, 1), xytext=(2.3, 0.0),
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8), fontsize=9)
ax.set_xlabel("$\\mu$")
ax.set_ylabel("$x$")
ax.legend(loc="upper left", fontsize=8)
ax.grid(False)
save(fig, "q2a_saddle_contour")

# numerical saddle check on a grid
mu_chk = np.linspace(0, 50, 5001)
x_chk = np.linspace(-20, 20, 40001)
left_ok = np.all(L(mu_chk, x_star) <= L(mu_star, x_star) + 1e-12)
right_ok = np.all(L(mu_star, x_star) <= L(mu_star, x_chk) + 1e-12)
log(f"\nQ2(a): L(mu*,x*) = {L(mu_star, x_star):.4f}")
log(f"  max_mu L(mu,x*) over mu in [0,50]   = {L(mu_chk, x_star).max():.4f}  (left ineq holds: {left_ok})")
log(f"  min_x  L(mu*,x) over x in [-20,20]  = {L(mu_star, x_chk).min():.4f}  (right ineq holds: {right_ok})")
log(f"  max |L(2,x) - ((x-1)^2+1)|           = {np.abs(L(2, x_chk) - ((x_chk - 1) ** 2 + 1)).max():.2e}")

# =================================================================== Q2(c) ==
# modified constraint x >= -1 :  h(x) = -1 - x
b2 = -1.0
mu = np.linspace(0, 6, 400)
fig, ax = plt.subplots(figsize=(6.0, 3.3))
ax.plot(mu, d_analytic(mu), color=SERIES[0], label="$x\\geq 1$:  $d(\\mu)=\\mu-\\mu^2/4$")
ax.plot(mu, -mu - mu ** 2 / 4, color=SERIES[1], label="$x\\geq -1$:  $d(\\mu)=-\\mu-\\mu^2/4$")
ax.plot(mu_s, [d_numeric(m, b2)[0] for m in mu_s], "s", ms=5, color=SERIES[1],
        mec="white", mew=1.0, label="numerical, $x\\geq -1$")
ax.plot(2, 1, "o", ms=8, color=INK, mec="white", mew=1.5, zorder=6)
ax.plot(0, 0, "o", ms=8, color=INK, mec="white", mew=1.5, zorder=6)
ax.annotate("$\\mu^*=2$ (active)", (2, 1), xytext=(2.6, 0.2),
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8), fontsize=9)
ax.annotate("$\\mu^*=0$ (inactive)", (0, 0), xytext=(0.5, -3.5),
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8), fontsize=9)
ax.set_xlim(0, 6)
ax.set_ylim(-9, 2)
ax.set_xlabel("$\\mu$")
ax.set_ylabel("$d(\\mu)$")
ax.legend(loc="lower left", fontsize=8.5)
save(fig, "q2c_modified_dual")

mu_dense = np.linspace(0, 10, 100_001)
d2 = np.array([-m - m ** 2 / 4 for m in mu_dense])
log(f"\nQ2(c): x >= -1 ->  argmin x^2 numeric = {d_numeric(0, b2)[1]:.4f}, h(0) = {h(0, b2):.1f}")
log(f"  mu* (grid search) = {mu_dense[np.argmax(d2)]:.4f}, d* = {d2.max() + 0.0:.4f}")

# =================================================================== Q3(a) ==
log("\nQ3(a): weak duality check  d(mu) <= p* = 1")
for m in [0, 1, 2, 3]:
    log(f"  mu={m}: d(mu) = {d_analytic(m):.4f}   gap p*-d = {1 - d_analytic(m):.4f}   "
        f"holds: {d_analytic(m) <= 1}")
mu_dense = np.linspace(0, 100, 100_001)
log(f"  max over mu in [0,100] of d(mu) = {d_analytic(mu_dense).max():.4f} <= 1")

# =================================================================== Q3(b) ==
mu = np.linspace(0, 5, 400)
fig, ax = plt.subplots(figsize=(6.0, 3.8))
for xv in np.linspace(-1, 3, 33):  # many faint lines -> envelope converges to d
    ax.plot(mu, L(mu, xv), color=GRID, lw=0.8, zorder=1,
            label="other $x\\in[-1,3]$" if xv == -1 else None)
for k, xv in enumerate([0, 1, 2]):
    ax.plot(mu, L(mu, xv), color=SERIES[k], lw=1.6, zorder=3,
            label=f"$x={xv}$:  $L={['\\mu', '1', '4-\\mu'][k]}$")
env = np.min([L(mu, xv) for xv in [0, 1, 2]], axis=0)
ax.plot(mu, env, color=INK, lw=2.8, ls=(0, (4, 2)), zorder=4,
        label="lower envelope $\\min\\{\\mu,\\,1,\\,4-\\mu\\}$")
ax.plot(mu, d_analytic(mu), color=SERIES[4], lw=1.8, zorder=5, label="true $d(\\mu)$")
ax.set_xlim(0, 5)
ax.set_ylim(-1.5, 4.5)
ax.set_xlabel("$\\mu$")
ax.set_ylabel("$L(\\mu,x)$")
ax.legend(loc="upper right", fontsize=8.5)
save(fig, "q3b_lower_envelope")

env_check = np.min([L(mu, xv) for xv in [0, 1, 2]], axis=0)
log("\nQ3(b): envelope min(mu, 1, 4-mu) >= d(mu) everywhere: "
    f"{np.all(env_check >= d_analytic(mu) - 1e-12)}; touches d at mu = 0, 2, 4")
# midpoint concavity test of the envelope
a, c = np.meshgrid(mu, mu)
envf = lambda m: np.minimum(np.minimum(m, 1.0), 4.0 - m)
log("  envelope midpoint-concave on grid: "
    f"{np.all(envf((a + c) / 2) >= (envf(a) + envf(c)) / 2 - 1e-12)}")

(RES_DIR / "numerical_checks.txt").write_text("\n".join(log_lines) + "\n")
print(f"\nFigures written to {FIG_DIR}\nLog written to {RES_DIR / 'numerical_checks.txt'}")
