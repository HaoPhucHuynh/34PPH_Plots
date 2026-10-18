import numpy as np
import matplotlib.pyplot as plt

# Settings
U_acc = 5000         # accelerating voltage [V]
dU_acc = 100         # uncertainty accelerating voltage [V]
dU_plate = 100       # uncertainty plate voltage [V]
dy = 0.05            # readout uncertainty y [cm]
x = np.arange(3, 11) # positions where y is read [cm]

# Measurements: plate voltage [V] and deflection y [cm] at x = 3..10 cm
data = {
    "Measurement 2": (1900, [-0.01, 0.06, 0.15, 0.25, 0.42, 0.59, 0.78, 0.92]),
    "Measurement 3": (3000, [0.02, 0.12, 0.27, 0.46, 0.70, 0.89, 1.19, 1.45]),
    "Measurement 4": (4600, [0.07, 0.22, 0.45, 0.75, 1.11, 1.54, 2.02, 2.39]),
    "Measurement 5": (3100, [0.02, 0.13, 0.28, 0.48, 0.72, 0.99, 1.31, 1.55]),
}

colors = ["#1f1f1e", "#2a78d6", "#1baf7a", "#eda100"]   # black, blue, green, yellow
markers = ["s", "o", "^", "D"]
FIT = "red"          # fit line colour
FIT_LW = 0.5         # fit line width (thin)


def style(ax):
    """Closed box, ticks inside, no grid."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)
    ax.tick_params(direction="in", length=4, top=False, right=False)
    ax.grid(False)


def fit_table(ax, rows, bbox, col=(0.28, 0.72)):
    """Small table with fit results inside the plot. bbox = [x, y, width, height] in axes units."""
    tb = ax.table(cellText=rows, cellLoc="left", bbox=bbox, colWidths=list(col))
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#8a8a85")
        cell.set_linewidth(0.6)
        cell.set_facecolor("white")
        if c == 0:
            cell.set_text_props(fontweight="bold")
    return tb


# Put all points in one list
U = np.concatenate([[U_p] * len(x) for U_p, y in data.values()])
xx = np.tile(x, len(data))
yy = np.concatenate([y for U_p, y in data.values()])

# Least-squares fit y = s*X through the origin, s = 1/d
X_all = U * xx**2 / (4 * U_acc)
s = np.sum(X_all * yy) / np.sum(X_all**2)
res = yy - s * X_all
ds = np.sqrt(np.sum(res**2) / (len(X_all) - 1) / np.sum(X_all**2))
d = 1 / s

# Total uncertainty: fit + U_acc (100/5000) + U_plate (about 3.3 %)
dd = d * np.sqrt((ds / s)**2 + (dU_acc / U_acc)**2 + 0.033**2)
print(f"1/d = {s:.4f} +/- {ds:.4f} 1/cm,  d = {d:.2f} +/- {dd:.2f} cm")

fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
for (name, (U_p, y)), c, m in zip(data.items(), colors, markers):
    X = U_p * x**2 / (4 * U_acc)
    dX = dU_plate * x**2 / (4 * U_acc)
    ax.errorbar(X, y, xerr=dX, yerr=dy, fmt=m, ms=5, color=c, ecolor=c,
                elinewidth=0.8, capsize=2, zorder=3, label=f"{name}, $U_{{plate}}$ = {U_p} V")
t = np.linspace(0, 25, 50)
ax.plot(t, s * t, color=FIT, lw=FIT_LW, zorder=2)

ax.set_xlabel(r"$X = U_{plate}\,x^2/(4U_{acc})$ (cm$^2$)")
ax.set_ylabel(r"$y$ (cm)")
ax.set_xlim(0, 24.5)
ax.set_ylim(-0.15, 2.6)
style(ax)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fit_table(ax, [["Model", r"$y = X/d$"],
               ["1/d", f"({s:.4f} ± {ds:.4f}) cm$^{{-1}}$"],
               ["d", f"({d:.1f} ± {dd:.1f}) cm"]],
          bbox=[0.52, 0.05, 0.46, 0.20])

fig.tight_layout()
fig.savefig("A7_plate_separation.png")
plt.show()