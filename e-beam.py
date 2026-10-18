import numpy as np
import matplotlib.pyplot as plt

# Settings
U_a = 5.0            # accelerating voltage [kV]
dU_p = 0.1           # uncertainty plate voltage [kV]
dy = 0.05            # readout uncertainty y [cm]
x = np.arange(3, 11) # positions where y is read [cm]

# Measurements: plate voltage [kV] and deflection y [cm] at x = 3..10 cm
data = {
    "Measurement 2": (1.9, [-0.01, 0.06, 0.15, 0.25, 0.42, 0.59, 0.78, 0.92]),
    "Measurement 3": (3.0, [0.02, 0.12, 0.27, 0.46, 0.70, 0.89, 1.19, 1.45]),
    "Measurement 4": (4.6, [0.07, 0.22, 0.45, 0.75, 1.11, 1.54, 2.02, 2.39]),
    "Measurement 5": (3.1, [0.02, 0.13, 0.28, 0.48, 0.72, 0.99, 1.31, 1.55]),
}

colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
markers = ["o", "s", "^", "D"]

fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)

X_all, y_all = [], []
for (name, (U_p, y)), c, m in zip(data.items(), colors, markers):
    y = np.array(y)
    X = U_p * x**2 / (4 * U_a)        # X = U_p x^2 / (4 U_a)  [cm^2]
    dX = dU_p * x**2 / (4 * U_a)
    ax.errorbar(X, y, xerr=dX, yerr=dy, fmt=m, ms=5.5, color=c, mec="white",
                mew=0.8, elinewidth=0.9, capsize=2,
                label=f"{name}, $U_p$ = {U_p} kV", zorder=3)
    X_all += list(X)
    y_all += list(y)

X_all = np.array(X_all)
y_all = np.array(y_all)

# Least-squares fit y = s*X through the origin, s = 1/d
s = np.sum(X_all * y_all) / np.sum(X_all**2)
res = y_all - s * X_all
ds = np.sqrt(np.sum(res**2) / (len(X_all) - 1) / np.sum(X_all**2))
d = 1 / s

# Total uncertainty: fit + U_a (0.1/5) + U_p (about 3.3 %)
rel = np.sqrt((ds / s)**2 + (0.1 / U_a)**2 + 0.033**2)
print(f"slope 1/d = {s:.4f} +/- {ds:.4f} 1/cm")
print(f"d = {d:.2f} +/- {d * rel:.2f} cm")

t = np.linspace(0, 25, 50)
ax.plot(t, s * t, color="#1f1f1e", lw=1.5, zorder=2,
        label=f"fit $y = X/d$,  $d$ = {d:.1f} cm")

ax.set_xlabel(r"$X = U_p\,x^2/(4U_a)$  [cm$^2$]")
ax.set_ylabel(r"deflection $y$  [cm]")
ax.set_xlim(0, 24.5)
ax.set_ylim(-0.15, 2.6)
ax.grid(True, color="#e6e5e0", lw=0.7)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="in", length=4)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")

fig.tight_layout()
fig.savefig("A7_plate_separation.png")
plt.show()