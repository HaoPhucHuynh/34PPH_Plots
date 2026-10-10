import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# ---------- data ----------

DATA = Path(__file__).parent / "OES_data"
files = {"H1": DATA / "H1text.txt",
         "H2": DATA / "H2,20 reps.txt",
         "H4": DATA / "H4_100reps_text.txt",
         "H5": DATA / "H5_100reps_text.txt"}

def load(path):
    """AvaSpec export: wavelength ; sample ; dark ; reference ; sample-dark (comma decimals)."""
    wl, y = [], []
    for line in open(path, encoding="latin-1"):
        parts = line.replace(",", ".").split(";")
        if len(parts) != 5:
            continue
        try:
            wl.append(float(parts[0])); y.append(float(parts[4]))
        except ValueError:
            continue
    return np.array(wl), np.array(y)

# ---------- NIST (H I effective lines, Balmer n -> 2) and Table 1, spectrometer 12 ----------
#            lambda_nm  A_ik [1/s]  g_i  E_i [eV]   C_s
lines = {"Ha": (656.28, 4.4101e7, 18, 12.0875, 0.75),
         "Hb": (486.13, 8.4193e6, 32, 12.7485, 2.09),
         "Hg": (434.05, 2.5304e6, 50, 13.0545, 1.93),
         "Hd": (410.17, 9.7320e5, 72, 13.2207, 2.45),
         "He": (397.01, 4.3889e5, 98, 13.3209, 2.35)}

SHIFT = -0.25   # measured calibration offset of spectrometer 12 (lines appear ~0.25 nm too blue)
HALF = 1.2      # integration half-width [nm] -> 4-5 pixels per line
# background pixels (observed wavelength, nm) on each side, chosen to avoid neighbouring bands
bg = {"Ha": [652.50, 653.08, 659.43, 660.01],
      "Hb": [484.47, 487.41, 488.00],
      "Hg": [432.53, 435.49],
      "Hd": [408.24, 412.98, 413.58],
      "He": [392.81, 393.40, 398.15, 398.75]}

def line_area(wl, y, name):
    lam = lines[name][0] + SHIFT
    win = np.abs(wl - lam) <= HALF
    idx = [np.argmin(np.abs(wl - b)) for b in bg[name]]
    p = np.polyfit(wl[idx], y[idx], 1)               # linear background
    return np.sum(y[win] - np.polyval(p, wl[win])), y[win].max()

names = list(lines)
E = np.array([lines[n][3] for n in names])
area = {}; peak = {}
for run, path in files.items():
    wl, y = load(path)
    out = [line_area(wl, y, n) for n in names]
    area[run] = np.array([o[0] for o in out]); peak[run] = np.array([o[1] for o in out])

def yval(I):
    A = np.array([lines[n][1] for n in names]); g = np.array([lines[n][2] for n in names])
    Cs = np.array([lines[n][4] for n in names])
    return np.log(I / (Cs * A * g))

# ---------- print table ----------
print("Integrated intensity [counts], background-subtracted")
print("run  " + "".join(f"{n:>11}" for n in names) + "   Ha/Hb   Hg/Hb   Hd/Hb")
for run in files:
    a = area[run]
    print(f"{run:4} " + "".join(f"{v:11.0f}" for v in a) +
          f"  {a[0]/a[1]:6.2f}  {a[2]/a[1]:6.3f}  {a[3]/a[1]:6.4f}")

# ---------- per-run fits ----------
def fits(use):
    s = []
    for run in files:
        y = yval(area[run])
        s.append(np.polyfit(E[use], y[use], 1))
    return np.array(s)

use_noHa = np.array([n != "Ha" for n in names])
use_all = np.ones(len(names), bool)
for label, use in [("Hb-He (Ha excluded)", use_noHa), ("all five lines", use_all)]:
    s = fits(use)
    kT = -1 / s[:, 0]
    print(f"\n{label}: slopes {np.round(s[:,0],2)}  kT per run {np.round(kT,3)} eV")
    a_m, a_sd = s[:, 0].mean(), s[:, 0].std(ddof=1)
    print(f"  mean slope {a_m:.2f} +- {a_sd:.2f} (std) /eV ->  kT = {-1/a_m:.3f} +- {a_sd/a_m**2:.3f} eV"
          f"   (+- {a_sd/a_m**2/np.sqrt(len(s)):.3f} as std error of mean)")

# two-point slopes per run (shows curvature)
print("\nlocal kT between neighbouring lines (run mean):")
Y = np.array([yval(area[r]) for r in files])
for i in range(len(names) - 1):
    sl = (Y[:, i+1] - Y[:, i]) / (E[i+1] - E[i])
    print(f"  {names[i]}-{names[i+1]}: kT = {np.mean(-1/sl):.3f} eV (spread {np.std(-1/sl, ddof=1):.3f})")

# ---------- plot ----------
s = fits(use_noHa)
# runs differ only by an overall factor (fibre pointing) -> remove each run's mean level before averaging
off = Y[:, use_noHa].mean(1)
Yshift = Y - off[:, None] + off.mean()
ym, ysd = Yshift.mean(0), Yshift.std(0, ddof=1)
a_m = s[:, 0].mean()
kT, dkT_runs = -1 / a_m, s[:, 0].std(ddof=1) / a_m**2
b_m = ym[use_noHa].mean() - a_m * E[use_noHa].mean()
# standard error of the slope from the scatter of the mean points about the line (captures curvature)
p, cov = np.polyfit(E[use_noHa], ym[use_noHa], 1, cov=True)
da_fit = np.sqrt(cov[0, 0])
dkT_fit = da_fit / p[0]**2
print(f"fit to mean points: slope {p[0]:.2f} +- {da_fit:.2f} /eV -> kT = {-1/p[0]:.3f} +- {dkT_fit:.3f} eV")
print("residuals:", np.round(ym[use_noHa] - np.polyval(p, E[use_noHa]), 3))
dkT = np.hypot(dkT_runs, dkT_fit)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for k, run in enumerate(files):
    ax.plot(E, Yshift[k], ".", color="0.65", ms=5, label="single runs" if k == 0 else None)
ax.errorbar(E[use_noHa], ym[use_noHa], ysd[use_noHa], fmt="o", color="C0", capsize=4,
            label="mean of 4 runs (± std)")
ax.errorbar(E[~use_noHa], ym[~use_noHa], ysd[~use_noHa], fmt="s", mfc="white", color="C3",
            capsize=4, label="H$\\alpha$ (saturated, not fitted)")
Ex = np.linspace(12.0, 13.4, 50)
ax.plot(Ex, a_m * Ex + b_m, "-", color="C0",
        label=f"fit H$\\beta$–H$\\epsilon$: $k_BT_{{exc}}$ = {kT:.2f} ± {dkT:.2f} eV")
for n, e, yy in zip(names, E, ym):
    ax.annotate({"Ha": "H$\\alpha$", "Hb": "H$\\beta$", "Hg": "H$\\gamma$",
                 "Hd": "H$\\delta$", "He": "H$\\epsilon$"}[n], (e, yy), xytext=(6, 6),
                textcoords="offset points")
ax.set_xlabel("$E_i$ (eV)")
ax.set_ylabel("$\\ln(I / C_s A_{ik} g_i)$")
ax.set_title("Boltzmann plot, hydrogen lamp (spectrometer 12)")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("boltzmann_plot.png", dpi=200)
print(f"\nplot saved; kT = {kT:.3f} +- {dkT:.3f} eV")
print("y mean:", np.round(ym, 3), "\ny std:", np.round(ysd, 3))
