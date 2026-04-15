import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def xi_from_spec(w, is_n0):
    """Correlation length from normalized TM eigenvalues.
    For n=0 sector use the 2nd largest |lambda|, otherwise the largest."""
    w_sorted = w[np.argsort(-np.abs(w))]
    lam = np.abs(w_sorted[1]) if is_n0 else np.abs(w_sorted[0])
    return -1.0 / np.log(lam)


def load_xi(D, opt_chi, chis, sectors):
    xi = {n: [] for n in sectors}
    chi_out = []
    for chi in chis:
        data_dir = f"./FCI_3x3_N3/D{D:d}/optchi_{opt_chi:d}/chi_{chi:d}"
        xi_chi = {}
        ok = True
        for n in sectors:
            fn = os.path.join(data_dir, f"TM_TAT_spec_n_sector_{n:d}.npy")
            if not os.path.exists(fn):
                ok = False
                break
            w = np.load(fn)
            xi_chi[n] = xi_from_spec(w, is_n0=(n == 0))
        if not ok:
            continue
        chi_out.append(chi)
        for n in sectors:
            xi[n].append(xi_chi[n])
    return np.array(chi_out), {n: np.array(v) for n, v in xi.items()}


def fmt_val_unc(val, unc):
    """Format value(uncertainty) in compact physics notation, e.g. 12.52(2)."""
    if unc <= 0 or not np.isfinite(unc):
        return f"{val:.3f}"
    dec = -int(np.floor(np.log10(unc)))
    if dec < 0:
        dec = 0
    unc_digit = int(round(unc * 10**dec))
    return f"{val:.{dec}f}({unc_digit})"


def quadratic_extrap(x, a, b, c):
    return a + b * x + c * x**2


plt.style.use("science")
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
fig.subplots_adjust(wspace=0.3)

sectors = [0, 1]
markers = {0: 'o', 1: 's'}
colors = {0: '#0072B2', 1: '#D55E00'}
fit_colors = {0: '#0072B2', 1: '#D55E00'}
sector_labels = {0: r"$n=0$", 1: r"$n=1$"}

configs = [
    ("(a)", "$D=4$", 4, 64, [64, 96, 128, 160, 192, 320]),
    ("(b)", "$D=9$", 9, 117, [81, 117, 180, 252, 360, 468]),
]

for idx, (panel_label, D_label, D, opt_chi, chis) in enumerate(configs):
    ax = axes[idx]
    chi_arr, xi_dict = load_xi(D, opt_chi, chis, sectors)
    inv_chi = 1.0 / chi_arr
    x_extrap = np.linspace(0, inv_chi.max() * 1.1, 100)
    xi_scale = np.sqrt(3) * 3

    print(f"\n{panel_label}:")
    for n in sectors:
        xi_arr = xi_dict[n] * xi_scale
        popt, pcov = curve_fit(quadratic_extrap, inv_chi, xi_arr)
        perr = np.sqrt(np.diag(pcov))
        xi_inf, xi_inf_err = popt[0], perr[0]
        xi_str = fmt_val_unc(xi_inf, xi_inf_err)
        print(f"  n={n}: xi_inf = {xi_inf:.4f} +/- {xi_inf_err:.4f}")

        ax.plot(x_extrap, quadratic_extrap(x_extrap, *popt), '--', lw=1,
                color=fit_colors[n])
        ax.plot(inv_chi, xi_arr, marker=markers[n], markersize=5,
                markerfacecolor='none', markeredgewidth=1.5, lw=0,
                color=colors[n],
                label=sector_labels[n] + rf": $\xi_{{\infty}}={xi_str}$")

    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel(r"$1/\chi$")
    ax.set_ylabel(r"$\xi^n$")
    ax.text(0.6, 0.95, f"{panel_label} {D_label}", transform=ax.transAxes,
            ha="left", va="top", fontsize=12, fontweight="extra bold")
    if D == 4:
        # ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(0.02, 0.5))
        ax.legend(fontsize=11, loc='upper right', bbox_to_anchor=(0.87, 0.7), frameon=False, handletextpad=-0.3)
    else:
        ax.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.05, 0.9), frameon=False, handletextpad=-0.3)

plt.savefig("./TM_xi.pdf")
print("\nSaved TM_xi.pdf")
