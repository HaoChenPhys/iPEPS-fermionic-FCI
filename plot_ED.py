import pickle
import math

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fmt_val_unc(val, unc):
    """Format value(uncertainty) in compact physics notation."""
    if unc <= 0:
        return f"{val}"
    unc_order = math.floor(math.log10(abs(unc)))
    leading = unc / 10**unc_order
    n_unc_digits = 2 if leading < 3 else 1
    n_dec = -unc_order + (n_unc_digits - 1)
    unc_int = round(round(unc, n_dec) * 10**n_dec)
    return f"{round(val, n_dec):.{n_dec}f}({unc_int})"


def load_spectrum(N1, N2, Ne, deg, phix, phiy, t1, V1, V2, V3, V4):
    filename = (
        f"ED_data/N1_{N1:d}_N2_{N2:d}_Ne_{Ne:d}_deg_{deg:d}/"
        f"spec_phix_{phix:.2f}_phiy_{phiy:.2f}_t1_{t1:.3f}"
        f"_V1_{V1:.2f}_V2_{V2:.2f}_V3_{V3:.2f}_V4_{V4:.2f}"
    )
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def ground_state_energy_per_site(evals, N1, N2, n_states=3):
    all_evals = np.sort(np.concatenate(list(evals.values())))
    return all_evals[:n_states].mean() / (N1 * N2)


def fit_energy(inv_sizes, energies, order=1):
    inv_sizes = np.asarray(inv_sizes)
    energies = np.asarray(energies)
    coeffs = np.polyfit(inv_sizes, energies, order)
    return coeffs


def plot_energy(clusters, Es, ax=None, fit=True, fit_color=None, **kwargs):
    plt.style.use('science')
    if ax is None:
        _, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=10)

    inv_sizes = [1 / (N1 * N2) for N1, N2, _, _ in clusters]
    ax.plot(inv_sizes, Es, lw=0, marker='x', markersize=6, **kwargs)

    if fit:
        x_fit = np.linspace(0, max(inv_sizes) * 1.05, 100)

        # Linear fit with covariance
        c1, cov1 = np.polyfit(np.asarray(inv_sizes), np.asarray(Es), 1, cov=True)
        sigma_e0 = np.sqrt(cov1[-1, -1])
        y_lin = np.polyval(c1, x_fit)
        fc = fit_color or ax.lines[-1].get_color()
        ax.plot(x_fit, y_lin, lw=1, ls='--', color=fc,
                label=rf"linear: $e_\infty={fmt_val_unc(c1[1], sigma_e0)}$")
        ax.errorbar(0, c1[1], yerr=sigma_e0, fmt='*', color=fc,
                    markersize=8, capsize=3)
        print(f"Linear fit: e = {c1[0]:.6f} * x + {c1[1]:.6f} +/- {sigma_e0:.6f}")

    ax.set_xlabel(r"$1/(N_1 N_2)$", fontsize=11, color=kwargs.get('color', 'black'))
    return ax


if __name__ == "__main__":
    clusters = [
        (5, 3, 5, 0),
        (9, 2, 6, 48),
        (21, 1, 7, 48),
        (12, 2, 8, 96),
        (9, 3, 9, 60),
    ]
    phix, phiy = 0.0, 0.0
    V1, V2, V3, V4 = 1.0, 0.0, 0.0, 0.0
    t1 = 0.10

    gs_Es = []
    for N1, N2, Ne, deg in clusters:
        evals = load_spectrum(N1, N2, Ne, deg, phix, phiy, t1, V1, V2, V3, V4)
        gs_Es.append(ground_state_energy_per_site(evals, N1, N2) / t1)

    # --- iPEPS: chi->inf extrapolated energies for each D ---
    # Import first (triggers side-effect plots in plot_E), then close them.
    from plot_E import energies_by_chi, polyfit_e_vs_invchi
    plt.close('all')

    lower_bounds = {4: 128, 5: 75, 6: 72, 7: 98, 8: 112, 9: 144}
    fit_degs = [2, 2, 2, 2, 2, 2]
    Ds = [4, 5, 6, 7, 8, 9]

    E0_list, sE0_list = [], []
    for i, D in enumerate(Ds):
        d = energies_by_chi(f"FCI_data/states/D{D:d}", lower_bound=lower_bounds[D])
        _, E0, sE0, _ = polyfit_e_vs_invchi(d, deg=fit_degs[i])
        E0_list.append(E0)
        sE0_list.append(sE0)
        print(f"D={D}: E0(chi->inf) = {E0:.6f} +/- {sE0:.6f}")

    # energies_by_chi returns energy*10 (already in units of t1)
    E0_arr = np.array(E0_list)
    sE0_arr = np.array(sE0_list)

    # --- Now create the plot: 1/D on bottom, 1/(N1 N2) on top ---
    fig, ax_D = plt.subplots()
    ed_color = '#C03030'       # crimson for ED
    ed_fit_color = '#E07040'   # warm orange for ED fit
    ipeps_color = '#2060C0'    # blue for iPEPS
    ipeps_fit_color = '#3090E0'  # lighter blue for iPEPS fit

    # ED on secondary (top) axis
    ax_ed = ax_D.twiny()
    ax_ed = plot_energy(clusters, gs_Es, ax=ax_ed,
                        color=ed_color, label="ED",
                        fit_color=ed_fit_color)
    ax_ed.tick_params(axis='x', colors=ed_color, labelsize=10)
    ax_ed.spines['top'].set_color(ed_color)

    inv_D = np.array([1.0 / D for D in Ds])
    D_arr = np.array(Ds, dtype=float)

    # plot iPEPS E0 vs 1/D
    ax_D.errorbar(inv_D, E0_arr, yerr=sE0_arr, fmt='s', capsize=3,
                  color=ipeps_color, markersize=5, label="iPEPS")

    # --- Try various fitting forms and select the best by AICc ---
    N = len(D_arr)
    inv_D_fit = np.linspace(0, max(inv_D) * 1.05, 200)

    def aicc(n, k, rss):
        """Corrected AIC for small samples."""
        if rss <= 0:
            return -np.inf
        aic = n * np.log(rss / n) + 2 * k
        if n - k - 1 > 0:
            aic += 2 * k * (k + 1) / (n - k - 1)
        return aic

    models = {}

    # 1) Linear: a/D + c  (k=2)
    c1 = np.polyfit(inv_D, E0_arr, 1)
    res1 = np.sum((E0_arr - np.polyval(c1, inv_D))**2)
    models["linear"] = dict(k=2, rss=res1, e_inf=c1[1],
                            curve=np.polyval(c1, inv_D_fit),
                            formula=f"e = {c1[0]:.6f}/D + {c1[1]:.6f}")

    # 2) Quadratic: a/D² + b/D + c  (k=3)
    c2, cov2 = np.polyfit(inv_D, E0_arr, 2, cov=True)
    sigma_c2 = np.sqrt(cov2[-1, -1])
    res2 = np.sum((E0_arr - np.polyval(c2, inv_D))**2)
    models["quadratic"] = dict(k=3, rss=res2, e_inf=c2[2], sigma=sigma_c2,
                               curve=np.polyval(c2, inv_D_fit),
                               formula=f"e = {c2[0]:.6f}/D^2 + {c2[1]:.6f}/D + {c2[2]:.6f}")

    # 3) Cubic: a/D³ + b/D² + c/D + d  (k=4)
    c3 = np.polyfit(inv_D, E0_arr, 3)
    res3 = np.sum((E0_arr - np.polyval(c3, inv_D))**2)
    models["cubic"] = dict(k=4, rss=res3, e_inf=c3[3],
                           curve=np.polyval(c3, inv_D_fit),
                           formula=f"e = {c3[0]:.4f}/D^3 + {c3[1]:.4f}/D^2 + {c3[2]:.6f}/D + {c3[3]:.6f}")

    # 4) Power-law: a * D^{-b} + c  (k=3)
    def power_model(D, a, b, c):
        return a * D**(-b) + c
    popt_pw, _ = curve_fit(power_model, D_arr, E0_arr,
                           p0=[0.01, 2.0, E0_arr[-1]], maxfev=10000)
    res_pw = np.sum((E0_arr - power_model(D_arr, *popt_pw))**2)
    D_dense = 1.0 / inv_D_fit[1:]  # skip 1/D=0
    models["power"] = dict(k=3, rss=res_pw, e_inf=popt_pw[2],
                           curve_x=inv_D_fit[1:],
                           curve_y=power_model(D_dense, *popt_pw),
                           formula=f"e = {popt_pw[0]:.6f} * D^(-{popt_pw[1]:.4f}) + {popt_pw[2]:.6f}")

    # 5) Exponential: a * exp(b/D) + c  (k=3)
    def exp_model(D, a, b, c):
        return a * np.exp(b / D) + c
    popt_ex, _ = curve_fit(exp_model, D_arr, E0_arr,
                           p0=[0.01, 1.0, E0_arr[-1]], maxfev=10000)
    res_ex = np.sum((E0_arr - exp_model(D_arr, *popt_ex))**2)
    models["exponential"] = dict(k=3, rss=res_ex, e_inf=popt_ex[2],
                                 curve_x=inv_D_fit[1:],
                                 curve_y=exp_model(D_dense, *popt_ex),
                                 formula=f"e = {popt_ex[0]:.6f} * exp({popt_ex[1]:.4f}/D) + {popt_ex[2]:.6f}")

    # Print comparison table
    print(f"\n{'Model':<14s} {'k':>2s} {'RSS':>12s} {'AICc':>10s} {'e_inf':>12s}")
    print("-" * 54)
    best_name, best_aicc = None, np.inf
    for name, m in models.items():
        a = aicc(N, m["k"], m["rss"])
        m["aicc"] = a
        print(f"{name:<14s} {m['k']:>2d} {m['rss']:>12.2e} {a:>10.2f} {m['e_inf']:>12.6f}")
        if a < best_aicc:
            best_aicc = a
            best_name = name

    print(f"\nBest model (lowest AICc): {best_name}")

    # Linear fit on D=7,8,9 subset
    mask = np.array([D in (7, 8, 9) for D in Ds])
    c_sub, cov_sub = np.polyfit(inv_D[mask], E0_arr[mask], 1, cov=True)
    sigma_sub = np.sqrt(cov_sub[-1, -1])
    y_sub = np.polyval(c_sub, inv_D_fit)
    ax_D.plot(inv_D_fit, y_sub, ls='--', lw=1, color=ipeps_fit_color,
              label=rf"linear ($D\geq7$): $e_\infty={fmt_val_unc(c_sub[1], sigma_sub)}$")
    ax_D.errorbar(0, c_sub[1], yerr=sigma_sub, fmt='*', color=ipeps_color,
                  markersize=8, capsize=3)
    print(f"Linear fit (D>=7): e = {c_sub[0]:.6f}/D + {c_sub[1]:.6f} +/- {sigma_sub:.6f}")

    # style bottom axis (1/D)
    ax_D.set_xlabel(r"$1/D$", fontsize=11, color=ipeps_color)
    ax_D.set_ylabel(r"$e_0/t_1$", fontsize=11)
    ax_D.tick_params(axis='x', colors=ipeps_color, labelsize=10)
    ax_D.tick_params(axis='y', labelsize=10)
    ax_D.spines['bottom'].set_color(ipeps_color)

    # combined legend: data first, fits at the end
    h1, l1 = ax_ed.get_legend_handles_labels()
    h2, l2 = ax_D.get_legend_handles_labels()
    all_h, all_l = h1 + h2, l1 + l2
    # sort: data labels first, fit labels (containing e_inf) at the end
    data = [(h, l) for h, l in zip(all_h, all_l) if "e_\\infty" not in l]
    fits = [(h, l) for h, l in zip(all_h, all_l) if "e_\\infty" in l]
    ordered = data + fits
    ax_ed.legend([h for h, _ in ordered], [l for _, l in ordered],
                 fontsize=7, loc='upper left',
                 frameon=False, handletextpad=0.4, labelspacing=0.35)

    fig.savefig("./figs/ED_data.pdf", bbox_inches="tight")
