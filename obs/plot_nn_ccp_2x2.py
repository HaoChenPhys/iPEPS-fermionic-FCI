import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import assoc_laguerre
from scipy.optimize import curve_fit


def plot_corrf(dist, corrf, lw=0, markersize=0, ax=None, label=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(dist, corrf, markersize=markersize, markerfacecolor='none', lw=lw, markeredgewidth=1.5, label=label, **kwargs)
    return ax


def plot_nn_corr_nB_2dirn(site, file_nB_nB, file_nB_nA, ax=None):
    colors = ['#009E73', '#D55E00']
    markers = ['x', 'o']
    for j, dirn in enumerate([(1, 0), (0, 1)]):
        filename = file_nB_nB + f"_site_{site:d}_dirn_({dirn[0]},{dirn[1]}).npy"
        with open(filename, "rb") as f:
            dist1 = np.load(f)
            nB_nB_corrf = np.load(f)
        dist1 = dist1.astype(dtype=np.float64)
        dist1 = np.sqrt(3)*np.arange(1, len(dist1)+1)
        ax = plot_corrf(dist1, nB_nB_corrf, markersize=7.5, marker=markers[j], ax=ax, color=colors[j])

        filename = file_nB_nA + f"_site_{site:d}_dirn_({dirn[0]},{dirn[1]}).npy"
        with open(filename, "rb") as f:
            dist2 = np.load(f)
            nB_nA_corrf = np.load(f)
        dist2 = dist2.astype(dtype=np.float64)
        for i in range(len(dist2)):
            dist2[i] = np.sqrt((2*i+1)**2*3/4 + 0.25)
        ax = plot_corrf(dist2, nB_nA_corrf, ax=ax, markersize=7.5, marker=markers[j], color=colors[j])


def compute_poly(rs, cn):
    g = 1 - np.exp(-rs**2/2)
    factor = np.exp(-rs**2/2)*rs**2
    for n, c in enumerate(cn, start=1):
        g += c*(-1)**n*factor*assoc_laguerre(rs**2, n-1, 2)/np.sqrt(np.pi*(n+1)*n)
    return g


def nn_Laughlin_1_3(rs):
    cn = [2.64496,  1.00274, -0.06065, -0.41040, -0.39510,
        -0.26016, -0.12206, -0.02167,  0.03658,  0.06148,
        0.06439,  0.05506,  0.04083,  0.02574,  0.01264,
        0.00280, -0.00414, -0.00825, -0.01011, -0.01028]
    return compute_poly(rs, cn)


def nn_IQHE(rs):
    cn = [0]
    return compute_poly(rs, cn)


def plot_ccp_corr(D, opt_chi, chis, site, dirn=(1, 0), ax=None, **kwargs):
    for chi in chis:
        if dirn == (1, 0):
            filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(1,0).npy"
        else:
            filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(0,1).npy"
        with open(filename, "rb") as f:
            dist1 = np.load(f)
            cA_cpA_corrf = np.load(f)
        ax = plot_corrf(dist1*np.sqrt(3), np.abs(cA_cpA_corrf), markersize=5, marker='o', lw=1.5, label=f'$\\chi={chi:d}$', ax=ax, **kwargs)
    return ax


def collect_c_cp_bulk(D, opt_chi, chis, num_points, site, dirn=(1, 0)):
    x, y = [], []
    for i, chi in enumerate(chis):
        if dirn == (1, 0):
            filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(1,0).npy"
        else:
            filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(0,1).npy"
        with open(filename, "rb") as f:
            dist1 = np.load(f)
            cA_cpA_corrf = np.load(f)
            x.append(dist1[:num_points[i]]*np.sqrt(3))
            y.append(np.abs(cA_cpA_corrf)[:num_points[i]])
    return np.concatenate(x), np.concatenate(y)


def linear_fit(x, y, bounds_tau_positive=True, p0=None, maxfev=20000):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def f(x, A, tau):
        return np.log(A) - x / tau
    param_names = ("A", "tau")
    npar = 2

    if p0 is None:
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]
        span = xs.max() - xs.min()
        tau0 = span / 3 if span > 0 else 1.0
        A0 = float(ys.max())
        p0 = (A0, tau0)

    if bounds_tau_positive:
        lower = [-np.inf] * npar
        upper = [ np.inf] * npar
        lower[1] = 0.0
        bounds = (lower, upper)
    else:
        bounds = (-np.inf, np.inf)

    popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
    perr = np.sqrt(np.diag(pcov))

    return {
        "popt": popt, "perr": perr, "pcov": pcov,
        "param_names": param_names, "model_func": f,
        "x_fit": x, "y_fit": y,
    }


# ── Figure: 2×2 layout ──
# Top row: c†c correlations (from plot_c_cp.py)
# Bottom row: nn correlations (from plot_nn_comp.py)

plt.style.use("science")
fig = plt.figure(figsize=(5.5, 4))
subfigs = fig.subfigures(2, 1, hspace=0.05)
axes = np.empty((2, 2), dtype=object)
axes[0, 0], axes[0, 1] = subfigs[0].subplots(1, 2, sharey=True)
axes[1, 0], axes[1, 1] = subfigs[1].subplots(1, 2, sharey=True)

# ── Top-left (a): c†c two directions ──
ax = axes[0, 0]
site = 0
colors = ['#009E73', '#D55E00']
for D, opt_chi, chi in [(9, 117, 468)]:
    filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(1,0).npy"
    with open(filename, "rb") as f:
        dist1 = np.load(f)
        cA_cpA_corrf = np.load(f)
    ax = plot_corrf(dist1*np.sqrt(3), np.abs(cA_cpA_corrf), markersize=5, marker='o', lw=1.5,
        label=r'$\boldsymbol{a_1}$', ax=ax, color=colors[0])

    filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(0,1).npy"
    with open(filename, "rb") as f:
        dist1 = np.load(f)
        cA_cpA_corrf = np.load(f)
    ax = plot_corrf(dist1*np.sqrt(3), np.abs(cA_cpA_corrf), markersize=5, marker='x', lw=1.5,
        label=r'$\boldsymbol{a_2}$', ax=ax, color=colors[1])

ax.text(0.09, 0.15, r"(a) $D=9, \chi=468$", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="extra bold")

uniq = {}
for h, lb in zip(*ax.get_legend_handles_labels()):
    if lb not in uniq:
        uniq[lb] = h
labels = list(uniq.keys())
handles = list(uniq.values())
order = np.argsort(labels)
ax.legend([handles[i] for i in order], [labels[i] for i in order], ncols=1, handlelength=1.4, fontsize=10, columnspacing=0.5, handletextpad=0.3, loc=3, bbox_to_anchor=(0.6, 0.6))

ax.set_ylim((5e-13, 2e-1))
ax.set_xlim((-1, 120))
ax.set_ylabel(r"$\langle \hat{c}_{\text{A},0} \hat{c}^\dagger_{\text{A},\boldsymbol{r}}\rangle$")
ax.set_xlabel(r"$|\boldsymbol{r}|/a$")
ax.set_yscale('log')

# ── Top-right (b): c†c chi convergence + fit ──
ax = axes[0, 1]
D, opt_chi = 9, 117
chis = [81, 117, 180, 252, 360, 468]
plot_ccp_corr(D, opt_chi, chis=chis, site=0, dirn=(0, 1), ax=ax)

num_points = [4]*6
x, y = collect_c_cp_bulk(D, opt_chi, chis, num_points, site, dirn=(1, 0))
p0 = (0.2, 0.6)
res = linear_fit(x, np.log(y), p0=p0)
print(dict(zip(res["param_names"], res["popt"])))
print("1σ:", dict(zip(res["param_names"], res["perr"])))

xs = np.linspace(1.5, 30, 100)
A, tau = res["popt"]
ax.plot(xs, A*np.exp(-xs/tau), ls='--', color='black')

ax.set_yscale('log')
ax.set_xlabel(r"$|\boldsymbol{r}|/a$")
ax.text(0.2, 0.15, r"(b)", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="extra bold")
ax.text(0.35, 0.55, r"$\xi_{\mathrm{bulk}}/a=$"+f'{tau:.2f}', transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="extra bold")

handles, labels = ax.get_legend_handles_labels()
seen = set()
h, l = [], []
for hd, lb in zip(handles, labels):
    if lb not in seen:
        seen.add(lb)
        h.append(hd)
        l.append(lb)

ax.legend(
    h, l,
    ncol=2, handletextpad=0.3, fontsize=9.5, handlelength=1.4,
    labelspacing=0.4, columnspacing=0.7, frameon=False, loc=3, bbox_to_anchor=(0.05, 0.55)
)
ax.set_xlim((-1, 120))

# ── Bottom-left (c): CI nn correlation ──
ax = axes[1, 0]
l = 1.63
rs = np.linspace(0, l*10, 200)
g = nn_IQHE(rs)
ax.plot(rs/l, g, color='k', label=r"$\nu=1$", lw=1.5, ls='dashed')

optchi = 108
chi = 324
CI_nB_nB = f"./CI_honeycomb_1x1/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nB_corrf"
CI_nB_nA = f"./CI_honeycomb_1x1/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nA_corrf"
plot_nn_corr_nB_2dirn(0, CI_nB_nB, CI_nB_nA, ax=ax)
ax.set_xlabel(r"$|\boldsymbol{x}|/a$")
ax.set_ylabel(r"$g(\boldsymbol{x})$")
ax.text(0.15, 0.15, r"(c) CI: $D=6, \chi=$"+f"{chi:d}", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="extra bold")
ax.legend(fontsize=10)
ax.set_xlim([-0.5, 10])
ax.set_ylim([-0.05, 1.15])

# ── Bottom-right (d): FCI nn correlation ──
ax = axes[1, 1]
rs = np.linspace(0, l*10, 200)
g = nn_Laughlin_1_3(rs)
ax.plot(rs/l, g, color='k', label=r"$\nu=1/3$", lw=1.5, ls='dashed')

D = 9
optchi = 117
chi = 468
FCI_nB_nB = f"./FCI_3x3_N3/D{D:d}/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nB_corrf"
FCI_nB_nA = f"./FCI_3x3_N3/D{D:d}/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nA_corrf"
plot_nn_corr_nB_2dirn(0, FCI_nB_nB, FCI_nB_nA, ax=ax)
ax.set_xlabel(r"$|\boldsymbol{x}|/a$")
ax.text(0.19, 0.15, r"(d) FCI: $D=9, \chi=$"+f"{chi:d}", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="extra bold")
ax.legend(fontsize=10)
ax.set_xlim([-0.5, 10])
ax.set_ylim([-0.05, 1.15])

plt.savefig("../figs/corrf/nn_ccp_2x2.pdf")
