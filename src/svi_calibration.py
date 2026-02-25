import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

CLEANED_DATA_DIR  = os.getenv("CLEANED_DATA_DIR",  "data/cleaned_data")
CLEANED_FILE_NAME = os.getenv("CLEANED_FILE_NAME", "temp_clean.csv")
IV_DATA_DIR       = "data/implied_volatility"
SVI_OUTPUT_DIR    = "data/svi_calibration"
VIZ_DIR           = "visualizations/svi_calibration"


for d in [SVI_OUTPUT_DIR, VIZ_DIR]:
    os.makedirs(d, exist_ok=True)

def svi_raw(x, a, b, rho, m, sigma):
    """
    Raw SVI total implied variance for log-moneyness vector x.

    w(x) = a + b * [ rho*(x-m) + sqrt((x-m)^2 + sigma^2) ]

    Returns
    w : total implied variance (IV^2 * T),  always ≥ 0 when params are valid
    """
    x = np.asarray(x, dtype=float)
    disc = np.sqrt((x - m) ** 2 + sigma ** 2)
    return a + b * (rho * (x - m) + disc)


def svi_to_iv(x, T, a, b, rho, m, sigma):
    """Convert raw SVI total variance to implied vol (annualised)."""
    w = svi_raw(x, a, b, rho, m, sigma)
    w = np.maximum(w, 1e-10)          # guard against tiny negatives
    return np.sqrt(w / T)


def bs_call_price(F, K, T, sigma):
    """
    Black-Scholes call price expressed in terms of forward price F.
        C = F*N(d1) - K*N(d2)   [undiscounted]
    F     : forward price for this maturity (scalar)
    K     : strike(s) — scalar or array
    T     : time to maturity in years (scalar)
    sigma : implied vol(s) — scalar or array, same shape as K
    """
    K     = np.asarray(K, float)
    sigma = np.asarray(sigma, float)
    if T <= 0:
        return np.maximum(F - K, 0.0)
    safe_sigma = np.maximum(sigma, 1e-8)
    d1 = (np.log(F / np.maximum(K, 1e-8)) + 0.5 * safe_sigma ** 2 * T) / \
         (safe_sigma * np.sqrt(T))
    d2 = d1 - safe_sigma * np.sqrt(T)
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def _estimate_forward_per_slice(g):
    """
    For a single maturity slice g, find the forward F and PV factor D that
    minimise squared error in  C - P ≈ D*(F - K)  for near-the-money strikes.
    This is the same optimisation used in forward_discount_from_parity.py.
    """
    K    = g["K"].to_numpy(float)
    imid = (g["C_MID"] - g["P_MID"]).to_numpy(float)

    # Use the 15 strikes closest to ATM (smallest |imid|)
    idx   = np.argsort(np.abs(imid))[:min(15, len(imid))]
    K0    = K[idx];  im0 = imid[idx]

    f0  = float(np.mean(im0 + K0))    # rough guess: imid + K ≈ F when D≈1
    pv0 = 1.0

    def obj(params):
        F, D = params
        return float(np.sum((D * (F - K0) - im0) ** 2))

    res = minimize(
        obj,
        x0=[f0, pv0],
        method="L-BFGS-B",
        bounds=[(K0.min(), K0.max()), (0.5, 1.2)],
    )
    F_hat, D_hat = res.x
    return float(F_hat), float(D_hat)


def build_forward_curve(df):
    """
    Return a DataFrame indexed by T with columns F (forward price) and D
    (discount factor).  Skips slices with fewer than 10 usable quotes.
    """
    rows = []
    for T, g in df.groupby("T"):
        g = g.dropna(subset=["K", "C_MID", "P_MID"])
        g = g[(g["C_MID"] > 0) & (g["P_MID"] > 0)]
        if len(g) < 10:
            continue
        F, D = _estimate_forward_per_slice(g)
        if F > 0 and D > 0:
            rows.append({"T": float(T), "F": F, "D": D})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def prepare_svi_input(df, forward_curve):
    """
    For each option row, attach the forward price and compute:
        x  = log(K / F)   forward log-moneyness (standard SVI convention)
        w  = IV^2 * T     total implied variance (what the Phase 2 objective fits)
        wt = 1 / spread^2 per-quote reliability weight (tighter spread = more weight)

    A single IV per option is used (from Daksh's secant solver, falling back to
    C_IV_MID).  This is consistent regardless of whether the option is a call or
    put, since put-call parity guarantees both sides imply the same IV.
    We prefer C_IV_CALC from Daksh's secant solver; fall back to C_IV_MID.
    """
    fc_dict = dict(zip(forward_curve["T"], forward_curve["F"]))

    rows = []
    for T, g in df.groupby("T"):
        if T not in fc_dict:
            continue
        F = fc_dict[T]
        g = g.copy()
        g["x"] = np.log(g["K"].to_numpy(float) / F)   # forward log-moneyness

        # Choose IV: secant-calculated preferred, else market mid
        use_iv = g["C_IV_CALC"].where(
            g["C_IV_CALC"].notna() & (g["C_IV_CALC"] > 0.01),
            other=g["C_IV_MID"]
        )
        g["iv"] = use_iv
        g = g[g["iv"].notna() & (g["iv"] > 0.01) & (g["iv"] < 2.0)]

        # Total implied variance — the quantity SVI directly models
        g["w"] = g["iv"] ** 2 * T

        # Inverse bid-ask spread weights: tighter spread = more trustworthy quote
        spread = (g["C_SPRD"] + g["P_SPRD"]).clip(lower=1e-4)
        g["wt"] = 1.0 / (spread ** 2)
        g["wt"] = g["wt"] / g["wt"].mean()    # normalise within slice

        # Only keep reasonable log-moneyness range (drop extreme wings)
        g = g[g["x"].between(-1.0, 0.3)]
        g = g[g["w"] > 0]
        if len(g) >= 5:
            rows.append(g[["T", "K", "x", "iv", "w", "wt", "S"]].copy())

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



# SSVI with square-root kernel:  phi(theta) = eta / sqrt(theta)
#
#   w_SSVI(T, x) = theta_T/2 * [1 + rho*phi*x + sqrt((phi*x + rho)^2 + 1-rho^2)]
#
# where theta_T = ATM total variance at maturity T.
#
# Free global params: eta (vol-of-vol), rho (spot-vol correlation)
# theta_T is estimated directly from ATM data for each slice.

def ssvi_sqrt_total_variance(x, theta, eta, rho):
    """SSVI with square-root power law phi(theta) = eta / sqrt(theta)."""
    phi = eta / np.sqrt(theta)
    return (theta / 2.0) * (1 + rho * phi * x +
                             np.sqrt((phi * x + rho) ** 2 + (1 - rho ** 2)))


def estimate_atm_variance(svi_data):
    """
    For each maturity T, estimate theta_T = ATM total variance by taking the
    median w of the 5 closest-to-ATM options (|x| smallest).
    """
    thetas = {}
    for T, g in svi_data.groupby("T"):
        g_sorted = g.reindex(g["x"].abs().sort_values().index)
        theta_est = float(g_sorted["w"].iloc[:5].median())
        thetas[T] = max(theta_est, 1e-6)
    return thetas


def fit_ssvi_global(svi_data):
    """
    Fit eta, rho globally across all slices.
    Uses weighted SSE of (w_model - w_market)^2.
    Returns (eta, rho) and the dict of theta_T per maturity.
    """
    thetas = estimate_atm_variance(svi_data)

    def objective(params):
        eta, rho = params
        sse = 0.0
        for T, g in svi_data.groupby("T"):
            if T not in thetas:
                continue
            theta = thetas[T]
            w_model = ssvi_sqrt_total_variance(g["x"].to_numpy(), theta, eta, rho)
            resid   = w_model - g["w"].to_numpy()
            sse    += float(np.sum(g["wt"].to_numpy() * resid ** 2))
        return sse

    # Differential evolution gives a robust global search over (eta, rho)
    result = differential_evolution(
        objective,
        bounds=[(0.01, 5.0), (-0.99, -0.01)],   # rho negative:equity skew
        seed=42,
        maxiter=500,
        tol=1e-8,
        polish=True,
    )
    eta_hat, rho_hat = result.x
    return float(eta_hat), float(rho_hat), thetas


def ssvi_to_raw_svi(T, theta, eta, rho):
    """
    Convert SSVI parameters (theta, eta, rho) to raw SVI (a, b, rho, m, sigma).
    Used to seed the per-slice optimiser with an arbitrage-free initial guess.

    Derivation: match the five raw SVI parameters to the SSVI formula.
    From the natural to raw mapping with Delta=0, mu=0:
        omega = theta,  nu = phi(theta) = eta / sqrt(theta)
        a = (omega/2)(1 - rho^2)
        b = (omega * nu) / 2
        sigma = sqrt(1 - rho^2) / nu
        m = -rho / nu
        (rho stays the same)
    """
    phi   = eta / np.sqrt(theta)
    omega = theta
    b     = omega * phi / 2.0
    sig   = np.sqrt(max(1.0 - rho ** 2, 1e-6)) / phi
    m_val = -rho / phi
    a     = (omega / 2.0) * (1.0 - rho ** 2)
    return a, b, rho, m_val, sig


# For each maturity slice independently, we fine-tune the 5 raw SVI params
# (a, b, rho, m, sigma) to fit the market total variance as closely as possible
# while penalising calendar-spread crossedness: we don't want the total-variance
# curve for a shorter maturity to ever exceed that of a longer maturity.
#
# Objective = weighted_SSE(slice) + penalty * crossedness(with neighbours)

def _svi_crossedness(params1, params2, x_grid=None):
    """
    Compute how much the shorter-maturity SVI slice (params1) exceeds the
    longer-maturity slice (params2) across a grid of log-moneyness values.
    This is Gatheral's 'crossedness' measure.  Returns 0 if no crossing.
    """
    if x_grid is None:
        x_grid = np.linspace(-1.5, 1.0, 300)
    w1 = svi_raw(x_grid, *params1)
    w2 = svi_raw(x_grid, *params2)
    violations = np.maximum(w1 - w2, 0.0)   # places where shorter > longer
    return float(violations.max())


def durrleman_penalty(p, x_grid=None, weight=500.0):
    """
    Soft penalty on Durrleman's g function.  Penalises any region where g < 0
    (butterfly arbitrage), proportional to the square of the violation.
    Adding this into the per-slice objective is cleaner than post-hoc fixes.
    """
    if x_grid is None:
        x_grid = np.linspace(-1.5, 1.0, 200)
    g = durrleman_g(x_grid, *p)
    violations = np.minimum(g, 0.0)        # only penalise negative g
    return weight * float(np.sum(violations ** 2))


def _svi_param_bounds(x_data, w_data):
    """
    Per-slice parameter bounds following the theory:
        a  ∈ [0, max(w)]       (positive level)
        b  ∈ [0, 5]            (non-negative wings)
        rho∈ (-0.999, 0)       (negative for equities)
        m  ∈ [2*min(x), 2*max(x)]
        sigma > 0              (controls ATM curvature)
    """
    w_max = float(np.max(w_data))
    x_min = float(np.min(x_data))
    x_max = float(np.max(x_data))
    return [
        (0.0,    w_max),       # a
        (1e-4,   5.0),         # b
        (-0.999, -0.001),      # rho
        (2 * x_min, 2 * x_max),   # m
        (1e-4,   2.0),         # sigma
    ]


def fit_svi_slice(x, w_market, weights, p0):
    """
    Fit one SVI slice.  Minimise weighted squared error in total variance space.
    p0  : initial guess [a, b, rho, m, sigma]  from SSVI conversion
    Returns fitted params as list [a, b, rho, m, sigma]
    """
    x = np.asarray(x, float);  w_market = np.asarray(w_market, float)
    weights = np.asarray(weights, float)
    bounds  = _svi_param_bounds(x, w_market)

    def obj(p):
        w_model = svi_raw(x, *p)
        resid   = w_model - w_market
        # penalise any negative total variance (inadmissible)
        neg_penalty = float(np.sum(np.maximum(-w_model, 0) ** 2)) * 1e4
        return float(np.sum(weights * resid ** 2)) + neg_penalty

    res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "ftol": 1e-12})
    return res.x.tolist()


def calibrate_all_slices(svi_data, eta, rho, thetas, penalty=200.0):
    """
    Phase 2: refine every slice in total-variance space + calendar penalty.

    Objective per slice:
        sum_i  wt_i * (w_svi(x_i) - w_market_i)^2
    where w = IV^2 * T is the total implied variance and wt_i = 1/spread^2.

    Using a single IV per option (from Daksh's secant solver) ensures the
    objective is consistent across the full moneyness range — OTM puts and
    OTM calls are treated identically since both are expressed in IV space,
    avoiding the call-vs-put price asymmetry of a price-space objective.

    Plus:
        - calendar crossedness penalty  (no total-variance lines crossing)
        - Durrleman butterfly penalty   (no negative risk-neutral density)

    Returns a dict {T: [a, b, rho, m, sigma]}.
    """
    sorted_T = sorted(svi_data["T"].unique())
    slices   = {}

    # Build initial guesses from SSVI conversion
    for T in sorted_T:
        theta = thetas.get(T, T * 0.04)
        a0, b0, r0, m0, s0 = ssvi_to_raw_svi(T, theta, eta, rho)
        slices[T] = [a0, b0, r0, m0, s0]

    # Forward pass: optimise each slice, penalise crossedness with previous
    for i, T in enumerate(sorted_T):
        g    = svi_data[svi_data["T"] == T]
        x_s  = g["x"].to_numpy(float)
        w_s  = g["w"].to_numpy(float)   # market total implied variance
        wt_s = g["wt"].to_numpy(float)  # 1/spread^2 weights
        p0   = slices[T]

        prev_params = slices[sorted_T[i - 1]] if i > 0 else None

        # Normalise SSE by its value at the SSVI initial guess.
        # Without this, long-dated slices (large w ~ sigma^2*T) dominate the
        # objective purely because their absolute variance values are bigger,
        # causing the optimiser to over-weight them and produce unrealistic
        # ATM vol levels.  Dividing by sse0 puts all 46 slices on the same
        # scale regardless of maturity.
        sse0 = float(np.sum(wt_s * (svi_raw(x_s, *p0) - w_s) ** 2))
        sse0 = max(sse0, 1e-12)

        def obj_with_penalty(p):
            w_model  = svi_raw(x_s, *p)
            # Core fit: normalised weighted SSE in total-variance space
            sse      = float(np.sum(wt_s * (w_model - w_s) ** 2)) / sse0
            # Guard against negative total variance
            neg_pen  = float(np.sum(np.maximum(-w_model, 0) ** 2)) * 1e4
            # Durrleman butterfly penalty (prevents density going negative)
            butt_pen = durrleman_penalty(p, weight=300.0)
            # Calendar crossedness penalty (no lines crossing in total-var plot)
            cal_pen  = 0.0
            if prev_params is not None:
                cal_pen = _svi_crossedness(prev_params, p) * penalty
            return sse + neg_pen + butt_pen + cal_pen

        bounds = _svi_param_bounds(x_s, w_s)
        res    = minimize(obj_with_penalty, p0, method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": 3000, "ftol": 1e-12})
        slices[T] = res.x.tolist()

    return slices


def durrleman_g(x, a, b, rho, m, sigma):
    """
    Durrleman's function g(x).  If g(x) < 0 anywhere, the slice has butterfly
    arbitrage (the risk-neutral density goes negative there).

    g(x) = (1 - x*dw/2w)^2 - (dw/2)^2 * (1/w + 1/4) + d2w/2

    where w = svi_raw(x), dw = dw/dx, d2w = d^2w/dx^2.
    """
    disc  = np.sqrt((x - m) ** 2 + sigma ** 2)
    w     = a + b * (rho * (x - m) + disc)
    dw    = b * (rho + (x - m) / disc)
    d2w   = b * sigma ** 2 / disc ** 3

    g = (1.0 - x * dw / (2.0 * w)) ** 2 \
        - (dw ** 2 / 4.0) * (1.0 / w + 0.25) \
        + d2w / 2.0
    return g


def check_butterfly_arbitrage(params_dict, x_grid=None):
    """
    Returns a dict {T: min_g} where min_g < 0 means butterfly arbitrage.
    """
    if x_grid is None:
        x_grid = np.linspace(-1.5, 1.0, 500)
    results = {}
    for T, p in params_dict.items():
        g_vals = durrleman_g(x_grid, *p)
        results[T] = float(g_vals.min())
    return results


def check_calendar_arbitrage(params_dict, x_grid=None):
    """
    For consecutive maturities (T1 < T2), check that w(T1,x) ≤ w(T2,x) for
    all x.  Returns a dict {(T1,T2): max_violation}.
    """
    if x_grid is None:
        x_grid = np.linspace(-1.5, 1.0, 500)
    sorted_T = sorted(params_dict.keys())
    results  = {}
    for i in range(len(sorted_T) - 1):
        T1, T2 = sorted_T[i], sorted_T[i + 1]
        w1 = svi_raw(x_grid, *params_dict[T1])
        w2 = svi_raw(x_grid, *params_dict[T2])
        results[(T1, T2)] = float(np.maximum(w1 - w2, 0).max())
    return results


def print_arbitrage_report(butterfly_results, calendar_results):
    """Print a human-readable arbitrage check summary."""
    print("\n" + "=" * 60)
    print("ARBITRAGE DIAGNOSTICS")
    print("=" * 60)

    print("\n--- Butterfly Arbitrage (min of Durrleman g per slice) ---")
    print(f"  {'T (years)':>12}  {'min g':>12}  {'Status':>12}")
    for T, g_min in sorted(butterfly_results.items()):
        status = "OK" if g_min >= -1e-4 else "VIOLATION"
        print(f"  {T:>12.4f}  {g_min:>12.6f}  {status:>12}")

    print("\n--- Calendar Spread Arbitrage (max w(T1)-w(T2) per pair) ---")
    print(f"  {'T1':>8}  {'T2':>8}  {'Crossedness':>14}  {'Status':>10}")
    for (T1, T2), cross in sorted(calendar_results.items()):
        status = "OK" if cross < 1e-6 else "VIOLATION"
        print(f"  {T1:>8.4f}  {T2:>8.4f}  {cross:>14.8f}  {status:>10}")



def plot_iv_smiles(svi_data, params_dict, max_slices=12):
    """
    For up to max_slices maturities, overlay market IV dots with SVI fitted IV
    curve vs log-moneyness.  This is the standard 'smile per slice' diagnostic.
    """
    sorted_T = sorted(params_dict.keys())
    # pick a representative spread of maturities
    if len(sorted_T) > max_slices:
        idx    = np.round(np.linspace(0, len(sorted_T) - 1, max_slices)).astype(int)
        plot_T = [sorted_T[i] for i in idx]
    else:
        plot_T = sorted_T

    ncols = 4
    nrows = int(np.ceil(len(plot_T) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = np.array(axes).flatten()

    x_fine = np.linspace(-1.0, 0.3, 300)

    for idx_ax, T in enumerate(plot_T):
        ax = axes[idx_ax]
        g  = svi_data[svi_data["T"] == T]
        p  = params_dict[T]
        iv_fit = svi_to_iv(x_fine, T, *p)

        ax.scatter(g["x"], g["iv"], s=8, alpha=0.5, color="steelblue",
                   label="Market IV")
        ax.plot(x_fine, iv_fit, "darkorange", lw=2, label="SVI fit")
        ax.set_title(f"T = {T:.4f} yr", fontsize=9)
        ax.set_xlabel("log(K/F)", fontsize=8)
        ax.set_ylabel("IV", fontsize=8)
        ax.set_ylim(0.05, 0.80)
        ax.grid(True, alpha=0.3)
        if idx_ax == 0:
            ax.legend(fontsize=7)

    # hide unused subplots
    for idx_ax in range(len(plot_T), len(axes)):
        axes[idx_ax].set_visible(False)

    plt.suptitle("SVI Fitted Implied Volatility Smiles – SPX Jan 2023",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "svi_iv_smiles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Smiles plot saved: {path}")


def plot_total_variance_surface(params_dict, max_slices=20):
    """
    Total-variance plot: w(T,x) vs log-moneyness, one curve per maturity.
    The lines should NOT cross – that is the visual proof of no calendar
    spread arbitrage (just like in Gatheral's Figures 3 and 14).
    """
    sorted_T = sorted(params_dict.keys())
    if len(sorted_T) > max_slices:
        idx      = np.round(np.linspace(0, len(sorted_T) - 1, max_slices)).astype(int)
        sorted_T = [sorted_T[i] for i in idx]

    x_grid = np.linspace(-1.0, 0.3, 400)
    cmap   = plt.cm.viridis(np.linspace(0, 1, len(sorted_T)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for color, T in zip(cmap, sorted_T):
        w = svi_raw(x_grid, *params_dict[T])
        ax.plot(x_grid, w, color=color, lw=1.5, label=f"T={T:.3f}")

    ax.set_xlabel("Log-moneyness  log(K/F)", fontsize=11)
    ax.set_ylabel("Total implied variance  σ²T", fontsize=11)
    ax.set_title("SVI Total Variance Surface – lines must NOT cross", fontsize=12)
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(min(sorted_T), max(sorted_T)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="T (years)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "svi_total_variance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Total variance plot saved: {path}")


def plot_iv_surface_3d(params_dict, max_slices=30):
    """3D scatter: fitted IV surface over (log-moneyness, maturity)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sorted_T = sorted(params_dict.keys())
    if len(sorted_T) > max_slices:
        idx      = np.round(np.linspace(0, len(sorted_T) - 1, max_slices)).astype(int)
        sorted_T = [sorted_T[i] for i in idx]

    x_grid = np.linspace(-0.8, 0.25, 80)
    X, T_mesh = np.meshgrid(x_grid, sorted_T)
    IV_mesh   = np.zeros_like(X)
    for i, T in enumerate(sorted_T):
        IV_mesh[i, :] = svi_to_iv(x_grid, T, *params_dict[T])

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, T_mesh, IV_mesh, cmap="plasma", alpha=0.85,
                           linewidth=0, antialiased=True)
    ax.set_xlabel("log(K/F)", labelpad=8)
    ax.set_ylabel("T (years)", labelpad=8)
    ax.set_zlabel("Implied Vol", labelpad=8)
    ax.set_title("SVI Implied Volatility Surface – SPX Jan 2023", pad=12)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="IV")
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "svi_iv_surface_3d.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  3D surface plot saved: {path}")


def plot_atm_term_structure(params_dict):
    """
    ATM implied vol vs maturity.  ATM = x = 0, so IV = sqrt(a + b*sigma) / sqrt(T).
    This plot shows how volatility changes across maturities.
    """
    sorted_T = sorted(params_dict.keys())
    atm_iv   = []
    for T in sorted_T:
        p = params_dict[T]
        w_atm = svi_raw(np.array([0.0]), *p)[0]
        atm_iv.append(np.sqrt(max(w_atm, 1e-8) / T))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sorted_T, atm_iv, "o-", color="steelblue", lw=2, ms=5)
    ax.set_xlabel("Time to Maturity (years)", fontsize=11)
    ax.set_ylabel("ATM Implied Volatility", fontsize=11)
    ax.set_title("ATM Implied Volatility Term Structure", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "svi_atm_term_structure.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ATM term structure saved: {path}")


def save_svi_parameters(params_dict, thetas, eta, rho):
    """Save the calibrated raw SVI parameters table to CSV."""
    rows = []
    for T, p in sorted(params_dict.items()):
        a, b, rho_p, m, sigma = p
        # Compute ATM quantities for the JW interpretation (informational)
        w_atm    = svi_raw(np.array([0.0]), *p)[0]
        atm_var  = max(w_atm / T, 1e-8)
        atm_vol  = np.sqrt(atm_var)
        disc_atm = np.sqrt(m ** 2 + sigma ** 2)
        atm_skew = b / (2 * np.sqrt(w_atm + 1e-10)) * (rho_p - m / disc_atm) if w_atm > 0 else np.nan
        rows.append({
            "T": T,
            "a": a, "b": b, "rho": rho_p, "m": m, "sigma": sigma,
            "theta_T": thetas.get(T, np.nan),
            "atm_vol": atm_vol,
            "atm_skew": atm_skew,
        })
    df_params = pd.DataFrame(rows)
    path = os.path.join(SVI_OUTPUT_DIR, "svi_parameters.csv")
    df_params.to_csv(path, index=False)
    print(f"\n  SVI parameters saved: {path}")
    return df_params


def save_fitted_surface(params_dict, svi_data):
    """
    For every observed option, attach the SVI-fitted total variance and IV.
    Useful downstream for residual analysis or option re-pricing.
    """
    out = svi_data.copy()
    out["w_svi"] = np.nan
    out["iv_svi"] = np.nan
    for T, p in params_dict.items():
        mask = out["T"] == T
        x_vals        = out.loc[mask, "x"].to_numpy(float)
        w_fit         = svi_raw(x_vals, *p)
        out.loc[mask, "w_svi"]  = w_fit
        out.loc[mask, "iv_svi"] = np.sqrt(np.maximum(w_fit, 0) / T)
    path = os.path.join(SVI_OUTPUT_DIR, "svi_fitted_surface.csv")
    out.to_csv(path, index=False)
    print(f"  Fitted surface saved: {path}")


def main():
    # Try the secant IV file first (richest, has C_IV_CALC); fall back to
    # the cleaned data file if the IV file is not present.
    iv_path      = os.path.join(IV_DATA_DIR, "implied_volatility_secant.csv")
    clean_path   = os.path.join(CLEANED_DATA_DIR, CLEANED_FILE_NAME)
    data_path    = iv_path if os.path.exists(iv_path) else clean_path

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Rows loaded: {len(df)}")

    # Strip stray spaces from column names (the raw files have some)
    df.columns = [c.strip().strip("[]").strip() for c in df.columns]

    # Ensure essential numeric columns exist
    for col in ["S", "K", "T", "C_MID", "P_MID", "C_IV_MID", "P_IV_MID",
                "C_SPRD", "P_SPRD", "Y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If C_IV_CALC not present, copy from C_IV_MID
    if "C_IV_CALC" not in df.columns:
        df["C_IV_CALC"] = df.get("C_IV_MID", np.nan)

    df = df.dropna(subset=["S", "K", "T", "C_MID", "P_MID"])
    df = df[df["T"] > 0]
    print(f"  Rows after basic filter: {len(df)}")

    #  Step 2: Estimate forward prices 
    print("\nEstimating forward prices via put-call parity...")
    forward_curve = build_forward_curve(df)
    print(f"  Maturities with valid forward: {len(forward_curve)}")
    fc_path = os.path.join(SVI_OUTPUT_DIR, "forward_curve_svi.csv")
    forward_curve.to_csv(fc_path, index=False)

    #  Step 3: Build SVI input dataset 
    print("\nPreparing SVI input (total variance, forward log-moneyness)...")
    svi_data = prepare_svi_input(df, forward_curve)
    if svi_data.empty:
        print("ERROR: No valid slices found. Check data paths.")
        return
    n_slices = svi_data["T"].nunique()
    print(f"  Usable option rows: {len(svi_data)}")
    print(f"  Maturity slices   : {n_slices}")

    #  Step 4: Phase 1 – SSVI global fit 
    print("\nPhase 1 – Fitting global SSVI (square-root kernel)...")
    eta, rho, thetas = fit_ssvi_global(svi_data)
    print(f"  Global SSVI fit:  eta = {eta:.4f},  rho = {rho:.4f}")
    print("  (These are the vol-of-vol and spot-vol correlation)")

    #  Step 5: Phase 2 – Per-slice refinement 
    print("\nPhase 2 – Refining per-slice with calendar-spread penalty...")
    params_dict = calibrate_all_slices(svi_data, eta, rho, thetas, penalty=200.0)
    print(f"  Calibrated {len(params_dict)} slices.")

    #  Step 6: Arbitrage diagnostics 
    print("\nRunning arbitrage diagnostics...")
    butterfly_results = check_butterfly_arbitrage(params_dict)
    calendar_results  = check_calendar_arbitrage(params_dict)
    print_arbitrage_report(butterfly_results, calendar_results)

    n_butterfly_violations = sum(1 for v in butterfly_results.values() if v < -1e-4)
    n_calendar_violations  = sum(1 for v in calendar_results.values()  if v >  1e-6)
    print(f"\n  Butterfly violations : {n_butterfly_violations} / {len(butterfly_results)}")
    print(f"  Calendar  violations : {n_calendar_violations} / {len(calendar_results)}")

    #  Step 7: Save results 
    print("\nSaving outputs...")
    df_params = save_svi_parameters(params_dict, thetas, eta, rho)
    save_fitted_surface(params_dict, svi_data)

    # Print the parameter table
    print("\nCalibrated SVI Parameters (first 10 slices):")
    print(df_params[["T", "a", "b", "rho", "m", "sigma",
                      "atm_vol", "atm_skew"]].head(10).to_string(index=False))

    #  Step 8: Visualisations 
    print("\nGenerating visualisations...")
    plot_iv_smiles(svi_data, params_dict, max_slices=12)
    plot_total_variance_surface(params_dict, max_slices=20)
    plot_iv_surface_3d(params_dict, max_slices=25)
    plot_atm_term_structure(params_dict)

    print("\n" + "=" * 65)
    print("SVI CALIBRATION COMPLETE")
    print(f"  Parameters  : {SVI_OUTPUT_DIR}/svi_parameters.csv")
    print(f"  Fitted data : {SVI_OUTPUT_DIR}/svi_fitted_surface.csv")
    print(f"  Plots       : {VIZ_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()