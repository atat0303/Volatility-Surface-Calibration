
from data_cleaning import preprocess
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

def _fit_forward_pv_from_parity_paper(K, imid, wt=None, n_atm=15,
                                     pv_bounds=(0.5, 2.0), f_bounds=None):
    """
    Fit (f, pv) to imid ≈ pv*(f-K) using only n_atm points with smallest |imid|.
    Weighted SSE if wt provided.
    """
    K = np.asarray(K, float)
    imid = np.asarray(imid, float)
    if wt is None:
        wt = np.ones_like(imid)
    else:
        wt = np.asarray(wt, float)

    # choose near-ATM: smallest |imid|
    idx = np.argsort(np.abs(imid))
    idx = idx[:min(n_atm, len(idx))]
    K0 = K[idx]
    im0 = imid[idx]
    w0 = wt[idx]

    # guesses like the paper
    pv0 = 1.0
    f0 = float(np.mean(im0 + K0))  # because imid + K ≈ f when pv≈1

    if f_bounds is None:
        f_bounds = (float(np.min(K0)), float(np.max(K0)))

    def obj(x):
        f, pv = float(x[0]), float(x[1])
        err = pv * (f - K0) - im0
        return float(np.sum(w0 * err * err))

    res = minimize(
        obj,
        x0=np.array([f0, pv0]),
        method="L-BFGS-B",
        bounds=[f_bounds, pv_bounds],
    )

    f_hat, pv_hat = float(res.x[0]), float(res.x[1])
    sse = float(res.fun)
    return f_hat, pv_hat, sse, idx

def infer_forward_discount_from_parity_best(
    df: pd.DataFrame,
    sofr_rate: float | None = None,     # <-- NEW (decimal, e.g. 0.0235)
    use_sofr_discount: bool = False,
    *,
    T_col="T",
    K_col="K",
    S_col="S",
    Cmid_col="C_MID",
    Pmid_col="P_MID",
    Cbid_col="C_BID",
    Cask_col="C_ASK",
    Pbid_col="P_BID",
    Pask_col="P_ASK",
    # Filtering / robustness knobs
    mny_max=0.08,          # |ln(K/S)| <= 12% is a solid "trusted" band for parity
    min_pts=10,            # per maturity
    spread_cap_frac=0.20,  # drop quotes where (Csprd+Psprd) / max(1, Cmid+Pmid) > cap
    w_floor=1e-10,         # prevent zero weights
    trim_frac=0.05,        # robust second-pass trimming by abs residual (10% each tail)
):
    """
    Estimate D(T), F(T), and implied r_eff(T), q_eff(T) from put-call parity using
    weighted least squares on:
        y = C - P = alpha + beta*K
    where:
        beta = -D
        alpha = A = S*e^{-qT} = D*F

    Outputs per maturity:
        D, F, A, r_eff, q_eff, carry_eff, plus fit diagnostics.
    """

    def _wls_fit(K, y, w):
        """
        Weighted LS for y = a + b*K.
        Returns a, b, residuals, rmse_w.
        """
        # X = [1, K]
        X0 = np.ones_like(K)
        X1 = K

        # Apply sqrt(w) to rows
        sw = np.sqrt(w)
        Xw0 = X0 * sw
        Xw1 = X1 * sw
        yw = y * sw

        # Solve (X'WX) beta = X'Wy via lstsq on weighted design
        Xw = np.column_stack([Xw0, Xw1])
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        a, b = float(coef[0]), float(coef[1])

        yhat = a + b * K
        resid = y - yhat
        rmse_w = float(np.sqrt(np.sum(w * resid * resid) / max(np.sum(w), 1e-16)))
        return a, b, resid, rmse_w

    rows = []
    d = df.copy()

    # basic required columns
    req = [T_col, K_col, S_col, Cmid_col, Pmid_col]
    missing = [c for c in req if c not in d.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    for T, g in d.groupby(T_col):
        T = float(T)
        if not np.isfinite(T) or T <= 0:
            continue

        g = g.dropna(subset=[K_col, S_col, Cmid_col, Pmid_col]).copy()
        if len(g) < min_pts:
            continue

        # Ensure positive-ish mids (parity inference is very sensitive to bad zeros)
        g = g[(g[Cmid_col] > 0) & (g[Pmid_col] > 0)].copy()
        if len(g) < min_pts:
            continue

        S_ref = float(np.nanmedian(g[S_col].to_numpy()))
        if not np.isfinite(S_ref) or S_ref <= 0:
            continue

        # Trusted moneyness band
        K = g[K_col].to_numpy(dtype=float)
        mny = np.abs(np.log(np.maximum(K, 1e-16) / S_ref))
        g = g.loc[mny <= mny_max].copy()
        if len(g) < min_pts:
            continue

        # Build spreads if available; else fall back to equal weights
        have_spreads = all(c in g.columns for c in [Cbid_col, Cask_col, Pbid_col, Pask_col])
        # Build parity target
        imid = (g[Cmid_col] - g[Pmid_col]).to_numpy(dtype=float)
        K = g[K_col].to_numpy(dtype=float)

        # weights (reuse yours)
        if have_spreads:
            Csprd = (g[Cask_col] - g[Cbid_col]).to_numpy(dtype=float)
            Psprd = (g[Pask_col] - g[Pbid_col]).to_numpy(dtype=float)
            sprd_y = np.maximum(Csprd + Psprd, 1e-6)
            wt = 1.0 / (sprd_y * sprd_y)
            wt = np.maximum(wt, w_floor)
        else:
            wt = np.ones_like(imid)

        # --- PAPER FIT: forward f and discount pv
        f_bounds = (float(np.min(K)), float(np.max(K)))   # like the paper: [minK, maxK]
        f_hat, pv_hat, sse, idx_atm = _fit_forward_pv_from_parity_paper(
            K, imid, wt=wt, n_atm=15, pv_bounds = (np.exp(-0.20*T), 1.05), f_bounds=f_bounds
        )

        D = pv_hat              # discount factor
        F = f_hat               # forward
        A = D * F               # = S*exp(-qT) in theory

        # sanity
        if not np.isfinite(D) or D <= 0: 
            continue
        if not np.isfinite(F) or F <= 0:
            continue
        if not np.isfinite(A) or A <= 0:
            continue

        r_eff = -np.log(D) / T
        q_eff = -np.log(A / S_ref) / T
        carry_eff = r_eff - q_eff

        rows.append({
            "T": float(T),
            "S_ref": float(S_ref),
            "D": float(D),
            "A": float(A),
            "F": float(F),
            "r_eff": float(r_eff),
            "q_eff": float(q_eff),
            "carry_eff": float(carry_eff),
            "rmse_w": float(np.sqrt(sse / max(np.sum(wt[idx_atm]), 1e-16))),  # optional
            "n": int(len(g)),
            "n_used": int(len(idx_atm)),
        })    
    out = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
    return out
def main():  
  DATA_FILE_NAME = os.getenv("DATA_FILE_NAME", "spx_eod_202301.txt")
  DATA_DIR = os.getenv("DATA_DIR", "data/raw/spx_eod_2023q1-cfph7w")
  DATA_INPUT = os.path.join(DATA_DIR, DATA_FILE_NAME)
  df = pd.read_csv(DATA_INPUT)
  df_proc = preprocess(df)
  fc = infer_forward_discount_from_parity_best(df_proc)
