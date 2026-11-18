
import argparse
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats


PS4_COLUMNS = {
    "nondurables": None,   # real consumption of nondurables (total)
    "services": None,      # real consumption of services (total)
    "population": None,    # population
    "income": None,        # real disposable income (total)
}


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)

def ols(y, X):
    """OLS with classical and White (HC0) covariance matrices."""
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    V_cl = (e @ e / (n - k)) * XtXi
    V_rob = XtXi @ ((X * (e**2)[:, None]).T @ X) @ XtXi
    return b, e, V_cl, V_rob


def tsls(y, X, Z):
    """2SLS: b = (X'Pz X)^-1 X'Pz y, with classical and White covariance matrices."""
    n = len(y)
    ZtZi = np.linalg.inv(Z.T @ Z)
    G = Z.T @ X                                   # m x k
    A = G.T @ ZtZi @ G                            # X'Pz X
    Ai = np.linalg.inv(A)
    b = Ai @ G.T @ ZtZi @ (Z.T @ y)
    e = y - X @ b
    V_cl = (e @ e / n) * Ai
    S = (Z * (e**2)[:, None]).T @ Z               # sum e_i^2 z_i z_i'
    V_rob = Ai @ (G.T @ ZtZi @ S @ ZtZi @ G) @ Ai
    return b, e, V_cl, V_rob


def mcs(y, X, Z, e_init):
    """Minimum chi-square estimator of Problem 1(c); returns b, V=(X'Z R^-1 Z'X)^-1, J, residuals."""
    R = (Z * (e_init**2)[:, None]).T @ Z          # R_hat = sum e_i^2 z_i z_i'
    Ri = np.linalg.inv(R)
    G = Z.T @ X
    V = np.linalg.inv(G.T @ Ri @ G)
    b = V @ G.T @ Ri @ (Z.T @ y)
    u = y - X @ b
    g = Z.T @ u
    J = g @ Ri @ g                                # value of the objective at the minimum
    return b, V, J, u


def table(names, b, *ses):
    labels = ["coef"] + [f"s.e.{i + 1}" for i in range(len(ses))]
    print(f"{'':>14}" + "".join(f"{l:>12}" for l in labels))
    for i, nm in enumerate(names):
        print(f"{nm:>14}{b[i]:12.5f}" + "".join(f"{np.sqrt(V[i, i]):12.5f}" for V in ses))


# ==========================================================================
# Problem 2
# ==========================================================================
def _pick(df, key, words, prefer_real=True):
    """Find the single column of df whose name contains one of `words`."""
    if PS4_COLUMNS.get(key):
        return PS4_COLUMNS[key]
    low = {c: str(c).lower() for c in df.columns}
    for w in words:
        hits = [c for c, s in low.items() if w in s]
        if len(hits) > 1 and prefer_real:
            hits = [c for c in hits if "real" in low[c]] or hits
        if len(hits) == 1:
            return hits[0]
    raise SystemExit(
        f"Problem 2: could not identify the '{key}' column automatically.\n"
        f"Columns found: {list(df.columns)}\n"
        f"Set PS4_COLUMNS['{key}'] at the top of this file to the right header."
    )


def load_ps4(path):
    df = pd.read_excel(path)
    nd = _pick(df, "nondurables", ["nondur"])
    sv = _pick(df, "services", ["serv"])
    pop = _pick(df, "population", ["pop"], prefer_real=False)
    inc = _pick(df, "income", ["disp", "dpi", "income"])
    print(f"Columns used: nondurables='{nd}', services='{sv}', population='{pop}', income='{inc}'")
    d = df[[nd, sv, pop, inc]].apply(pd.to_numeric, errors="coerce").dropna()
    C = ((d[nd] + d[sv]) / d[pop]).to_numpy()     # real consumption per capita
    Y = (d[inc] / d[pop]).to_numpy()              # real disposable income per capita
    print(f"Observations: {len(C)} (rows assumed to be in chronological order)")
    return C, Y


def problem2(path):
    hr("PROBLEM 2: Hall's hypothesis and the Campbell-Mankiw regression")
    C, Y = load_ps4(path)
    N = len(C)

    # ---- (a) C_t on 4 lags; test beta2 = beta3 = beta4 = 0 -----------------
    print("\n(a) C_t = b0 + b1 C_{t-1} + b2 C_{t-2} + b3 C_{t-3} + b4 C_{t-4} + e_t")
    y = C[4:]
    X = np.column_stack([np.ones(N - 4), C[3:-1], C[2:-2], C[1:-3], C[:-4]])
    n, k = X.shape
    b, e, V_cl, V_rob = ols(y, X)
    table(["const", "C(t-1)", "C(t-2)", "C(t-3)", "C(t-4)"], b, V_cl, V_rob)
    print("(s.e.1 = classical, s.e.2 = White)")

    # restricted model: only constant and C_{t-1}
    _, e_r, _, _ = ols(y, X[:, :2])
    q = 3
    F = ((e_r @ e_r - e @ e) / q) / (e @ e / (n - k))
    print(f"\nH0: b2=b3=b4=0   F({q},{n - k}) = {F:.3f},  p-value = {stats.f.sf(F, q, n - k):.4f}")
    R = np.zeros((q, k)); R[0, 2] = R[1, 3] = R[2, 4] = 1.0
    d = R @ b
    W = d @ np.linalg.inv(R @ V_rob @ R.T) @ d
    pW = stats.chi2.sf(W, q)
    print(f"White-robust Wald chi2({q}) = {W:.3f},  p-value = {pW:.4f}")
    if min(stats.f.sf(F, q, n - k), pW) < 0.05:
        print("=> Reject at 5%: lagged consumption helps predict C_t, evidence AGAINST Hall's hypothesis.")
    else:
        print("=> Cannot reject at 5%: no extra predictive power of lags, consistent with Hall's hypothesis.")

    # ---- (c) log-growth regression by 2SLS -------------------------------
    print("\n(c) dlogC_t = g0 + g1 dlogY_t + e_t, instruments: const, dlogC_{t-2..t-5}")
    dc = np.r_[np.nan, np.diff(np.log(C))]        # dc[t] = log(C_t / C_{t-1})
    dy = np.r_[np.nan, np.diff(np.log(Y))]
    t = np.arange(6, N)                           # need dc_{t-5}, so t >= 6
    yv = dc[t]
    X = np.column_stack([np.ones(len(t)), dy[t]])
    Z = np.column_stack([np.ones(len(t)), dc[t - 2], dc[t - 3], dc[t - 4], dc[t - 5]])
    n = len(t); k = X.shape[1]; m = Z.shape[1]
    print(f"Sample size n = {n}, regressors k = {k}, instruments m = {m} (incl. constant)")

    b_ols, _, Vc_ols, Vr_ols = ols(yv, X)
    b_iv, e_iv, Vc_iv, Vr_iv = tsls(yv, X, Z)
    print("\nOLS (inconsistent, see part (b)):")
    table(["const", "dlogY"], b_ols, Vc_ols, Vr_ols)
    print("\n2SLS:")
    table(["const", "dlogY"], b_iv, Vc_iv, Vr_iv)
    print("(s.e.1 = classical, s.e.2 = White)")
    z = b_iv[1] / np.sqrt(Vr_iv[1, 1])
    print(f"\n2SLS gamma1 = {b_iv[1]:.4f}, White s.e. = {np.sqrt(Vr_iv[1, 1]):.4f}, "
          f"t = {z:.2f}, 95% CI = [{b_iv[1] - 1.96 * np.sqrt(Vr_iv[1, 1]):.3f}, "
          f"{b_iv[1] + 1.96 * np.sqrt(Vr_iv[1, 1]):.3f}]")
    print("gamma1 estimates the share of consumers who spend their current income.")

    # ---- (d) exogeneity of dlogY and overidentifying restrictions --------
    print("\n(d1) Durbin-Wu-Hausman test that dlogY is exogenous (control-function form)")
    pi = np.linalg.solve(Z.T @ Z, Z.T @ X[:, 1])
    v = X[:, 1] - Z @ pi                          # first-stage residual
    print(f"First stage: partial F for the 4 excluded instruments = ", end="")
    _, e1, _, _ = ols(X[:, 1], Z)
    _, e0, _, _ = ols(X[:, 1], Z[:, :1])
    F1 = ((e0 @ e0 - e1 @ e1) / (m - 1)) / (e1 @ e1 / (n - m))
    print(f"{F1:.2f}" + ("   (< 10: weak instruments, 2SLS results unreliable)" if F1 < 10 else ""))
    Xa = np.column_stack([X, v])
    ba, _, Vca, Vra = ols(yv, Xa)
    p_dwh = []
    for lab, V in [("classical", Vca), ("White", Vra)]:
        tt = ba[2] / np.sqrt(V[2, 2])
        p_dwh.append(2 * stats.norm.sf(abs(tt)))
        print(f"  coef on first-stage residual = {ba[2]:.4f}, {lab} t = {tt:.2f}, p-value = {p_dwh[-1]:.4f}")
    if min(p_dwh) < 0.05:
        print("=> Reject exogeneity at 5%: dlogY is endogenous, OLS is inconsistent, 2SLS is needed.")
    else:
        print("=> Cannot reject exogeneity of dlogY at 5%.")

    print("\n(d2) Test of the overidentifying restrictions (m - k = %d degrees of freedom)" % (m - k))
    sargan = (e_iv @ Z @ np.linalg.solve(Z.T @ Z, Z.T @ e_iv)) / (e_iv @ e_iv / n)
    print(f"  Sargan (homoskedastic):   {sargan:8.3f},  p-value = {stats.chi2.sf(sargan, m - k):.4f}")
    b_m, V_m, J, _ = mcs(yv, X, Z, e_iv)
    print(f"  Hansen J / min chi-square (robust): {J:8.3f},  p-value = {stats.chi2.sf(J, m - k):.4f}")
    print(f"  (Q1(c) estimator: gamma1_MCS = {b_m[1]:.4f}, s.e. = {np.sqrt(V_m[1, 1]):.4f})")
    print("  Prefer the robust J: Sargan assumes homoskedasticity and over-rejects without it.")
    if stats.chi2.sf(J, m - k) < 0.05:
        print("=> Reject: at least one instrument is invalid (or the model is misspecified).")
    else:
        print("=> Cannot reject: the lagged-growth instruments are consistent with being valid.")


# ==========================================================================
# Problem 3
# ==========================================================================
def problem3(path):
    hr("PROBLEM 3(d): random walk with drift, ML estimates on the S&P 500")
    df = pd.read_excel(path)
    df.columns = ["date", "sp"]
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    assert df["sp"].notna().all() and (df["sp"] > 0).all(), "missing or non-positive index levels"

    x = np.log(df["sp"].to_numpy() / df["sp"].iloc[0])        # x_0 = 0
    dx = np.diff(x)
    T = len(dx)

    delta = dx.mean()                                          # = x_T / T
    sigma2 = np.mean((dx - delta) ** 2)                        # ML: divide by T
    sigma = np.sqrt(sigma2)
    se_delta = sigma / np.sqrt(T)
    se_sigma2 = sigma2 * np.sqrt(2.0 / T)

    print(f"Sample: {df['date'].iloc[0]:%b %Y} to {df['date'].iloc[-1]:%b %Y}, T = {T} monthly changes\n")
    print(f"delta_hat  (monthly drift)     = {delta: .6f}  (s.e. {se_delta:.6f})")
    print(f"sigma2_hat (monthly variance)  = {sigma2: .6e}  (s.e. {se_sigma2:.3e})")
    print(f"sigma_hat  (monthly std. dev.) = {sigma: .6f}")
    print(f"Annualized: drift = {12 * delta:.4f} ({100 * 12 * delta:.2f}%/yr), "
          f"volatility = {np.sqrt(12) * sigma:.4f} ({100 * np.sqrt(12) * sigma:.2f}%/yr)")
    print(f"t-stat for delta = 0: {delta / se_delta:.2f}")

    assert np.isclose(delta, x[-1] / T)
    ll = lambda d, s2: -0.5 * T * np.log(2 * np.pi * s2) - 0.5 * np.sum((dx - d) ** 2) / s2
    for d, s2 in [(delta * 1.1, sigma2), (delta, sigma2 * 1.1), (delta * 0.9, sigma2 * 0.9)]:
        assert ll(d, s2) < ll(delta, sigma2)


# ==========================================================================
# Problem 4
# ==========================================================================
def problem4(r=3, p=0.4, n=1000, reps=20000, big=2_000_000, seed=1):
    hr("PROBLEM 4: Poisson ML when the truth is Negative Binomial NB(r, p)")
    rng = np.random.default_rng(seed)

    # ---- (a) correctly specified Poisson: asymptotic variance of P0_hat ----
    lam0 = 2.0
    ybar = rng.poisson(lam0, size=(reps, n)).mean(axis=1)
    P0_hat = np.exp(-ybar)
    print(f"(a) Poisson(lambda={lam0}), n={n}: lambda_hat = ybar, P0_hat = exp(-ybar)")
    print(f"    delta method: sqrt(n)(P0_hat - P0) -> N(0, lambda*exp(-2*lambda)) = "
          f"N(0, {lam0 * np.exp(-2 * lam0):.5f})")
    print(f"    Monte Carlo variance of sqrt(n)(P0_hat - P0) over {reps} reps: "
          f"{n * P0_hat.var():.5f}")

    # ---- (b) misspecified: limits -------------------------------------------
    lam_star = r * (1 - p) / p
    var_y = r * (1 - p) / p**2
    y = rng.negative_binomial(r, p, size=big)      # pmf C(y+r-1,y) (1-p)^y p^r, as in the problem
    print(f"\n(b) Truth NB(r={r}, p={p}):  E y = r(1-p)/p = {lam_star:.4f},  Var y = r(1-p)/p^2 = {var_y:.4f}")
    print(f"    lambda_hat = ybar in a sample of {big:,}: {y.mean():.4f}   (limit {lam_star:.4f})")
    print(f"    P0_hat = exp(-ybar) -> exp(-r(1-p)/p) = {np.exp(-lam_star):.5f}   "
          f"(simulated {np.exp(-y.mean()):.5f})")
    print(f"    true P0 = p^r = {p**r:.5f}   => the Poisson-based estimator is asymptotically biased "
          f"downward (exp(-r(1-p)/p) < p^r for p<1)")

    # ---- (c) KL minimiser ---------------------------------------------------
    ys = np.arange(0, 400)
    g = stats.nbinom.pmf(ys, r, p)
    kl = lambda lam: np.sum(g * (np.log(g) - stats.poisson.logpmf(ys, lam)))
    res = optimize.minimize_scalar(kl, bounds=(0.1, 50), method="bounded", options={"xatol": 1e-10})
    print(f"\n(c) argmin_lambda KL(g, f_lambda) = {res.x:.5f}   (r(1-p)/p = {lam_star:.5f})")
    print("    Same as the ML limit in (b): the QMLE converges to the KL-minimising (pseudo-true) value.")

    # ---- (d) score variance vs. minus expected Hessian ----------------------
    score = -1 + y / lam_star
    hess = -y / lam_star**2
    print(f"\n(d) at lambda* = {lam_star:.4f}:")
    print(f"    Var(score)     = Var(y)/lambda*^2 = 1/(r(1-p)) = {1 / (r * (1 - p)):.5f}   "
          f"(simulated {score.var():.5f})")
    print(f"    -E(Hessian)    = E(y)/lambda*^2   = p/(r(1-p)) = {p / (r * (1 - p)):.5f}   "
          f"(simulated {-hess.mean():.5f})")
    print(f"    ratio Var(score)/(-E Hessian) = 1/p = {1 / p:.3f}: the information-matrix equality fails.")
    print(f"    Sandwich A^-1 B A^-1 = Var(y) = {var_y:.4f} versus the naive ML variance 1/A = lambda* = "
          f"{lam_star:.4f}  (naive s.e. too small by factor sqrt(p) = {np.sqrt(p):.3f}).")


# ==========================================================================
# Problem 5
# ==========================================================================
def problem5(sigma2=100.0, nm=100, nw=100, ybar_m=64.0, ybar_w=63.0):
    hr("PROBLEM 5: LR test of H0: beta_m <= beta_w against H1: beta_m > beta_w")
    # Unrestricted MLE: (64, 63) satisfies beta_m > beta_w, so it is the unconstrained optimum.
    # Restricted MLE (constraint binds): beta_m = beta_w = pooled mean.
    pooled = (nm * ybar_m + nw * ybar_w) / (nm + nw)
    ssr_diff = nm * (ybar_m - pooled) ** 2 + nw * (ybar_w - pooled) ** 2
    LR = ssr_diff / sigma2
    LR_formula = (ybar_m - ybar_w) ** 2 / (sigma2 * (1 / nm + 1 / nw))
    assert np.isclose(LR, LR_formula)
    print(f"(a) restricted MLE (pooled mean) = {pooled:.2f};  LR = {LR:.4f}"
          f"   [= (Ybar_m - Ybar_w)^2 / (sigma^2 (1/nm + 1/nw))]")

    # Null distribution at the least favourable point beta_m = beta_w:
    # LR = z^2 * 1{z > 0}, i.e. 1/2 chi2_0 + 1/2 chi2_1.
    crit = stats.chi2.ppf(1 - 2 * 0.05, 1)
    print(f"(b) 5% critical value of 1/2 chi2_0 + 1/2 chi2_1: {crit:.4f}  ->  "
          f"{'reject' if LR > crit else 'do not reject'} H0")
    pval = 0.5 * stats.chi2.sf(LR, 1)
    print(f"(c) p-value = 1/2 P(chi2_1 > {LR:.2f}) = {pval:.4f}")
    z = (ybar_m - ybar_w) / np.sqrt(sigma2 * (1 / nm + 1 / nw))
    print(f"    check: one-sided z-test, z = {z:.4f}, p = {stats.norm.sf(z):.4f}")
    print(f"    (for reference, a naive chi2_1 p-value would be {stats.chi2.sf(LR, 1):.4f}, which is "
          f"two-sided and too large)")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps4", default="PS4data.xls", help="data file for Problem 2")
    ap.add_argument("--sp500", default="SP500Index.xlsx", help="data file for Problem 3(d)")
    ap.add_argument("--only", nargs="*", type=int, choices=[2, 3, 4, 5], help="run only these problems")
    a = ap.parse_args()
    run = set(a.only) if a.only else {2, 3, 4, 5}

    if 2 in run:
        if os.path.exists(a.ps4):
            problem2(a.ps4)
        else:
            print(f"\n[Problem 2 skipped: data file '{a.ps4}' not found; pass it with --ps4]")
    if 3 in run:
        if os.path.exists(a.sp500):
            problem3(a.sp500)
        else:
            print(f"\n[Problem 3(d) skipped: data file '{a.sp500}' not found; pass it with --sp500]")
    if 4 in run:
        problem4()
    if 5 in run:
        problem5()


if __name__ == "__main__":
    main()
