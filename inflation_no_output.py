"""
Version alignée sur 02_layerA_static_beta.py :
  - psi^(h)     = coeff de surprise dans d_ois ~ surprise    (bp/pp)
  - gamma_pi^(h)= coeff de surprise dans pi_{t+h} ~ surprise + pi_lag
  - beta^(h)    = psi^(h) / gamma_pi^(h)   [delta = 0 implicite]

Pas d'équation y, pas de delta séparé : on suppose que la réaction OIS
est intégralement attribuée à l'inflation (ou que delta*gamma_y est
absorbé dans psi et reste une limite de l'estimation baseline).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from GMM_estimation import (
    NW_LAGS,
    HORIZONS_M_DAILY,
    inspect_reference_months,
    build_event_table_daily,
)
from Inflation_only import build_hicp_panel

# ÉTAPE 1a — psi^(h) : réaction OIS à la surprise (bp/pp)


def estimate_psi(panel):
    """
    d_ois[bp] ~ cst + psi * surprise_HICP
    surprise = actual - consensus
    Exact même spec que le code de référence.
    """
    surprise = panel["w_HICP"] - panel["wt_HICP"]
    X = sm.add_constant(pd.DataFrame({"surprise": surprise}, index=panel.index))
    return sm.OLS(panel["dois"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})


# ÉTAPE 1b — gamma_pi^(h) : empreinte inflation


def estimate_gamma_pi(panel):
    """
    pi_{t+h} ~ cst + gamma_pi * surprise_HICP + pi_lag
    Même spec que le code de référence (ctrl=True).
    """
    surprise = panel["w_HICP"] - panel["wt_HICP"]
    X = sm.add_constant(
        pd.DataFrame(
            {
                "surprise": surprise,
                "pi_lag": panel["pilag_HICP"],
            },
            index=panel.index,
        )
    )
    return sm.OLS(panel["pitgt"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})


# ÉTAPE 2 — beta^(h) = psi^(h) / gamma_pi^(h), delta method
def compute_beta(psi, se_psi, gamma, se_gamma):
    """
    Delta method : Var(beta) = Var(psi)/gamma^2 + psi^2 * Var(gamma)/gamma^4
    Hypothèse : Cov(psi, gamma) = 0 (régressions sur des sorties disjointes).
    """
    if abs(gamma) < 1e-6:
        return np.nan, np.nan
    beta = psi / gamma
    var = se_psi**2 / gamma**2 + psi**2 * se_gamma**2 / gamma**4
    return beta, np.sqrt(var)


# DRIVER
def estimate_layer_a(events, m, verbose=True):
    """
    Estimation HICP-only, séparation psi / gamma_pi propre.
    Pas d'équation y, pas de delta — identique à 02_layerA_static_beta.py.
    """
    panel = build_hicp_panel(events, m)  # inchangé

    if len(panel) < 10:
        return None

    model_psi = estimate_psi(panel)
    model_gamma = estimate_gamma_pi(panel)

    psi = model_psi.params["surprise"]
    se_psi = model_psi.bse["surprise"]
    gamma = model_gamma.params["surprise"]
    se_gamma = model_gamma.bse["surprise"]
    beta, se_beta = compute_beta(psi, se_psi, gamma, se_gamma)

    if verbose:
        t_psi = psi / se_psi if se_psi > 0 else np.nan
        t_g = gamma / se_gamma if se_gamma > 0 else np.nan
        t_b = beta / se_beta if se_beta > 0 else np.nan
        print(f"\n=== [layer A] Maturité {m}M — T = {len(panel)} obs. ===")
        print(f"  psi      = {psi:8.3f} bp/pp  (se={se_psi:.3f}, t={t_psi:.2f})")
        print(
            f"  gamma_pi = {gamma:8.4f}       (se={se_gamma:.4f}, t={t_g:.2f}, "
            f"R²={model_gamma.rsquared:.3f})"
        )
        print(f"  beta     = {beta:8.3f}        (se_dm={se_beta:.3f}, t={t_b:.2f})")

    return {
        "model_psi": model_psi,
        "model_gamma": model_gamma,
        "psi": psi,
        "se_psi": se_psi,
        "gamma_pi": gamma,
        "se_gamma": se_gamma,
        "beta": beta,
        "se_beta": se_beta,
        "panel": panel,
    }


# =========================================================================
# BOOTSTRAP — SE correctes
# =========================================================================


def bootstrap_layer_a(events, m, n_boot=500, seed=0, verbose=True):
    panel_full = build_hicp_panel(events, m)
    months = panel_full.index.to_numpy()
    rng = np.random.default_rng(seed)

    psis, betas = [], []
    for _ in range(n_boot):
        idx = rng.choice(months, size=len(months), replace=True)
        panel_b = panel_full.loc[idx].reset_index(drop=True)
        try:
            mp = estimate_psi(panel_b)
            mg = estimate_gamma_pi(panel_b)
            p = mp.params["surprise"]
            g = mg.params["surprise"]
            psis.append(p)
            if abs(g) > 1e-6:
                betas.append(p / g)
        except Exception:
            continue

    psis = np.array(psis)
    betas = np.array(betas)
    result = {
        "psi_se": psis.std(ddof=1),
        "psi_ci90": np.percentile(psis, [5, 95]),
        "beta_se": betas.std(ddof=1),
        "beta_ci90": np.percentile(betas, [5, 95]),
        "n_valid": len(psis),
    }
    if verbose:
        print(
            f"  [bootstrap {result['n_valid']}/{n_boot}]  "
            f"se(psi)={result['psi_se']:.3f}  "
            f"se(beta)={result['beta_se']:.3f}  "
            f"IC90(beta)={np.round(result['beta_ci90'], 3)}"
        )
    return result


# POINT D'ENTRÉE


def run_layer_a(df_surprise_daily, df_inflation, df_output, bootstrap=True, n_boot=300):
    events = build_event_table_daily(df_surprise_daily, df_inflation, df_output)
    inspect_reference_months(events)

    results = {}
    for m in HORIZONS_M_DAILY:
        fit = estimate_layer_a(events, m)
        if fit is None:
            continue
        if bootstrap:
            fit["bootstrap"] = bootstrap_layer_a(events, m, n_boot=n_boot)
        results[m] = fit
    return results, events
