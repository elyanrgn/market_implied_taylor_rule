"""
=============================================================================
HPB (2011) — version OLS simplifiée avec orthogonalisation PMI ⊥ HICP
=============================================================================
Trois étapes :
  0. Orthogonalisation : s_PMI_orth = résidu de (s_PMI ~ s_HICP)
     -> garantit Cov(s_HICP, s_PMI_orth) = 0 -> det(Γ) >> 0

  1a. pi_{tau+m} ~ cst + s_HICP + s_PMI_orth + pi_lag
       -> gamma_pi_H, gamma_pi_P (empreintes inflation)

  1b. y_{tau+m}  ~ cst + s_HICP + s_PMI_orth + y_lag
       -> gamma_y_H, gamma_y_P (empreintes output)

  2.  dois_total ~ beta * X_pi + delta * X_y
      avec X_pi = gamma_pi_H * s_HICP + gamma_pi_P * s_PMI_orth
           X_y  = gamma_y_H  * s_HICP + gamma_y_P  * s_PMI_orth
      (empilage HICP + PMI, intercepts séparés par indicateur)
=============================================================================
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

from GMM_estimation import (
    build_event_table_daily,
    NW_LAGS,
    HORIZONS_M_DAILY,
    inspect_reference_months,
)

# =========================================================================
# 0. CONSTRUCTION DU PANEL JOINT (HICP + PMI, même index tau)
# =========================================================================


def build_orthogonal_panel(events, m):
    """
    Construit le panel mensuel avec :
      - surprises brutes s_HICP, s_PMI
      - surprise orthogonalisée s_PMI_orth = résidu(s_PMI ~ cst + s_HICP)
      - d_ois pour les deux indicateurs (en bp, après ×100 dans build_df_surprise)
    """
    chg_col = f"OIS_{m}m_chg"
    level_col = f"OIS_{m}m_level_prev"

    # --- Extraction HICP ---
    df_h = (
        events.loc[
            events["indicator"] == "HICP",
            [
                "tau",
                "actual",
                "consensus",
                "pi_lag1",
                f"pi_target_m{m}",
                f"y_target_m{m}",
                chg_col,
                level_col,
            ],
        ]
        .copy()
        .rename(
            columns={
                "actual": "w_HICP",
                "consensus": "wt_HICP",
                "pi_lag1": "pilag",
                f"pi_target_m{m}": "pitgt",
                f"y_target_m{m}": "ytgt",
                chg_col: "dois_HICP",
                level_col: "oislvl_HICP",
            }
        )
        .dropna()
        .drop_duplicates("tau")
        .set_index("tau")
    )

    # --- Extraction PMI ---
    df_p = (
        events.loc[
            events["indicator"] == "PMI",
            [
                "tau",
                "actual",
                "consensus",
                "y_lag2",
                chg_col,
                level_col,
            ],
        ]
        .copy()
        .rename(
            columns={
                "actual": "w_PMI",
                "consensus": "wt_PMI",
                "y_lag2": "ylag",
                chg_col: "dois_PMI",
                level_col: "oislvl_PMI",
            }
        )
        .dropna()
        .drop_duplicates("tau")
        .set_index("tau")
    )

    # --- Join interne ---
    panel = df_h.join(df_p, how="inner").sort_index()

    if len(panel) < 20:
        warnings.warn(
            f"[orth, m={m}M] seulement {len(panel)} obs. communes "
            f"HICP/PMI — identification fragile."
        )

    # --- Surprises brutes ---
    panel["s_HICP"] = panel["w_HICP"] - panel["wt_HICP"]
    panel["s_PMI"] = panel["w_PMI"] - panel["wt_PMI"]

    # --- Étape 0 : orthogonalisation s_PMI ⊥ s_HICP ---
    # Régression s_PMI ~ cst + s_HICP
    X_orth = sm.add_constant(panel["s_HICP"])
    res_orth = sm.OLS(panel["s_PMI"], X_orth).fit()
    panel["s_PMI_orth"] = res_orth.resid

    # Vérification : corrélation résiduelle doit être ~0
    corr_check = panel["s_HICP"].corr(panel["s_PMI_orth"])
    if abs(corr_check) > 0.01:
        warnings.warn(
            f"[orth, m={m}M] Corrélation résiduelle s_HICP/s_PMI_orth = "
            f"{corr_check:.4f} ≠ 0 — problème d'orthogonalisation."
        )

    return panel, res_orth


# =========================================================================
# 1. ÉTAPES 1a ET 1b — EMPREINTES (gamma_pi, gamma_y)
# =========================================================================


def estimate_fingerprints(panel):
    """
    Deux régressions OLS avec surprises orthogonalisées.
    Renvoie (model_pi, model_y, gamma_pi, gamma_y) sous forme de dicts.
    """
    regressors = ["s_HICP", "s_PMI_orth"]

    # --- Équation inflation ---
    X_pi = sm.add_constant(
        panel[regressors + ["pilag"]].rename(columns={"pilag": "pi_lag"})
    )
    model_pi = sm.OLS(panel["pitgt"], X_pi).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS}
    )

    # --- Équation output ---
    # ylag = y_lag2 PMI (disponible via join)
    X_y = sm.add_constant(
        panel[regressors + ["ylag"]].rename(columns={"ylag": "y_lag"})
    )
    model_y = sm.OLS(panel["ytgt"], X_y).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS}
    )

    gamma_pi = {
        "HICP": model_pi.params["s_HICP"],
        "PMI_orth": model_pi.params["s_PMI_orth"],
    }
    gamma_y = {
        "HICP": model_y.params["s_HICP"],
        "PMI_orth": model_y.params["s_PMI_orth"],
    }
    return model_pi, model_y, gamma_pi, gamma_y


def diagnose_identification(gamma_pi, gamma_y, m, verbose=True):
    """
    Calcule det(Γ) et le nombre de conditionnement de la matrice d'empreintes.
    Un det >> 0 est nécessaire pour que (beta, delta) soit identifié.
    """
    G = np.array(
        [
            [gamma_pi["HICP"], gamma_pi["PMI_orth"]],
            [gamma_y["HICP"], gamma_y["PMI_orth"]],
        ]
    )
    det = np.linalg.det(G)
    cond = np.linalg.cond(G)
    status = (
        "✓ identifié"
        if abs(det) > 0.1
        else "⚠ fragile" if abs(det) > 0.01 else "✗ non ident."
    )
    if verbose:
        print(f"  det(Γ) = {det:9.4f}  cond(Γ) = {cond:8.1f}  {status}")
    if abs(det) < 1e-3:
        warnings.warn(
            f"[orth, m={m}M] det(Γ)={det:.2e} ≈ 0 — (beta,delta) non identifié "
            f"même après orthogonalisation. Le signal PMI ⊥ HICP est trop faible."
        )
    return det, cond


# =========================================================================
# 2. ÉTAPE 2 — RÉGRESSION PRINCIPALE (beta, delta)
# =========================================================================


def estimate_main_orth(panel, gamma_pi, gamma_y, separate_intercepts=True):
    """
    Empile HICP et PMI. Construit X_pi et X_y comme combinaisons linéaires
    des surprises orthogonalisées pondérées par les empreintes estimées.

    Pour indicateur k ∈ {HICP, PMI} :
      X_pi^k = gamma_pi[k_orth] * s_k_orth      (k=HICP -> s_HICP, k=PMI -> s_PMI_orth)
      X_y^k  = gamma_y[k_orth]  * s_k_orth

    puis d_ois^k = beta * X_pi^k + delta * X_y^k + eta^k + eps^k
    """
    rows = []
    for indicator, s_col in [("HICP", "s_HICP"), ("PMI", "s_PMI_orth")]:
        surprise = panel[s_col]
        rows.append(
            pd.DataFrame(
                {
                    "d_ois": panel[f"dois_{indicator}"].to_numpy(),
                    "X_pi": (
                        gamma_pi[indicator if indicator == "HICP" else "PMI_orth"]
                        * surprise
                    ).to_numpy(),
                    "X_y": (
                        gamma_y[indicator if indicator == "HICP" else "PMI_orth"]
                        * surprise
                    ).to_numpy(),
                    "indicator": indicator,
                },
                index=panel.index,
            )
        )

    stacked = pd.concat(rows, ignore_index=True)

    X = stacked[["X_pi", "X_y"]].copy()
    if separate_intercepts:
        dummies = pd.get_dummies(
            stacked["indicator"], prefix="eta", drop_first=False, dtype=float
        )
        X = pd.concat([X, dummies], axis=1)
    else:
        X = sm.add_constant(X)

    model = sm.OLS(stacked["d_ois"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS}
    )
    return model, stacked


# DRIVER PAR MATURITÉ


def estimate_ols_orth(events, m, verbose=True):
    """
    Estimation OLS trois étapes avec orthogonalisation pour la maturité m.
    Retourne un dict complet avec les résultats et le diagnostic d'identification.
    """
    panel, model_orth = build_orthogonal_panel(events, m)

    if len(panel) < 10:
        return None

    if verbose:
        print(f"\n=== [OLS+orth] Maturité {m}M — T = {len(panel)} mois ===")
        r2_orth = model_orth.rsquared
        rho_raw = panel["s_HICP"].corr(panel["s_PMI"])
        print(
            f"  Corrélation brute HICP/PMI = {rho_raw:.3f}  "
            f"R²(PMI ~ HICP) = {r2_orth:.3f}  "
            f"[corrélation résiduelle ≈ 0 par construction]"
        )

    # Étape 1 : empreintes
    model_pi, model_y, gamma_pi, gamma_y = estimate_fingerprints(panel)

    if verbose:
        print(
            f"  gamma_pi : HICP={gamma_pi['HICP']:8.4f}  "
            f"PMI_orth={gamma_pi['PMI_orth']:8.4f}  "
            f"(R²={model_pi.rsquared:.3f})"
        )
        print(
            f"  gamma_y  : HICP={gamma_y['HICP']:8.4f}  "
            f"PMI_orth={gamma_y['PMI_orth']:8.4f}  "
            f"(R²={model_y.rsquared:.3f})"
        )

    # Diagnostic d'identification
    det, cond = diagnose_identification(gamma_pi, gamma_y, m, verbose)

    # Étape 2 : régression principale
    model_main, stacked = estimate_main_orth(panel, gamma_pi, gamma_y)

    beta = model_main.params["X_pi"]
    delta = model_main.params["X_y"]
    se_b = model_main.bse["X_pi"]
    se_d = model_main.bse["X_y"]

    if verbose:
        t_b = beta / se_b if se_b > 0 else np.nan
        t_d = delta / se_d if se_d > 0 else np.nan
        print(
            f"  beta  = {beta:8.3f}  (se naïve={se_b:.3f}, t={t_b:.2f})"
            f"  ← bootstrap pour SE valides"
        )
        print(f"  delta = {delta:8.3f}  (se naïve={se_d:.3f}, t={t_d:.2f})")

    return {
        "model_orth": model_orth,
        "model_pi": model_pi,
        "model_y": model_y,
        "model_main": model_main,
        "gamma_pi": gamma_pi,
        "gamma_y": gamma_y,
        "det_G": det,
        "cond_G": cond,
        "beta": beta,
        "delta": delta,
        "se_beta_naive": se_b,
        "se_delta_naive": se_d,
        "panel": panel,
    }


# BOOTSTRAP — SE correctes pour beta et delta


def bootstrap_ols_orth(events, m, n_boot=500, seed=0, verbose=True):
    """
    Bootstrap par blocs mensuels — corrige les SE dues au régresseur généré
    et à la double étape d'orthogonalisation.

    Chaque tirage refait les 3 étapes complètes :
      0. orthogonalisation (s_PMI_orth sur le sous-échantillon)
      1. empreintes (gamma_pi, gamma_y)
      2. régression principale (beta, delta)
    """
    panel_full, _ = build_orthogonal_panel(events, m)
    months = panel_full.index.to_numpy()
    rng = np.random.default_rng(seed)

    betas, deltas = [], []
    det_vals = []

    for _ in range(n_boot):
        idx = rng.choice(months, size=len(months), replace=True)
        panel_b = panel_full.loc[idx].copy()

        # Re-orthogonaliser sur le sous-échantillon
        panel_b["s_HICP"] = panel_b["w_HICP"] - panel_b["wt_HICP"]
        panel_b["s_PMI"] = panel_b["w_PMI"] - panel_b["wt_PMI"]
        try:
            X_o = sm.add_constant(panel_b["s_HICP"])
            panel_b["s_PMI_orth"] = sm.OLS(panel_b["s_PMI"], X_o).fit().resid
            panel_b = panel_b.reset_index(drop=True)

            mp, my, gp, gy = estimate_fingerprints(panel_b)
            det = np.linalg.det(
                np.array(
                    [
                        [gp["HICP"], gp["PMI_orth"]],
                        [gy["HICP"], gy["PMI_orth"]],
                    ]
                )
            )
            det_vals.append(det)

            mm, _ = estimate_main_orth(panel_b, gp, gy)
            betas.append(mm.params["X_pi"])
            deltas.append(mm.params["X_y"])
        except Exception:
            continue

    betas = np.array(betas)
    deltas = np.array(deltas)
    result = {
        "beta_se": betas.std(ddof=1),
        "beta_ci80": np.percentile(betas, [15, 85]),
        "delta_se": deltas.std(ddof=1),
        "delta_ci80": np.percentile(deltas, [15, 85]),
        "det_median": np.median(det_vals) if det_vals else np.nan,
        "n_valid": len(betas),
        "n_requested": n_boot,
    }
    if verbose:
        print(
            f"  [bootstrap {result['n_valid']}/{n_boot}]  "
            f"se(beta)={result['beta_se']:.3f}  "
            f"IC70(beta)={np.round(result['beta_ci80'], 3)}  |  "
            f"se(delta)={result['delta_se']:.3f}  "
            f"IC70(delta)={np.round(result['delta_ci80'], 3)}"
        )
        print(f"  det(Γ) médian sur bootstraps = {result['det_median']:.4f}")
    return result


# POINT D'ENTRÉE


def run_ols_orth(
    df_surprise_daily, df_inflation, df_output, bootstrap=True, n_boot=300
):
    """
    Pipeline complet : orthogonalisation PMI⊥HICP + OLS trois étapes.

    Parameters
    ----------
    df_surprise_daily : sortie de build_df_surprise(df_announcements, ois_daily)
    df_inflation      : DataFrame [Date, pi_realized]
    df_output         : DataFrame [Date, y_realized]
    """
    events = build_event_table_daily(df_surprise_daily, df_inflation, df_output)
    inspect_reference_months(events)

    results = {}
    for m in HORIZONS_M_DAILY:
        fit = estimate_ols_orth(events, m)
        if fit is None:
            continue
        if bootstrap:
            fit["bootstrap"] = bootstrap_ols_orth(events, m, n_boot=n_boot)
        results[m] = fit

    # Résumé de l'identification sur toutes les maturités
    print("\n=== Résumé identification (det(Γ) par maturité) ===")
    print(
        f"{'m':>3} {'det(Γ)':>10} {'cond(Γ)':>10} "
        f"{'beta':>8} {'se_boot':>8} {'delta':>8} {'se_boot':>8}"
    )
    for m, fit in results.items():
        bs = fit.get("bootstrap", {})
        print(
            f"{m:>3} {fit['det_G']:>10.4f} {fit['cond_G']:>10.1f} "
            f"{fit['beta']:>8.3f} {bs.get('beta_se', np.nan):>8.3f} "
            f"{fit['delta']:>8.3f} {bs.get('delta_se', np.nan):>8.3f}"
        )

    return results, events
