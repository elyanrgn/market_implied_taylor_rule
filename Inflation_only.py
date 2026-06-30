"""
=============================================================================
Version réduite HICP-only — 2 régressions OLS (pas de PMI, pas d'output)
=============================================================================
Étape 1 : pi_{tau+m} ~ w_HICP + wt_HICP + pilag_HICP + oislvl_HICP
          -> donne gamma_pi (empreinte inflation HICP)

Étape 2 : d_ois ~ gamma_pi * (w_HICP - wt_HICP)
          -> donne phi = beta * gamma_pi directement

AVANTAGES par rapport à la version jointe :
- Exactement identifié -> plus de J-test pathologique
- gamma_y absent -> plus de multicolinéarité sur l'output
- Une seule empreinte -> optimization triviale (OLS pur, pas de GMM)

LIMITE : phi = beta * gamma_pi n'est pas séparé. Pour récupérer beta seul
il faut diviser par gamma_pi (delta méthode pour les SE).
=============================================================================
"""

import numpy as np
from GMM_estimation import (
    NW_LAGS,
    build_event_table_daily,
    HORIZONS_M_DAILY,
    inspect_reference_months,
)
import statsmodels.api as sm
import pandas as pd

# =========================================================================
# 1. PANEL HICP-ONLY (depuis build_monthly_panel_daily existant)
# =========================================================================


def build_hicp_panel(events, m):
    """
    Extrait uniquement les colonnes HICP de build_monthly_panel_daily.
    Index : tau (mois de référence).
    """
    chg_col = f"OIS_{m}m_chg"
    level_col = f"OIS_{m}m_level_prev"

    df = events.loc[
        events["indicator"] == "HICP",
        [
            "tau",
            "actual",
            "consensus",
            "pi_lag1",
            f"pi_target_m{m}",
            chg_col,
            level_col,
        ],
    ].copy()

    df = df.rename(
        columns={
            "actual": "w_HICP",
            "consensus": "wt_HICP",
            "pi_lag1": "pilag_HICP",
            f"pi_target_m{m}": "pitgt",
            chg_col: "dois",
            level_col: "oislvl",
        }
    )

    df = df.dropna().drop_duplicates(subset="tau", keep="first")

    if len(df) < 20:
        import warnings

        warnings.warn(f"[HICP-only, m={m}M] seulement {len(df)} obs.")

    return df.set_index("tau")


# =========================================================================
# 2. ÉTAPE 1 — EMPREINTE INFLATION (gamma_pi)
# =========================================================================


def estimate_pi_hicp(panel):
    """
    pi_{tau+m} ~ cst + w_HICP + wt_HICP + pilag_HICP + oislvl
    gamma_pi = coeff de w_HICP (réaction de l'inflation au chiffre annoncé)
    """
    y = panel["pitgt"]
    X = sm.add_constant(panel[["w_HICP", "wt_HICP", "pilag_HICP", "oislvl"]])
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})


# =========================================================================
# 3. ÉTAPE 2 — RÉACTION OIS (phi = beta * gamma_pi)
# =========================================================================


def estimate_ois_hicp(panel, gamma_pi):
    """
    d_ois ~ cst + phi * gamma_pi * (w_HICP - wt_HICP)
    Renvoie phi (= beta estimé si gamma_pi est connu).
    """
    surprise = panel["w_HICP"] - panel["wt_HICP"]
    X = sm.add_constant(
        pd.DataFrame(
            {
                "X_pi": gamma_pi * surprise,
            },
            index=panel.index,
        )
    )

    return sm.OLS(panel["dois"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})


# =========================================================================
# 4. DRIVER PAR MATURITÉ
# =========================================================================


def estimate_hicp_only(events, m, verbose=True):
    """
    Estimation OLS deux étapes HICP-only pour la maturité m.

    Returns
    -------
    dict avec model_pi, model_ois, gamma_pi, phi (=beta*gamma_pi),
         beta (= phi / gamma_pi, delta méthode), panel
    """
    panel = build_hicp_panel(events, m)

    if verbose:
        print(f"\n=== [HICP-only] Maturité {m}M — T = {len(panel)} obs. ===")

    if len(panel) < 10:
        return None

    # Étape 1
    model_pi = estimate_pi_hicp(panel)
    gamma_pi = model_pi.params["w_HICP"]

    # Étape 2
    model_ois = estimate_ois_hicp(panel, gamma_pi)
    phi = model_ois.params["X_pi"]
    se_phi = model_ois.bse["X_pi"]

    # beta = phi / gamma_pi (delta method : se_beta ≈ se_phi / |gamma_pi|)
    beta_dm = phi / gamma_pi if abs(gamma_pi) > 1e-6 else np.nan
    se_beta_dm = se_phi / abs(gamma_pi) if abs(gamma_pi) > 1e-6 else np.nan

    if verbose:
        t_g = gamma_pi / model_pi.bse["w_HICP"]
        print(
            f"  gamma_pi = {gamma_pi:8.4f}  "
            f"(se={model_pi.bse['w_HICP']:.4f}, t={t_g:.2f}, "
            f"R²={model_pi.rsquared:.3f})"
        )
        t_phi = phi / se_phi if se_phi > 0 else np.nan
        print(
            f"  phi      = {phi:8.4f}  "
            f"(se naïve={se_phi:.4f}, t={t_phi:.2f})  ← SE INVALIDE, cf. bootstrap"
        )
        t_b = beta_dm / se_beta_dm if se_beta_dm > 0 else np.nan
        print(
            f"  beta (delta-method) = {beta_dm:8.3f}  "
            f"(se≈{se_beta_dm:.3f}, t≈{t_b:.2f})"
        )

    return {
        "model_pi": model_pi,
        "model_ois": model_ois,
        "gamma_pi": gamma_pi,
        "phi": phi,
        "se_phi_naive": se_phi,
        "beta_dm": beta_dm,
        "se_beta_dm": se_beta_dm,
        "panel": panel,
    }


# =========================================================================
# 5. BOOTSTRAP — SE correctes (Pagan 1986)
# =========================================================================


def bootstrap_se_hicp(events, m, n_boot=500, seed=0, verbose=True):
    """
    Bootstrap par blocs mensuels pour corriger les SE (régresseur généré).
    Ré-échantillonne des tau entiers et refait les 2 régressions.
    """
    panel_full = build_hicp_panel(events, m)
    months = panel_full.index.to_numpy()
    rng = np.random.default_rng(seed)

    phis, betas = [], []
    for _ in range(n_boot):
        idx = rng.choice(months, size=len(months), replace=True)
        panel_b = panel_full.loc[idx].reset_index(drop=True)
        try:
            mp = estimate_pi_hicp(panel_b)
            gp = mp.params["w_HICP"]
            mo = estimate_ois_hicp(panel_b, gp)
            phis.append(mo.params["X_pi"])
            betas.append(mo.params["X_pi"] / gp if abs(gp) > 1e-6 else np.nan)
        except Exception:
            continue

    phis = np.array(phis)
    betas = np.array([b for b in betas if not np.isnan(b)])

    result = {
        "phi_se": phis.std(ddof=1),
        "phi_ci90": np.percentile(phis, [5, 95]),
        "beta_se": betas.std(ddof=1),
        "beta_ci90": np.percentile(betas, [5, 95]),
        "n_valid": len(phis),
        "n_requested": n_boot,
    }
    if verbose:
        print(f"  [bootstrap, {result['n_valid']}/{n_boot} valides]")
        print(
            f"    se(phi)  = {result['phi_se']:.4f}  "
            f"IC90 = {np.round(result['phi_ci90'], 4)}"
        )
        print(
            f"    se(beta) = {result['beta_se']:.3f}   "
            f"IC90 = {np.round(result['beta_ci90'], 3)}"
        )
    return result


# =========================================================================
# 6. POINT D'ENTRÉE
# =========================================================================


def run_hicp_only(
    df_surprise_daily, df_inflation, df_output, bootstrap=True, n_boot=300
):
    """
    Pipeline complet HICP-only sur toutes les maturités.

    Parameters
    ----------
    df_surprise_daily : sortie de build_df_surprise(df_announcements, ois_daily)
    df_inflation      : DataFrame [Date, pi_realized]
    df_output         : inutilisé ici, conservé pour cohérence d'interface
    bootstrap         : si True, calcule les SE bootstrap
    n_boot            : nombre de tirages

    Returns
    -------
    results : dict {m: fit_dict}
    events  : table d'événements enrichie
    """
    # build_event_table_daily construit HICP + PMI, mais on n'utilisera que HICP
    events = build_event_table_daily(df_surprise_daily, df_inflation, df_output)
    inspect_reference_months(events)

    results = {}
    for m in HORIZONS_M_DAILY:
        fit = estimate_hicp_only(events, m)
        if fit is None:
            continue
        if bootstrap:
            fit["bootstrap"] = bootstrap_se_hicp(events, m, n_boot=n_boot)
        results[m] = fit

    return results, events
