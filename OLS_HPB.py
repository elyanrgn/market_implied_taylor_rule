from GMM_estimation import (
    NW_LAGS,
    INDICATORS,
    build_monthly_panel_daily,
    build_event_table_daily,
    HORIZONS_M_DAILY,
    inspect_reference_months,
)

"""
=============================================================================
Réplication HPB (2011) — version OLS simplifiée, pipeline *daily*
=============================================================================
Même infrastructure que la version GMM (build_df_surprise, build_event_table_daily,
build_monthly_panel_daily). Trois régressions OLS par maturité :
  Étape 1a : pi_{tau+m} ~ w_HICP, wt_HICP, w_PMI, wt_PMI, pilag, oislvl_HICP, oislvl_PMI
  Étape 1b : y_{tau+m}  ~ w_HICP, wt_HICP, w_PMI, wt_PMI, ylag,  oislvl_HICP, oislvl_PMI
  Étape 2  : d_ois (empilé HICP+PMI) ~ gamma_pi*surprise + gamma_y*surprise (intercepts)
Les SE de l'étape 2 sont invalides (Pagan 1986) — utiliser bootstrap_se_daily().
=============================================================================
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ÉTAPE 1 — RÉGRESSIONS DES RÉALISATIONS (empreintes gamma)


def _ols_nw(y, X, lags=NW_LAGS):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def estimate_pi_equation_daily(panel, indicators=INDICATORS):
    """
    Régression de pi_{tau+m} sur les surprises et contrôles -- UNE régression
    PAR INDICATEUR (et non une régression jointe avec tous les indicateurs
    comme régresseurs simultanés). Ce choix est nécessaire dès qu'un
    indicateur est moins fréquent que les autres (ex. GDP, outer join) :
    une régression jointe imposerait une disponibilité simultanée de TOUS
    les indicateurs sur chaque ligne, ce qui reviendrait silencieusement à
    un inner join. Ici, chaque équation n'utilise que les lignes où SON
    indicateur est valide.

    Returns
    -------
    dict {indicator: résultat OLS statsmodels}
    """
    models = {}
    for indicator in indicators:
        cols_needed = [
            f"pitgt_{indicator}",
            f"w_{indicator}",
            f"wt_{indicator}",
            f"pilag_{indicator}",
            f"oislvl_{indicator}",
        ]
        sub = panel[cols_needed].dropna()
        if len(sub) < len(cols_needed) + 2:
            import warnings

            warnings.warn(
                f"[{indicator}] équation pi : seulement {len(sub)} obs. "
                f"valides après dropna -- estimation potentiellement instable."
            )
        y = sub[f"pitgt_{indicator}"]
        X = sub[
            [
                f"w_{indicator}",
                f"wt_{indicator}",
                f"pilag_{indicator}",
                f"oislvl_{indicator}",
            ]
        ]
        models[indicator] = _ols_nw(y, X)
    return models


def estimate_y_equation_daily(panel, indicators=INDICATORS):
    """
    Symétrique à estimate_pi_equation_daily, pour y_{tau+m} -- une
    régression par indicateur, échantillon propre à chaque indicateur.
    """
    models = {}
    for indicator in indicators:
        cols_needed = [
            f"ytgt_{indicator}",
            f"w_{indicator}",
            f"wt_{indicator}",
            f"ylag_{indicator}",
            f"oislvl_{indicator}",
        ]
        sub = panel[cols_needed].dropna()
        if len(sub) < len(cols_needed) + 2:
            import warnings

            warnings.warn(
                f"[{indicator}] équation y : seulement {len(sub)} obs. "
                f"valides après dropna -- estimation potentiellement instable."
            )
        y = sub[f"ytgt_{indicator}"]
        X = sub[
            [
                f"w_{indicator}",
                f"wt_{indicator}",
                f"ylag_{indicator}",
                f"oislvl_{indicator}",
            ]
        ]
        models[indicator] = _ols_nw(y, X)
    return models


# ÉTAPE 2 — RÉGRESSION PRINCIPALE (beta, delta)


def estimate_main_equation_daily(panel, gamma_pi, gamma_y, separate_intercepts=True):
    """
    Empile les observations HICP et PMI, construit les régresseurs générés
    gamma_pi[k]*(w_k - wt_k) et gamma_y[k]*(w_k - wt_k), et régresse d_ois.

    Parameters
    ----------
    panel            : sortie de build_monthly_panel_daily(events, m)
    gamma_pi, gamma_y: dicts {"HICP": float, "PMI": float}
    separate_intercepts: si True, intercepts séparés par indicateur (recommandé)

    Returns
    -------
    model   : résultat OLS statsmodels
    stacked : DataFrame empilé utilisé pour la régression
    """
    rows = []
    for indicator in INDICATORS:
        surprise = panel[f"w_{indicator}"] - panel[f"wt_{indicator}"]
        block = pd.DataFrame(
            {
                "d_ois": panel[f"dois_{indicator}"],
                "X_pi": gamma_pi[indicator] * surprise,
                "X_y": gamma_y[indicator] * surprise,
                "indicator": indicator,
            }
        )
        # Indispensable depuis l'outer join : les mois où cet indicateur
        # n'a pas d'événement (ex. GDP hors mois de publication) portent
        # des NaN qu'il faut exclure ICI, bloc par bloc -- jamais via un
        # dropna() global sur le panel avant l'étape 2.
        block = block.dropna(subset=["d_ois", "X_pi", "X_y"])
        rows.append(block)
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


# 3. DRIVER PAR MATURITÉ


def estimate_simple_daily(events, m, verbose=True):
    """
    Estimation OLS en trois étapes pour la maturité m (en mois).

    Parameters
    ----------
    events : sortie de build_event_table_daily()
    m      : horizon en mois (entier, dans HORIZONS_M)

    Returns
    -------
    dict avec model_pi, model_y, model_main, gamma_pi, gamma_y,
         beta, delta, se_beta_naive, se_delta_naive, panel
    """
    panel = build_monthly_panel_daily(events, m)

    # Vérification de cohérence des cibles réalisées -- restreinte aux
    # lignes où les DEUX indicateurs comparés sont effectivement présents
    # (avec l'outer join, comparer panel[...] - panel[...] sans filtrer les
    # NaN produirait des NaN dans diff, pas une erreur, et masquerait un
    # vrai désalignement).
    for prefix in ("pitgt", "ytgt"):
        ref_indicator = INDICATORS[0]
        for other in INDICATORS[1:]:
            both_valid = panel[
                [f"{prefix}_{ref_indicator}", f"{prefix}_{other}"]
            ].dropna()
            if both_valid.empty:
                continue
            diff = (
                (
                    both_valid[f"{prefix}_{ref_indicator}"]
                    - both_valid[f"{prefix}_{other}"]
                )
                .abs()
                .max()
            )
            if diff > 1e-8:
                raise ValueError(
                    f"[m={m}M] {prefix}_{ref_indicator} ≠ {prefix}_{other} "
                    f"(écart max={diff:.2e}, {len(both_valid)} mois communs) "
                    f"— vérifiez l'alignement de tau avant de continuer."
                )

    if verbose:
        print(f"\n=== [OLS simple, daily] Maturité {m}M — T = {len(panel)} mois ===")

    if len(panel) < 10:
        import warnings

        warnings.warn(
            f"Maturité {m}M : panel trop petit ({len(panel)} obs), estimation ignorée."
        )
        return None

    # ----- Étape 1 -----
    models_pi = estimate_pi_equation_daily(panel)
    models_y = estimate_y_equation_daily(panel)

    gamma_pi = {k: models_pi[k].params[f"w_{k}"] for k in INDICATORS}
    gamma_y = {k: models_y[k].params[f"w_{k}"] for k in INDICATORS}

    if verbose:
        print("  Empreintes (étape 1, une régression par indicateur) :")
        for k in INDICATORS:
            ratio = gamma_y[k] / gamma_pi[k] if gamma_pi[k] != 0 else np.nan
            print(
                f"    {k:5s} : gamma_pi={gamma_pi[k]:8.4f}  "
                f"gamma_y={gamma_y[k]:8.4f}  ratio(gamma_y/gamma_pi)={ratio:6.3f}"
            )
        # Diagnostic d'identification : si tous les ratios sont proches,
        # (beta, delta) restent quasi non identifiés malgré l'ajout d'un
        # indicateur supplémentaire (cf. discussion sur la colinéarité
        # intra-bloc X_pi/X_y).
        ratios = np.array(
            [gamma_y[k] / gamma_pi[k] for k in INDICATORS if gamma_pi[k] != 0]
        )
        if (
            len(ratios) >= 2
            and (ratios.max() - ratios.min()) < 0.2 * np.abs(ratios).mean()
        ):
            import warnings

            warnings.warn(
                f"Maturité {m}M : les ratios gamma_y/gamma_pi sont proches "
                f"entre indicateurs ({np.round(ratios, 3)}) -- (beta, delta) "
                f"restent faiblement identifiés malgré l'indicateur ajouté."
            )

    # ----- Étape 2 -----
    model_main, stacked = estimate_main_equation_daily(panel, gamma_pi, gamma_y)

    beta_hat = model_main.params["X_pi"]
    delta_hat = model_main.params["X_y"]
    se_beta = model_main.bse["X_pi"]
    se_delta = model_main.bse["X_y"]

    if verbose:
        t_b = beta_hat / se_beta if se_beta > 0 else np.nan
        t_d = delta_hat / se_delta if se_delta > 0 else np.nan
        print(
            f"  beta_hat  = {beta_hat:8.3f}  "
            f"(se naïve={se_beta:.3f}, t={t_b:.2f})  ← SE INVALIDE, cf. bootstrap"
        )
        print(
            f"  delta_hat = {delta_hat:8.3f}  "
            f"(se naïve={se_delta:.3f}, t={t_d:.2f})"
        )

    return {
        "models_pi": models_pi,
        "models_y": models_y,
        "model_main": model_main,
        "gamma_pi": gamma_pi,
        "gamma_y": gamma_y,
        "beta": beta_hat,
        "delta": delta_hat,
        "se_beta_naive": se_beta,
        "se_delta_naive": se_delta,
        "panel": panel,
    }


# 4. BOOTSTRAP — SE correctes (corrige le régresseur généré, Pagan 1986)


def bootstrap_se_daily(events, m, n_boot=500, seed=0, verbose=True):
    """
    Bootstrap par blocs mensuels pour corriger les SE de l'étape 2.
    Ré-échantillonne des mois entiers (index tau) et refait les 3 régressions.

    Parameters
    ----------
    events : sortie de build_event_table_daily()
    m      : horizon en mois
    n_boot : nombre de tirages bootstrap
    seed   : graine aléatoire

    Returns
    -------
    dict avec beta_se, delta_se, beta_ci90, delta_ci90, n_valid, n_requested
    """
    panel_full = build_monthly_panel_daily(events, m, verbose_coverage=verbose)
    months = panel_full.index.to_numpy()
    rng = np.random.default_rng(seed)

    betas, deltas = [], []
    for _ in range(n_boot):
        sample_idx = rng.choice(months, size=len(months), replace=True)
        panel_b = panel_full.loc[sample_idx].reset_index(drop=True)
        try:
            mp = estimate_pi_equation_daily(panel_b)
            my = estimate_y_equation_daily(panel_b)
            gp = {k: mp[k].params[f"w_{k}"] for k in INDICATORS}
            gy = {k: my[k].params[f"w_{k}"] for k in INDICATORS}
            mm, _ = estimate_main_equation_daily(panel_b, gp, gy)
            betas.append(mm.params["X_pi"])
            deltas.append(mm.params["X_y"])
        except Exception:
            continue  # quasi-colinéarité, ou tirage sans aucune obs. pour
            # un indicateur rare (GDP) -> bloc vide en étape 1/2 -> ignoré

    betas = np.array(betas)
    deltas = np.array(deltas)
    result = {
        "beta_se": betas.std(ddof=1),
        "delta_se": deltas.std(ddof=1),
        "beta_ci90": np.percentile(betas, [5, 95]),
        "delta_ci90": np.percentile(deltas, [5, 95]),
        "n_valid": len(betas),
        "n_requested": n_boot,
    }
    if result["n_valid"] < 0.8 * n_boot:
        import warnings

        warnings.warn(
            f"Maturité {m}M : seulement {result['n_valid']}/{n_boot} tirages "
            f"bootstrap valides -- signe que l'indicateur le moins fréquent "
            f"(GDP) disparaît trop souvent d'un tirage par blocs mensuels, "
            f"ce qui rend les SE bootstrap elles-mêmes peu fiables."
        )
    if verbose:
        print(f"  [bootstrap, {result['n_valid']}/{n_boot} tirages valides]")
        print(
            f"    se(beta)  = {result['beta_se']:.3f}   "
            f"IC90 = {np.round(result['beta_ci90'], 3)}"
        )
        print(
            f"    se(delta) = {result['delta_se']:.3f}   "
            f"IC90 = {np.round(result['delta_ci90'], 3)}"
        )
    return result


# 5. POINT D'ENTRÉE


def run_all_maturities_simple_daily(
    df_surprise_daily, df_inflation, df_output, bootstrap=True, n_boot=300
):
    """
    Point d'entrée unique — pipeline daily OLS simplifié.

    Parameters
    ----------
    df_surprise_daily : sortie de build_df_surprise(df_announcements, ois_daily)
    df_inflation      : DataFrame [Date, pi_realized]
    df_output         : DataFrame [Date, y_realized]
    bootstrap         : si True, calcule les SE bootstrap pour chaque maturité
    n_boot            : nombre de tirages bootstrap

    Returns
    -------
    results : dict {m: fit_dict}
    events  : table d'événements enrichie
    """
    # Construction des événements — même pipeline que la version GMM
    events = build_event_table_daily(df_surprise_daily, df_inflation, df_output)
    inspect_reference_months(events)

    results = {}
    for m in HORIZONS_M_DAILY:
        fit = estimate_simple_daily(events, m)
        if fit is None:
            continue
        if bootstrap:
            fit["bootstrap"] = bootstrap_se_daily(events, m, n_boot=n_boot)
        results[m] = fit

    return results, events
