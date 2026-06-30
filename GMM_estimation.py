"""
=============================================================================
Réplication Hamilton, Pruitt & Borger (2011, AEJ:Macro) — adaptation zone euro
=============================================================================

Système GMM à deux indicateurs (HICP Core flash, PMI) destiné à identifier le
couple (beta, delta) de la règle de Taylor perçue par le marché, séparément
pour chaque maturité OIS m in {1, 3, 6, 12} mois (1M, 3M, 6M, 1Y).

Référence des équations : Hamilton, Pruitt & Borger (2011), Sections I-II,
équations (5), (10)-(13).

-----------------------------------------------------------------------------
HYPOTHÈSES STRUCTURANTES — À VALIDER EXPLICITEMENT AVANT TOUTE INTERPRÉTATION
-----------------------------------------------------------------------------

1. MOIS DE RÉFÉRENCE (tau) DE CHAQUE ÉVÉNEMENT
   - HICP : si publié en tout début de mois (jour <= HICP_SPILLOVER_CUTOFF),
     on suppose qu'il s'agit du flash du mois PRÉCÉDENT ("fin ou début de
     mois" mentionné). Sinon, flash du mois en cours.
   - PMI  : rattaché au mois PRÉCÉDENT par défaut (REF_MONTH_OFFSET['PMI']=-1)
     car "toujours en début de mois" correspond typiquement au calendrier du
     PMI FINAL, pas du PMI flash (qui sort en semaine 3 du mois en cours).
     >>> Si votre df_ois_pmi est en réalité le PMI flash malgré une date de
     publication en début de mois, passez REF_MONTH_OFFSET['PMI'] = 0 et
     ré-inspectez inspect_reference_months() avant de relancer.

2. RESTRICTIONS DE PARCIMONIE DE L'ARTICLE (section II.B)
   Le retard y_{tau-2} est exclu de l'équation auxiliaire de pi, et le
   retard pi_{tau-1} est exclu de l'équation auxiliaire de y — mais les deux
   retards restent dans le vecteur d'instruments z_k, ce qui crée une
   restriction sur-identifiante testable, exactement comme dans l'article.

3. PANEL MENSUEL RECTANGULAIRE
   Le calcul HAC correct nécessite un index temporel commun aux deux
   indicateurs (cf. zeta_t(h) dans l'article, qui empile TOUS les
   indicateurs au même pas de temps mensuel). On restreint donc le panel
   aux mois où LES DEUX indicateurs ont un événement valide. Étant donné vos
   comptes (154 vs 155), ceci ne devrait faire perdre que quelques mois de
   bord d'échantillon — à vérifier via le diagnostic imprimé.
=============================================================================
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import chi2

INDICATORS = ["HICP", "PMI", "GDP"]
MATURITY_COL = {1: "OIS_1M", 3: "OIS_3M", 6: "OIS_6M", 12: "OIS_1Y"}
MATURITY_COLS = {
    1: "OIS_1M_n",
    2: "OIS_2M_n",
    3: "OIS_3M_n",
    4: "OIS_4M_n",
    5: "OIS_5M_n",
    6: "OIS_6M_n",
    7: "OIS_7M_n",
    8: "OIS_8M_n",
    9: "OIS_9M_n",
    10: "OIS_10M_n",
    11: "OIS_11M_n",
    12: "OIS_1Y_n",
}
HORIZONS_M_DAILY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # daily adaptation
OIS_LEVEL_COL = {1: "OIS_1M_n", 3: "OIS_3M_n", 6: "OIS_6M_n", 12: "OIS_1Y_n"}
HORIZONS_M = [1, 3, 6, 12]

HICP_SPILLOVER_CUTOFF = 5  # jour du mois au-delà duquel on ne suppose
# plus de débordement du flash HICP sur M+1
REF_MONTH_OFFSET = {"HICP": 0, "PMI": -1, "GDP": -1}  # cf. hypothèse 1 ci-dessus

PI_LAG = 1  # pi_{tau-1} : connu des deux types d'événement
Y_LAG = 2  # y_{tau-2}  : connu des deux types d'événement

NW_LAGS = 13  # Newey-West, comme dans HPB (2011), note 4

N_AUX_PARAMS_PER_INDICATOR = 10  # gamma_pi, xi_pi, zeta_pi, rho_pi, kappa_pi,
# gamma_y,  xi_y,  zeta_y, rho_y, kappa_y

# ARITHMÉTIQUE DE MOIS — robuste, évite les pièges de calendrier


def to_month_index(date_like):
    """Convertit une date en index entier (annee*12 + mois-1)."""
    d = pd.to_datetime(date_like)
    return d.dt.year * 12 + (d.dt.month - 1)


def infer_tau(release_dates, indicator):
    """Mois de référence (index entier) de chaque événement."""
    release_dates = pd.to_datetime(release_dates)
    idx = release_dates.dt.year * 12 + (release_dates.dt.month - 1)

    if indicator == "HICP":
        spillover = release_dates.dt.day <= HICP_SPILLOVER_CUTOFF
        idx = idx - spillover.astype(int)
    elif indicator == "PMI":
        idx = idx + REF_MONTH_OFFSET["PMI"]
    elif indicator == "GDP":
        # Pas d'heuristique de spillover ni d'offset : tau = mois calendaire
        # de PUBLICATION du survey (Advance/Preliminary/Final), pas le
        # trimestre cible. Décision documentée : chaque publication (A/P/F)
        # est traitée comme un événement de surprise indépendant, daté par
        # son propre mois de publication -> pas de collision de tau entre
        # A/P/F d'un même trimestre (sauf cas limite où deux publications
        # tombent le même mois calendaire, cf. drop_duplicates en aval).
        pass
    else:
        raise ValueError(f"Indicateur inconnu : {indicator}")
    return idx


# PRÉPARATION DES TABLES D'ÉVÉNEMENTS


def prepare_events(df_ois, indicator):
    """Met en forme une table de surprises (df_ois ou df_ois_pmi)."""
    df = df_ois.copy()
    df["event_time"] = df.index  # timestamp exact -> tri chrono
    df["release_date"] = pd.to_datetime(df["Date"])
    df["surprise"] = df["actual"] - df["consensus"]
    df["tau"] = infer_tau(df["release_date"], indicator)
    df["indicator"] = indicator
    return df.reset_index(drop=True)


# Daily events
def prepare_events_daily(df_surprise, indicator):
    """
    df_surprise : sortie de build_df_surprise(), filtrée sur un indicateur.
    """
    df = df_surprise[df_surprise["indicator"] == indicator].copy()
    df["event_time"] = pd.to_datetime(df["Date"])
    df["release_date"] = df["event_time"]
    df["surprise"] = df["actual"] - df["consensus"]
    df["tau"] = infer_tau(df["release_date"], indicator)

    # Variation OIS (dépend de la maturité, extrait au moment du build_monthly_panel)
    # Niveau pré-événement (t-1) disponible directement
    df["indicator"] = indicator
    return df.reset_index(drop=True)


# Merge pre anouncement level of OIS
def attach_ois_levels(events, ois_daily):
    ev = events.copy()
    ois = ois_daily.copy()

    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce", utc=True)
    ois["Date"] = pd.to_datetime(ois["Date"], errors="coerce", utc=True)

    ev = ev.dropna(subset=["event_time"]).copy()
    ois = ois.dropna(subset=["Date"]).copy()

    ev["event_day"] = ev["event_time"].dt.tz_convert("Europe/Brussels").dt.normalize()
    ois["event_day"] = ois["Date"].dt.tz_convert("Europe/Brussels").dt.normalize()

    ev = ev.sort_values("event_day")
    ois = ois.sort_values("event_day")

    merged = pd.merge_asof(
        ev, ois, on="event_day", direction="backward", allow_exact_matches=True
    )
    return merged


def attach_monthly_series(events, df_series, value_col, out_prefix, lags=(), leads=()):
    """Ajoute, par mois de référence (tau), les retards et les cibles
    futures d'une série mensuelle (df_inflation ou df_output)."""
    s = df_series.copy()
    s["m_idx"] = to_month_index(s["Date"])
    series_map = s.set_index("m_idx")[value_col]

    out = events.copy()
    for L in lags:
        key = out_prefix + f"_lag{L}"
        out[key] = out["tau"].apply(lambda t: series_map.get(t - L, np.nan))

    for H in leads:
        key = out_prefix + f"_target_m{H}"
        out[key] = out["tau"].apply(lambda t: series_map.get(t + H, np.nan))
    return out


def build_event_table(df_ois, df_ois_pmi, df_inflation, df_output, ois_daily):
    """Construit la table d'événements complète, avec contrôles et cibles."""
    ev_hicp = prepare_events(df_ois, "HICP")
    ev_pmi = prepare_events(df_ois_pmi, "PMI")
    events = pd.concat([ev_hicp, ev_pmi], ignore_index=True)
    events = events.sort_values("event_time").reset_index(drop=True)

    events = attach_monthly_series(
        events, df_inflation, "pi_realized", "pi", lags=[PI_LAG], leads=HORIZONS_M
    )
    events = attach_monthly_series(
        events, df_output, "y_realized", "y", lags=[Y_LAG], leads=HORIZONS_M
    )

    events = attach_ois_levels(events, ois_daily)
    return events


def build_event_table_daily(df_surprise, df_inflation, df_output):
    """
    df_surprise  : sortie de build_df_surprise() contenant HICP et PMI.
    df_inflation : DataFrame avec [Date, pi_realized].
    df_output    : DataFrame avec [Date, y_realized].
    """
    ev_list = [prepare_events_daily(df_surprise, ind) for ind in INDICATORS]
    events = pd.concat(ev_list, ignore_index=True)
    events = events.sort_values("event_time").reset_index(drop=True)

    events = attach_monthly_series(
        events, df_inflation, "pi_realized", "pi", lags=[PI_LAG], leads=HORIZONS_M_DAILY
    )
    events = attach_monthly_series(
        events, df_output, "y_realized", "y", lags=[Y_LAG], leads=HORIZONS_M_DAILY
    )
    return events


def inspect_reference_months(events, n=8):
    """À EXÉCUTER avant de faire confiance à l'alignement calendaire."""
    cols = ["indicator", "release_date", "tau"]
    print("--- Premiers événements (vérifier le mois de référence inféré) ---")
    print(events[cols].head(n).to_string(index=False))
    print("--- Derniers événements ---")
    print(events[cols].tail(n).to_string(index=False))
    print("\nNombre d'événements par indicateur :")
    print(events["indicator"].value_counts())


# PANEL MENSUEL RECTANGULAIRE (un indicateur HICP + un PMI par ligne)


def build_monthly_panel(events, m):
    """Panel mensuel unique combinant les deux indicateurs — nécessaire
    pour un calcul HAC correct en présence de cibles communes
    (pi_{tau+m}, y_{tau+m}) partagées par les deux blocs d'indicateurs.
    Restreint aux mois où les DEUX indicateurs ont un événement valide
    (cf. hypothèse 4 en tête de fichier)."""
    frames = []
    for indicator in INDICATORS:
        cols = [
            "tau",
            "actual",
            "consensus",
            "pi_lag1",
            "y_lag2",
            f"pi_target_m{m}",
            f"y_target_m{m}",
            MATURITY_COL[m],
            OIS_LEVEL_COL[m],
        ]  # niveaux]
        sub = events.loc[events["indicator"] == indicator, cols].copy()
        sub = sub.rename(
            columns={
                "actual": f"w_{indicator}",
                "consensus": f"wt_{indicator}",
                "pi_lag1": f"pilag_{indicator}",
                "y_lag2": f"ylag_{indicator}",
                f"pi_target_m{m}": f"pitgt_{indicator}",
                f"y_target_m{m}": f"ytgt_{indicator}",
                MATURITY_COL[m]: f"dois_{indicator}",
                # Niveau OIS
                OIS_LEVEL_COL[m]: f"oislvl_{indicator}",
            }
        )
        sub = sub.dropna().drop_duplicates(subset="tau", keep="first")
        frames.append(sub.set_index("tau"))

    import functools

    panel = functools.reduce(
        lambda left, right: left.join(right, how="inner"), frames
    ).sort_index()
    return panel


def build_monthly_panel_daily(events, m, verbose_coverage=True):
    """
    Identique à build_monthly_panel() mais lit les colonnes daily :
      - dois_{indicator}   <- OIS_{m}m_chg    (variation en niveau)
      - oislvl_{indicator} <- OIS_{m}m_level_prev  (niveau j-1, contrôle)

    JOIN : outer (et non inner) sur l'ensemble des indicateurs de
    INDICATORS. Décision délibérée : un indicateur peu fréquent (ex. GDP,
    ~3-4 publications/an contre ~12 pour HICP/PMI) ne doit pas réduire le
    panel aux seuls mois où TOUS les indicateurs coïncident -> cela
    sacrifierait l'essentiel des observations HICP/PMI. Conséquence
    assumée : le panel contient des NaN par bloc d'indicateur, à gérer
    explicitement en aval (étapes 1 et 2), jamais via un dropna() global.
    """
    chg_col = f"OIS_{m}m_chg"
    level_col = f"OIS_{m}m_level_prev"

    frames = []
    for indicator in INDICATORS:
        cols = [
            "tau",
            "actual",
            "consensus",
            "pi_lag1",
            "y_lag2",
            f"pi_target_m{m}",
            f"y_target_m{m}",
            chg_col,
            level_col,
        ]
        sub = events.loc[events["indicator"] == indicator, cols].copy()
        sub = sub.rename(
            columns={
                "actual": f"w_{indicator}",
                "consensus": f"wt_{indicator}",
                "pi_lag1": f"pilag_{indicator}",
                "y_lag2": f"ylag_{indicator}",
                f"pi_target_m{m}": f"pitgt_{indicator}",
                f"y_target_m{m}": f"ytgt_{indicator}",
                chg_col: f"dois_{indicator}",
                level_col: f"oislvl_{indicator}",
            }
        )
        sub = sub.dropna().drop_duplicates(subset="tau", keep="first")
        frames.append(sub.set_index("tau"))

    import functools

    panel = functools.reduce(
        lambda left, right: left.join(right, how="outer"), frames
    ).sort_index()

    if verbose_coverage:
        n_tot = len(panel)
        print(f"  [build_monthly_panel_daily, m={m}M] T total (outer) = {n_tot} mois")
        for indicator in INDICATORS:
            n_valid = panel[f"w_{indicator}"].notna().sum()
            print(
                f"    couverture {indicator:5s} : {n_valid}/{n_tot} mois "
                f"({100 * n_valid / max(n_tot, 1):.0f}%)"
            )

    if len(panel) < 40:
        import warnings

        warnings.warn(
            f"Maturité {m}M : seulement {len(panel)} obs. mensuelles — "
            f"GMM sous-identifié probable."
        )
    return panel


# PARAMÉTRISATION DU VECTEUR THETA


def theta_length(restrict_eta=True):
    n_eta = 1 if restrict_eta else len(INDICATORS)
    return N_AUX_PARAMS_PER_INDICATOR * len(INDICATORS) + 2 + n_eta


def unpack_theta(theta, restrict_eta=True):
    params, pos = {}, 0
    for k in INDICATORS:
        params[k] = {
            "gamma_pi": theta[pos + 0],
            "xi_pi": theta[pos + 1],
            "zeta_pi": theta[pos + 2],
            "rho_pi": theta[pos + 3],
            "kappa_pi": theta[pos + 4],
            "gamma_y": theta[pos + 5],
            "xi_y": theta[pos + 6],
            "zeta_y": theta[pos + 7],
            "rho_y": theta[pos + 8],
            "kappa_y": theta[pos + 9],
        }
        pos += N_AUX_PARAMS_PER_INDICATOR

    beta, delta = theta[pos], theta[pos + 1]
    pos += 2

    if restrict_eta:
        eta = {k: theta[pos] for k in INDICATORS}
    else:
        eta = {k: theta[pos + i] for i, k in enumerate(INDICATORS)}

    return params, beta, delta, eta


# 5. MOMENTS GMM — éq. (10)-(11)-(12)-(13) de l'article


def compute_G(theta, panel, restrict_eta=True):
    """
    Matrice T x (3 * 6 * 2) des moments g_t(theta) :
    - 3 résidus (pi, y, OIS)
    - 6 instruments (w, wt, pilag, ylag, ois_level, const)
    - 2 indicateurs
    """
    params, beta, delta, eta = unpack_theta(theta, restrict_eta)
    n = len(panel)
    blocks = []

    for indicator in INDICATORS:
        w = panel[f"w_{indicator}"].to_numpy()
        wt = panel[f"wt_{indicator}"].to_numpy()
        pilag = panel[f"pilag_{indicator}"].to_numpy()
        ylag = panel[f"ylag_{indicator}"].to_numpy()
        target_pi = panel[f"pitgt_{indicator}"].to_numpy()
        target_y = panel[f"ytgt_{indicator}"].to_numpy()
        d_ois = panel[f"dois_{indicator}"].to_numpy()

        # niveau OIS pré-événement correspondant à la maturité courante
        ois_level = panel[f"oislvl_{indicator}"].to_numpy()

        z = np.column_stack([w, wt, pilag, ylag, ois_level, np.ones(n)])

        p = params[indicator]

        pi_hat = (
            p["gamma_pi"] * w
            + p["xi_pi"] * wt
            + p["zeta_pi"] * pilag
            + p["rho_pi"] * ois_level
            + p["kappa_pi"]
        )

        y_hat = (
            p["gamma_y"] * w
            + p["xi_y"] * wt
            + p["zeta_y"] * ylag
            + p["rho_y"] * ois_level
            + p["kappa_y"]
        )

        resid_pi = target_pi - pi_hat
        resid_y = target_y - y_hat

        slope = beta * p["gamma_pi"] + delta * p["gamma_y"]
        ois_hat = eta[indicator] + slope * (w - wt)
        resid_ois = d_ois - ois_hat

        blocks.append(resid_pi[:, None] * z)
        blocks.append(resid_y[:, None] * z)
        blocks.append(resid_ois[:, None] * z)

    return np.column_stack(blocks)


def gmm_objective(theta, panel, W, restrict_eta=True):
    G = compute_G(theta, panel, restrict_eta)
    g_bar = G.mean(axis=0)
    return float(g_bar @ W @ g_bar)


def newey_west_S(G, n_lags=NW_LAGS):
    """Estimateur HAC de Newey-West (1987) de la variance asymptotique de
    g_bar, comme dans HPB (2011), note 4."""
    T = G.shape[0]
    Gc = G - G.mean(axis=0)
    S = Gc.T @ Gc / T
    n_lags = min(n_lags, T - 2)
    for L in range(1, n_lags + 1):
        w = 1 - L / (n_lags + 1)
        Gamma = Gc[L:].T @ Gc[:-L] / T
        S += w * (Gamma + Gamma.T)
    return S


def numerical_jacobian(f, x, eps=1e-5):
    f0 = f(x)
    n, m = len(x), len(f0)
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return J


# 6. VALEURS DE DÉPART — procédure en deux étapes décrite par HPB (II.A)


def warm_start_core(panel):
    core = []
    fingerprints = {}

    for indicator in INDICATORS:
        X_pi = sm.add_constant(
            panel[
                [
                    f"w_{indicator}",
                    f"wt_{indicator}",
                    f"pilag_{indicator}",
                    f"oislvl_{indicator}",
                ]
            ]
        )
        ols_pi = sm.OLS(panel[f"pitgt_{indicator}"], X_pi).fit()

        X_y = sm.add_constant(
            panel[
                [
                    f"w_{indicator}",
                    f"wt_{indicator}",
                    f"ylag_{indicator}",
                    f"oislvl_{indicator}",
                ]
            ]
        )
        ols_y = sm.OLS(panel[f"ytgt_{indicator}"], X_y).fit()

        gamma_pi = ols_pi.params[f"w_{indicator}"]
        xi_pi = ols_pi.params[f"wt_{indicator}"]
        zeta_pi = ols_pi.params[f"pilag_{indicator}"]
        rho_pi = ols_pi.params[f"oislvl_{indicator}"]
        kappa_pi = ols_pi.params["const"]

        gamma_y = ols_y.params[f"w_{indicator}"]
        xi_y = ols_y.params[f"wt_{indicator}"]
        zeta_y = ols_y.params[f"ylag_{indicator}"]
        rho_y = ols_y.params[f"oislvl_{indicator}"]
        kappa_y = ols_y.params["const"]

        core.extend(
            [
                gamma_pi,
                xi_pi,
                zeta_pi,
                rho_pi,
                kappa_pi,
                gamma_y,
                xi_y,
                zeta_y,
                rho_y,
                kappa_y,
            ]
        )

        fingerprints[indicator] = (gamma_pi, gamma_y)

    naive_slopes = []
    for indicator in INDICATORS:
        s = panel[f"w_{indicator}"] - panel[f"wt_{indicator}"]
        ols_slope = sm.OLS(panel[f"dois_{indicator}"], sm.add_constant(s)).fit()
        naive_slopes.append(ols_slope.params.iloc[1])

    A = np.array([fingerprints[k] for k in INDICATORS])
    beta0, delta0 = np.linalg.lstsq(A, np.array(naive_slopes), rcond=None)[0]
    core.extend([beta0, delta0])

    return np.array(core), A


# 7. ESTIMATION GMM EN DEUX ÉTAPES (éq. 14-15 de l'article)


def _two_step_from(theta0, panel, restrict_eta, n_moments):
    """Un essai complet d'estimation GMM en deux étapes à partir d'un point
    de départ donné. Retourne (theta_hat, objectif_final, succes_etape2)."""
    W0 = np.eye(n_moments)
    res1 = minimize(
        gmm_objective,
        theta0,
        args=(panel, W0, restrict_eta),
        method="BFGS",
        options={"maxiter": 3000, "gtol": 1e-7},
    )

    G1 = compute_G(res1.x, panel, restrict_eta)
    S1 = newey_west_S(G1)
    W1 = np.linalg.pinv(S1)

    res2 = minimize(
        gmm_objective,
        res1.x,
        args=(panel, W1, restrict_eta),
        method="BFGS",
        options={"maxiter": 3000, "gtol": 1e-7},
    )
    return res2.x, res2.fun, res2.success


def estimate_gmm(panel, restrict_eta=True, n_restarts=5, perturb_scale=0.3, seed=0):
    """Estimation GMM en deux étapes, avec redémarrages multiples pour
    limiter le risque d'optimum local — particulièrement pertinent ici car
    le terme bilinéaire (beta*gamma_pi + delta*gamma_y) peut admettre
    plusieurs combinaisons (beta, delta) proches en valeur d'objectif si les
    empreintes des deux indicateurs sont peu différenciées (cf. discussion
    sur la rotation d'identification : HICP et PMI doivent avoir des ratios
    gamma_y/gamma_pi suffisamment distincts)."""
    n_theta = theta_length(restrict_eta)
    core0, fingerprint_matrix = warm_start_core(panel)
    n_eta = 1 if restrict_eta else len(INDICATORS)
    theta0 = np.concatenate([core0, np.zeros(n_eta)])
    assert len(theta0) == n_theta, (len(theta0), n_theta)

    n_moments = 3 * 6 * len(INDICATORS)  # = 30

    rng = np.random.default_rng(seed)
    candidates = [theta0]
    for _ in range(n_restarts):
        noise = rng.normal(0, perturb_scale, size=theta0.shape)
        candidates.append(theta0 + noise * np.abs(theta0).clip(min=0.1))

    best_obj, best_theta, best_ok = np.inf, None, False
    objectives_seen = []
    for cand in candidates:
        theta_try, obj_try, ok_try = _two_step_from(
            cand, panel, restrict_eta, n_moments
        )
        objectives_seen.append(obj_try)
        if obj_try < best_obj:
            best_obj, best_theta, best_ok = obj_try, theta_try, ok_try

    theta_hat = best_theta
    objectives_seen = np.array(objectives_seen)
    spread = objectives_seen.max() - objectives_seen.min()
    if spread > 1e-3 * max(abs(best_obj), 1.0):
        import warnings

        warnings.warn(
            f"Les redémarrages convergent vers des objectifs différents "
            f"(écart={spread:.4g}) -> signe possible d'optima locaux ou "
            f"d'identification faible de (beta, delta). Inspectez "
            f"fit['restart_objectives'] avant d'interpréter les résultats."
        )
    G2 = compute_G(theta_hat, panel, restrict_eta)
    S2 = newey_west_S(G2)
    W2 = np.linalg.pinv(S2)

    D = numerical_jacobian(
        lambda th: compute_G(th, panel, restrict_eta).mean(axis=0), theta_hat
    )

    T = len(panel)
    V = np.linalg.pinv(D.T @ W2 @ D) / T
    se = np.sqrt(np.clip(np.diag(V), 0, None))

    J_stat = T * gmm_objective(theta_hat, panel, W2, restrict_eta)
    df = n_moments - n_theta
    p_value = 1 - chi2.cdf(J_stat, df) if df > 0 else np.nan

    return {
        "theta": theta_hat,
        "se": se,
        "V": V,
        "J_stat": J_stat,
        "df": df,
        "p_value": p_value,
        "T": T,
        "S": S2,
        "fingerprints": fingerprint_matrix,
        "converged": bool(best_ok),
        "restart_objectives": objectives_seen,
    }


# RAPPORT PAR MATURITÉ


def report_maturity(events, m, verbose=True):
    panel = build_monthly_panel(events, m)
    if verbose:
        print(f"\n=== Maturité {m} mois — T = {len(panel)} observations mensuelles ===")
        if len(panel) < 40:
            print(
                "  /!\\ échantillon très réduit pour un GMM à 19+ paramètres "
                "— interprétez les résultats avec une prudence accrue."
            )

    fit_r = estimate_gmm(panel, restrict_eta=True)
    fit_u = estimate_gmm(panel, restrict_eta=False)

    pos_beta = N_AUX_PARAMS_PER_INDICATOR * len(INDICATORS)
    beta_hat, delta_hat = fit_r["theta"][pos_beta], fit_r["theta"][pos_beta + 1]
    se_beta, se_delta = fit_r["se"][pos_beta], fit_r["se"][pos_beta + 1]

    if verbose:
        print(
            f"  Convergence (meilleur des redémarrages) : "
            f"{fit_r['converged']} | objectifs des redémarrages : "
            f"{np.round(fit_r['restart_objectives'], 4)}"
        )
        print(" Empreintes (gamma_pi, gamma_y) par indicateur :")
        for i, k in enumerate(INDICATORS):
            print(f"    {k:5s} : {fit_r['fingerprints'][i]}")
        print(
            f"  beta_hat  = {beta_hat:8.3f}  (se = {se_beta:.3f}, "
            f"t = {beta_hat / se_beta if se_beta > 0 else np.nan:.2f})"
        )
        print(
            f"  delta_hat = {delta_hat:8.3f}  (se = {se_delta:.3f}, "
            f"t = {delta_hat / se_delta if se_delta > 0 else np.nan:.2f})"
        )
        print(
            f"  J-test (modèle restreint) : J = {fit_r['J_stat']:.2f}, "
            f"df = {fit_r['df']}, p = {fit_r['p_value']:.3f}"
        )

        dJ = fit_r["J_stat"] - fit_u["J_stat"]
        df_cross = len(INDICATORS) - 1  # n_eta unrestricted - n_eta restricted(=1)
        p_cross = 1 - chi2.cdf(max(dJ, 0), df=df_cross)
        print(
            f"  Test restriction croisée (eta commun à {INDICATORS}) : "
            f"dJ = {dJ:.2f}, df = {df_cross}, p = {p_cross:.3f}"
        )

    return fit_r, fit_u, panel


def run_all_maturities(df_ois, df_ois_pmi, df_inflation, df_output, ois_daily):
    events = build_event_table(df_ois, df_ois_pmi, df_inflation, df_output, ois_daily)
    inspect_reference_months(events)

    results = {}
    for m in HORIZONS_M:
        results[m] = report_maturity(events, m)
    return results, events


# Daily adaptation


# Builds surprises with ois_daily
def build_df_surprise(df_announcements, ois_daily):
    ois = ois_daily.copy()
    ois["Date"] = pd.to_datetime(ois["Date"])
    ois = ois.sort_values("Date").reset_index(drop=True)

    # Variations en niveau pour toutes les maturités
    for m, col in MATURITY_COLS.items():
        ois[f"{col}_prev"] = ois[col].shift(1)
        ois[f"{col}_chg"] = (ois[col] - ois[f"{col}_prev"]) * 100  # bp
    ann = df_announcements.copy()
    ann["Date"] = (
        pd.to_datetime(ann["Date"], utc=True)
        .dt.tz_convert("Europe/Brussels")
        .dt.normalize()
        .dt.tz_localize(None)
    )

    ois["Date"] = ois["Date"].dt.normalize()
    df_s = ann.merge(ois, on="Date", how="left")

    rename = {}
    for m, col in MATURITY_COLS.items():
        rename[col] = f"OIS_{m}m_level"
        rename[f"{col}_prev"] = f"OIS_{m}m_level_prev"
        rename[f"{col}_chg"] = f"OIS_{m}m_chg"
    df_s = df_s.rename(columns=rename)

    n_missing = (
        df_s[[f"OIS_{m}m_chg" for m in HORIZONS_M_DAILY]].isna().any(axis=1).sum()
    )
    if n_missing > 0:
        import warnings

        warnings.warn(
            f"{n_missing} annonce(s) sans correspondance dans ois_daily "
            f"(jours fériés ou données manquantes). Ces lignes auront NaN "
            f"et seront exclues de l'estimation."
        )
    return df_s


def run_all_maturities_daily(df_surprise, df_inflation, df_output):
    """
    df_surprise  : sortie de build_df_surprise()
    df_inflation : [Date, pi_realized]
    df_output    : [Date, y_realized]
    """
    events = build_event_table_daily(df_surprise, df_inflation, df_output)
    inspect_reference_months(events)

    results = {}
    for m in HORIZONS_M_DAILY:
        panel = build_monthly_panel_daily(events, m)
        print(f"\n=== Maturité {m}M — T = {len(panel)} obs. ===")
        fit_r, fit_u, _ = report_maturity_from_panel(panel, m)
        results[m] = (fit_r, fit_u, panel)
    return results, events


def report_maturity_from_panel(panel, m, verbose=True):
    """Même logique que report_maturity() mais prend le panel déjà construit."""
    fit_r = estimate_gmm(panel, restrict_eta=True)
    fit_u = estimate_gmm(panel, restrict_eta=False)

    pos_beta = N_AUX_PARAMS_PER_INDICATOR * len(INDICATORS)
    beta_hat, delta_hat = fit_r["theta"][pos_beta], fit_r["theta"][pos_beta + 1]
    se_beta, se_delta = fit_r["se"][pos_beta], fit_r["se"][pos_beta + 1]

    if verbose:
        print(
            f"  Convergence : {fit_r['converged']} | "
            f"objectifs : {np.round(fit_r['restart_objectives'], 4)}"
        )
        for i, k in enumerate(INDICATORS):
            print(
                f"  Empreinte {k:5s} (gamma_pi, gamma_y) = {fit_r['fingerprints'][i]}"
            )
        t_b = beta_hat / se_beta if se_beta > 0 else np.nan
        t_d = delta_hat / se_delta if se_delta > 0 else np.nan
        print(f"  beta_hat  = {beta_hat:8.3f}  (se={se_beta:.3f}, t={t_b:.2f})")
        print(f"  delta_hat = {delta_hat:8.3f}  (se={se_delta:.3f}, t={t_d:.2f})")
        print(
            f"  J-test : J={fit_r['J_stat']:.2f}, "
            f"df={fit_r['df']}, p={fit_r['p_value']:.3f}"
        )
        dJ = fit_r["J_stat"] - fit_u["J_stat"]
        df_cross = len(INDICATORS) - 1  # n_eta unrestricted - n_eta restricted(=1)
        p_cross = 1 - chi2.cdf(max(dJ, 0), df=df_cross)
        print(
            f"  Test restriction croisée : dJ={dJ:.2f}, df={df_cross}, p={p_cross:.3f}"
        )

    return fit_r, fit_u, panel
