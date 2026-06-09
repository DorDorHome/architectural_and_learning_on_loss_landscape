"""
Regression Diagnostics — Comprehensive diagnostic tests and statistics.

Computes:
  - Goodness-of-fit: R², Adj-R², SSR, SSE, SST, AIC, BIC, F-stat, log-likelihood
  - Coefficient diagnostics: SE, t-stats, p-values, 95% CI, standardized coefficients
  - Residual diagnostics: Durbin-Watson, Jarque-Bera, Breusch-Pagan, Goldfeld-Quandt
  - Multicollinearity: Condition number, VIF
  - Influence: Cook's distance, leverage (hat values), DFFITS
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .regression_analysis import RegressionResult
from .data_processing import unsanitize_name


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_diagnostics(
    result: RegressionResult,
    diagnostics_config: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Compute all diagnostics enabled in the YAML config.

    Parameters
    ----------
    result : RegressionResult
        Fitted regression result.
    diagnostics_config : dict
        Boolean flags for each diagnostic (from YAML ``diagnostics`` section).

    Returns
    -------
    dict
        Nested dict with keys: "goodness_of_fit", "coefficients",
        "residual_tests", "multicollinearity", "influence".
    """
    diag: Dict[str, Any] = {}

    diag["goodness_of_fit"] = goodness_of_fit(result, diagnostics_config)
    diag["coefficients"] = coefficient_diagnostics(result, diagnostics_config)
    diag["residual_tests"] = residual_diagnostics(result, diagnostics_config)
    diag["multicollinearity"] = multicollinearity_diagnostics(result, diagnostics_config)
    diag["influence"] = influence_diagnostics(result, diagnostics_config)

    return diag


# ---------------------------------------------------------------------------
# Goodness-of-fit
# ---------------------------------------------------------------------------

def goodness_of_fit(
    result: RegressionResult,
    config: Dict[str, bool],
) -> Dict[str, Any]:
    """Compute goodness-of-fit statistics."""
    gof: Dict[str, Any] = {}
    y = result.y.values if hasattr(result.y, 'values') else np.array(result.y)
    y_hat = result.fitted_values
    y_bar = np.mean(y)
    resid = result.residuals

    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y_bar) ** 2))
    ssr = sst - sse  # SSR = SST - SSE

    n = result.n_obs
    k = result.n_predictors

    if config.get("r_squared", True):
        gof["R_squared"] = 1 - sse / sst if sst > 0 else np.nan

    if config.get("adjusted_r_squared", True):
        if sst > 0 and n > k + 1:
            gof["Adj_R_squared"] = 1 - (sse / (n - k - 1)) / (sst / (n - 1))
        else:
            gof["Adj_R_squared"] = np.nan

    if config.get("ssr", True):
        gof["SSR"] = ssr

    if config.get("sse", True):
        gof["SSE"] = sse

    if config.get("sst", True):
        gof["SST"] = sst

    if config.get("f_statistic", True):
        if result.backend == "statsmodels":
            gof["F_statistic"] = result.model.fvalue
            gof["F_p_value"] = result.model.f_pvalue
        else:
            # Manual F-stat for sklearn
            if k > 0 and (n - k - 1) > 0 and sse > 0:
                f_stat = (ssr / k) / (sse / (n - k - 1))
                gof["F_statistic"] = f_stat
                from scipy.stats import f as f_dist
                gof["F_p_value"] = 1 - f_dist.cdf(f_stat, k, n - k - 1)
            else:
                gof["F_statistic"] = np.nan
                gof["F_p_value"] = np.nan

    if config.get("aic", True) and result.backend == "statsmodels":
        gof["AIC"] = result.model.aic

    if config.get("bic", True) and result.backend == "statsmodels":
        gof["BIC"] = result.model.bic

    if config.get("log_likelihood", True) and result.backend == "statsmodels":
        gof["Log_Likelihood"] = result.model.llf

    return gof


# ---------------------------------------------------------------------------
# Coefficient diagnostics
# ---------------------------------------------------------------------------

def coefficient_diagnostics(
    result: RegressionResult,
    config: Dict[str, bool],
) -> pd.DataFrame:
    """
    Build a coefficient summary table.

    Columns: coefficient, std_error, t_stat, p_value, ci_lower, ci_upper,
    standardized_coeff (optional).
    """
    if result.backend == "statsmodels":
        return _coeff_diag_statsmodels(result, config)
    else:
        return _coeff_diag_sklearn(result, config)


def _coeff_diag_statsmodels(result: RegressionResult, config: Dict[str, bool]) -> pd.DataFrame:
    """Coefficient diagnostics from a statsmodels result."""
    model = result.model
    params = model.params
    bse = model.bse
    tvalues = model.tvalues
    pvalues = model.pvalues

    rows = []
    for name in params.index:
        row = {"variable": unsanitize_name(name)}
        row["coefficient"] = params[name]

        if config.get("p_values", True):
            row["std_error"] = bse.get(name, np.nan)
            row["t_statistic"] = tvalues.get(name, np.nan)
            row["p_value"] = pvalues.get(name, np.nan)

        if config.get("confidence_intervals", True):
            try:
                ci = model.conf_int(alpha=0.05)
                row["ci_lower"] = ci.loc[name, 0]
                row["ci_upper"] = ci.loc[name, 1]
            except Exception:
                row["ci_lower"] = np.nan
                row["ci_upper"] = np.nan

        if config.get("standardized_coefficients", True) and name != "const":
            try:
                x_std = result.X[name].std()
                y_std = result.y.std()
                if y_std > 0 and x_std > 0:
                    row["standardized_coeff"] = params[name] * x_std / y_std
                else:
                    row["standardized_coeff"] = np.nan
            except Exception:
                row["standardized_coeff"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _coeff_diag_sklearn(result: RegressionResult, config: Dict[str, bool]) -> pd.DataFrame:
    """Coefficient diagnostics from a sklearn result (limited)."""
    rows = []
    for name, coef in result.coefficients.items():
        row = {"variable": unsanitize_name(name), "coefficient": coef}
        # sklearn doesn't provide SE/t/p natively
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def residual_diagnostics(
    result: RegressionResult,
    config: Dict[str, bool],
) -> Dict[str, Any]:
    """Compute residual diagnostic tests."""
    diag: Dict[str, Any] = {}
    resid = result.residuals

    if config.get("durbin_watson", True):
        try:
            from statsmodels.stats.stattools import durbin_watson
            diag["Durbin_Watson"] = float(durbin_watson(resid))
        except Exception as e:
            warnings.warn(f"Durbin-Watson failed: {e}")

    if config.get("jarque_bera", True):
        try:
            from statsmodels.stats.stattools import jarque_bera
            jb_stat, jb_pval, skew, kurtosis = jarque_bera(resid)
            diag["Jarque_Bera_stat"] = float(jb_stat)
            diag["Jarque_Bera_p_value"] = float(jb_pval)
            diag["Skewness"] = float(skew)
            diag["Kurtosis"] = float(kurtosis)
        except Exception as e:
            warnings.warn(f"Jarque-Bera failed: {e}")

    if config.get("breusch_pagan", True) and result.backend == "statsmodels":
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_pval, bp_fstat, bp_fpval = het_breuschpagan(
                result.model.resid, result.model.model.exog
            )
            diag["Breusch_Pagan_stat"] = float(bp_stat)
            diag["Breusch_Pagan_p_value"] = float(bp_pval)
            diag["Breusch_Pagan_F_stat"] = float(bp_fstat)
            diag["Breusch_Pagan_F_p_value"] = float(bp_fpval)
        except Exception as e:
            warnings.warn(f"Breusch-Pagan failed: {e}")

    if config.get("goldfeld_quandt", True) and result.backend == "statsmodels":
        try:
            from statsmodels.stats.diagnostic import het_goldfeldquandt
            gq_stat, gq_pval, gq_ordering = het_goldfeldquandt(
                result.model.model.endog, result.model.model.exog
            )
            diag["Goldfeld_Quandt_stat"] = float(gq_stat)
            diag["Goldfeld_Quandt_p_value"] = float(gq_pval)
        except Exception as e:
            warnings.warn(f"Goldfeld-Quandt failed: {e}")

    return diag


# ---------------------------------------------------------------------------
# Multicollinearity diagnostics
# ---------------------------------------------------------------------------

def multicollinearity_diagnostics(
    result: RegressionResult,
    config: Dict[str, bool],
) -> Dict[str, Any]:
    """Compute multicollinearity diagnostics."""
    diag: Dict[str, Any] = {}

    if config.get("condition_number", True):
        if result.backend == "statsmodels":
            try:
                diag["Condition_Number"] = float(result.model.condition_number)
            except Exception:
                # Manual computation
                X_vals = result.X.values.astype(float)
                sv = np.linalg.svd(X_vals, compute_uv=False)
                if sv[-1] > 0:
                    diag["Condition_Number"] = float(sv[0] / sv[-1])
                else:
                    diag["Condition_Number"] = np.inf
        else:
            X_vals = result.X.values.astype(float)
            sv = np.linalg.svd(X_vals, compute_uv=False)
            if sv[-1] > 0:
                diag["Condition_Number"] = float(sv[0] / sv[-1])
            else:
                diag["Condition_Number"] = np.inf

    if config.get("vif", True):
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # VIF should be computed on the design matrix WITHOUT the constant
            X_no_const = result.X.drop(columns=["const"], errors="ignore")
            if X_no_const.shape[1] >= 2:
                vif_data = {}
                X_arr = X_no_const.values.astype(float)
                for i, col in enumerate(X_no_const.columns):
                    try:
                        vif_val = variance_inflation_factor(X_arr, i)
                        vif_data[unsanitize_name(col)] = float(vif_val)
                    except Exception:
                        vif_data[unsanitize_name(col)] = np.nan
                diag["VIF"] = vif_data
            else:
                diag["VIF"] = {"note": "VIF requires >= 2 predictors"}
        except Exception as e:
            warnings.warn(f"VIF computation failed: {e}")

    return diag


# ---------------------------------------------------------------------------
# Influence diagnostics
# ---------------------------------------------------------------------------

def influence_diagnostics(
    result: RegressionResult,
    config: Dict[str, bool],
) -> Dict[str, Any]:
    """Compute influence diagnostics (Cook's D, leverage, DFFITS)."""
    diag: Dict[str, Any] = {}

    if result.backend != "statsmodels":
        if any(config.get(k, False) for k in ("cooks_distance", "leverage", "dffits")):
            diag["note"] = "Influence diagnostics require statsmodels backend (OLS/WLS/Robust)"
        return diag

    try:
        from statsmodels.stats.outliers_influence import OLSInfluence
        influence = OLSInfluence(result.model)
    except Exception as e:
        warnings.warn(f"Could not compute influence measures: {e}")
        return diag

    if config.get("cooks_distance", True):
        try:
            cooks_d, cooks_pval = influence.cooks_distance
            diag["cooks_distance"] = cooks_d
            diag["cooks_distance_p_values"] = cooks_pval
            # Summary statistics
            n = result.n_obs
            threshold = 4.0 / n
            n_influential = int(np.sum(cooks_d > threshold))
            diag["cooks_distance_threshold"] = threshold
            diag["n_influential_cooks"] = n_influential
        except Exception as e:
            warnings.warn(f"Cook's distance failed: {e}")

    if config.get("leverage", True):
        try:
            hat_values = influence.hat_matrix_diag
            diag["leverage"] = hat_values
            # Threshold: 2(k+1)/n
            k = result.n_predictors
            n = result.n_obs
            threshold = 2.0 * (k + 1) / n
            diag["leverage_threshold"] = threshold
            diag["n_high_leverage"] = int(np.sum(hat_values > threshold))
        except Exception as e:
            warnings.warn(f"Leverage computation failed: {e}")

    if config.get("dffits", True):
        try:
            dffits_vals = influence.dffits[0]
            diag["dffits"] = dffits_vals
            # Threshold: 2*sqrt(k/n)
            k = result.n_predictors
            n = result.n_obs
            threshold = 2.0 * np.sqrt(k / n) if n > 0 else np.inf
            diag["dffits_threshold"] = threshold
            diag["n_influential_dffits"] = int(np.sum(np.abs(dffits_vals) > threshold))
        except Exception as e:
            warnings.warn(f"DFFITS computation failed: {e}")

    return diag


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_diagnostics_report(
    diagnostics: Dict[str, Any],
    group_name: Optional[str] = None,
) -> str:
    """
    Format diagnostics into a human-readable report string.

    Parameters
    ----------
    diagnostics : dict
        Output from ``compute_diagnostics``.
    group_name : str, optional
        Group label for stratified regression.

    Returns
    -------
    str
    """
    lines = []
    header = "REGRESSION DIAGNOSTICS"
    if group_name:
        header += f"  [Group: {group_name}]"
    lines.append(f"\n{'=' * 70}")
    lines.append(header)
    lines.append('=' * 70)

    # --- Goodness-of-fit ---
    gof = diagnostics.get("goodness_of_fit", {})
    if gof:
        lines.append("\n--- Goodness of Fit ---")
        for key, val in gof.items():
            if isinstance(val, float):
                lines.append(f"  {key:30s} = {val:.6f}")
            else:
                lines.append(f"  {key:30s} = {val}")

    # --- Coefficients ---
    coeff_df = diagnostics.get("coefficients")
    if coeff_df is not None and isinstance(coeff_df, pd.DataFrame) and not coeff_df.empty:
        lines.append("\n--- Coefficient Diagnostics ---")
        # Format as aligned table
        lines.append(coeff_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # --- Residual tests ---
    resid = diagnostics.get("residual_tests", {})
    if resid:
        lines.append("\n--- Residual Diagnostics ---")
        for key, val in resid.items():
            if isinstance(val, float):
                lines.append(f"  {key:35s} = {val:.6f}")
            else:
                lines.append(f"  {key:35s} = {val}")

    # --- Multicollinearity ---
    multi = diagnostics.get("multicollinearity", {})
    if multi:
        lines.append("\n--- Multicollinearity ---")
        for key, val in multi.items():
            if key == "VIF" and isinstance(val, dict):
                lines.append(f"  VIF:")
                for var, vif_val in val.items():
                    flag = ""
                    if isinstance(vif_val, (int, float)):
                        if vif_val > 10:
                            flag = " ** SEVERE"
                        elif vif_val > 5:
                            flag = " * moderate"
                    lines.append(f"    {var:30s} = {vif_val:.4f}{flag}" if isinstance(vif_val, float) else f"    {var:30s} = {vif_val}")
            elif isinstance(val, float):
                flag = ""
                if key == "Condition_Number" and val > 30:
                    flag = " ** HIGH"
                lines.append(f"  {key:30s} = {val:.4f}{flag}")
            else:
                lines.append(f"  {key:30s} = {val}")

    # --- Influence ---
    inf = diagnostics.get("influence", {})
    if inf:
        summary_keys = [
            "cooks_distance_threshold", "n_influential_cooks",
            "leverage_threshold", "n_high_leverage",
            "dffits_threshold", "n_influential_dffits",
            "note",
        ]
        influence_summary = {k: v for k, v in inf.items() if k in summary_keys}
        if influence_summary:
            lines.append("\n--- Influence Diagnostics ---")
            for key, val in influence_summary.items():
                if isinstance(val, float):
                    lines.append(f"  {key:35s} = {val:.6f}")
                else:
                    lines.append(f"  {key:35s} = {val}")

    lines.append('=' * 70)
    return "\n".join(lines)
