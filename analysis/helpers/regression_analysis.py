"""
Regression Analysis — Fit regression models with a uniform result interface.

Supports:
  - OLS (statsmodels)
  - WLS (statsmodels)
  - Robust / RLM (statsmodels)
  - Ridge (sklearn, wrapped)
  - Lasso (sklearn, wrapped)
  - Grouped / stratified regression (same model per group)
  - Group comparison tables
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data_processing import (
    encode_categoricals,
    apply_transforms,
    get_design_matrix,
    sanitize_key,
    unsanitize_name,
)


# ---------------------------------------------------------------------------
# Unified result container
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    """
    Uniform container for regression results, regardless of backend.

    Attributes
    ----------
    model : object
        The fitted model object (statsmodels Results or sklearn estimator).
    backend : str
        "statsmodels" or "sklearn".
    regression_type : str
        E.g. "ols", "wls", "ridge", "lasso", "robust".
    y : pd.Series
        Response vector.
    X : pd.DataFrame
        Design matrix (with constant if applicable).
    fitted_values : np.ndarray
        Predicted values (ŷ).
    residuals : np.ndarray
        y - ŷ.
    coefficients : pd.Series
        Named coefficient values.
    n_obs : int
        Number of observations.
    n_predictors : int
        Number of predictors (excluding constant).
    group_name : Optional[str]
        Name of the group if this is a stratified regression.
    """
    model: Any
    backend: str
    regression_type: str
    y: pd.Series
    X: pd.DataFrame
    fitted_values: np.ndarray
    residuals: np.ndarray
    coefficients: pd.Series
    n_obs: int
    n_predictors: int
    group_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def fit_regression(
    y: pd.Series,
    X: pd.DataFrame,
    regression_type: str = "ols",
    alpha: float = 1.0,
    group_name: Optional[str] = None,
    **kwargs,
) -> RegressionResult:
    """
    Fit a regression model and return a uniform RegressionResult.

    Parameters
    ----------
    y : pd.Series
        Response variable.
    X : pd.DataFrame
        Design matrix (should include constant for intercept models).
    regression_type : str
        One of: "ols", "wls", "ridge", "lasso", "robust".
    alpha : float
        Regularization strength for ridge/lasso.
    group_name : str, optional
        Label for grouped regression.

    Returns
    -------
    RegressionResult
    """
    # Validate inputs
    if len(y) != len(X):
        raise ValueError(f"y length ({len(y)}) != X length ({len(X)})")
    if len(y) == 0:
        raise ValueError("Cannot fit regression with 0 observations")

    n_predictors = X.shape[1] - (1 if "const" in X.columns else 0)
    if len(y) <= n_predictors:
        raise ValueError(
            f"Insufficient observations ({len(y)}) for {n_predictors} predictors. "
            f"Need at least {n_predictors + 1}."
        )

    dispatch = {
        "ols": _fit_ols,
        "wls": _fit_wls,
        "robust": _fit_robust,
        "ridge": _fit_ridge,
        "lasso": _fit_lasso,
    }

    if regression_type not in dispatch:
        raise ValueError(
            f"Unknown regression_type: {regression_type!r}. "
            f"Choose from: {list(dispatch.keys())}"
        )

    result = dispatch[regression_type](y, X, alpha=alpha, **kwargs)
    result.group_name = group_name
    return result


# ---------------------------------------------------------------------------
# Grouped / stratified regression
# ---------------------------------------------------------------------------

def run_grouped_regression(
    df: pd.DataFrame,
    regression_config: Dict[str, Any],
) -> Dict[str, RegressionResult]:
    """
    Split DataFrame by group_by column(s) and fit the same model in each group.

    Parameters
    ----------
    df : pd.DataFrame
        Full analysis DataFrame (already cleaned and sanitized).
    regression_config : dict
        The ``analysis.regression`` section of the YAML config.

    Returns
    -------
    dict mapping group_label -> RegressionResult
    """
    group_by = regression_config.get("group_by")
    dep_var = regression_config["dependent_variable"]
    indep_vars = regression_config["independent_variables"]
    reg_type = regression_config.get("regression_type", "ols")
    alpha = regression_config.get("alpha", 1.0)
    transforms = regression_config.get("transforms", [])

    if group_by is None:
        # No grouping — single pooled regression
        df_t = apply_transforms(df.copy(), transforms)
        y, X = get_design_matrix(df_t, dep_var, indep_vars)
        result = fit_regression(y, X, reg_type, alpha=alpha, group_name="all")
        return {"all": result}

    # Ensure group_by is a list
    if isinstance(group_by, str):
        group_by = [group_by]

    # Resolve column names
    group_cols = [sanitize_key(g) if sanitize_key(g) in df.columns else g for g in group_by]
    for col in group_cols:
        if col not in df.columns:
            raise KeyError(
                f"group_by column '{col}' not found in DataFrame. "
                f"Available: {list(df.columns)}"
            )

    results = {}
    if len(group_cols) == 1:
        groups = df.groupby(group_cols[0])
    else:
        groups = df.groupby(group_cols)

    for group_key, group_df in groups:
        label = str(group_key)
        n = len(group_df)

        # Check minimum size
        n_pred = len(indep_vars)
        if n <= n_pred + 1:
            warnings.warn(
                f"Group '{label}' has only {n} observations for {n_pred} predictors; "
                f"skipping (need >= {n_pred + 2})."
            )
            continue

        print(f"\n  --- Group: {label} (n={n}) ---")
        try:
            group_df_t = apply_transforms(group_df.copy(), transforms)
            y, X = get_design_matrix(group_df_t, dep_var, indep_vars)
            result = fit_regression(y, X, reg_type, alpha=alpha, group_name=label)
            results[label] = result
        except Exception as e:
            warnings.warn(f"Regression failed for group '{label}': {e}")
            continue

    if not results:
        raise RuntimeError("No groups had sufficient data for regression.")

    return results


def compare_group_results(
    results: Dict[str, RegressionResult],
) -> pd.DataFrame:
    """
    Build a side-by-side comparison table of key statistics across groups.

    Returns
    -------
    pd.DataFrame
        One row per group with columns for R², F-stat, n, and each coefficient
        with its p-value.
    """
    rows = []
    for label, res in results.items():
        row = {"group": label, "n": res.n_obs}

        if res.backend == "statsmodels":
            model = res.model
            row["R_squared"] = model.rsquared
            row["Adj_R_squared"] = model.rsquared_adj
            row["F_statistic"] = model.fvalue
            row["F_p_value"] = model.f_pvalue

            for name, coef in model.params.items():
                if name == "const":
                    continue
                display_name = unsanitize_name(name)
                row[f"coeff({display_name})"] = coef
                row[f"p({display_name})"] = model.pvalues.get(name, np.nan)
        else:
            # sklearn — limited diagnostics
            ss_res = np.sum(res.residuals ** 2)
            ss_tot = np.sum((res.y - res.y.mean()) ** 2)
            row["R_squared"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            for name, coef in res.coefficients.items():
                if name == "const":
                    continue
                display_name = unsanitize_name(name)
                row[f"coeff({display_name})"] = coef

        rows.append(row)

    comparison = pd.DataFrame(rows)
    return comparison


# ---------------------------------------------------------------------------
# Backend: statsmodels
# ---------------------------------------------------------------------------

def _fit_ols(y, X, **kwargs) -> RegressionResult:
    """Ordinary Least Squares via statsmodels."""
    model = sm.OLS(y, X).fit()
    return _statsmodels_result(model, y, X, "ols")


def _fit_wls(y, X, weights=None, **kwargs) -> RegressionResult:
    """Weighted Least Squares via statsmodels."""
    if weights is None:
        warnings.warn("WLS without explicit weights; using equal weights (equivalent to OLS).")
        weights = np.ones(len(y))
    model = sm.WLS(y, X, weights=weights).fit()
    return _statsmodels_result(model, y, X, "wls")


def _fit_robust(y, X, **kwargs) -> RegressionResult:
    """Robust regression via statsmodels RLM (Huber T norm)."""
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    return _statsmodels_result(model, y, X, "robust")


def _statsmodels_result(model, y, X, reg_type) -> RegressionResult:
    """Wrap a statsmodels fitted model into RegressionResult."""
    n_pred = X.shape[1] - (1 if "const" in X.columns else 0)
    return RegressionResult(
        model=model,
        backend="statsmodels",
        regression_type=reg_type,
        y=y,
        X=X,
        fitted_values=model.fittedvalues.values if hasattr(model.fittedvalues, 'values') else np.array(model.fittedvalues),
        residuals=model.resid.values if hasattr(model.resid, 'values') else np.array(model.resid),
        coefficients=model.params,
        n_obs=int(model.nobs),
        n_predictors=n_pred,
    )


# ---------------------------------------------------------------------------
# Backend: sklearn (for Ridge/Lasso)
# ---------------------------------------------------------------------------

def _fit_ridge(y, X, alpha=1.0, **kwargs) -> RegressionResult:
    """Ridge regression via sklearn."""
    return _sklearn_fit(y, X, "ridge", alpha)


def _fit_lasso(y, X, alpha=1.0, **kwargs) -> RegressionResult:
    """Lasso regression via sklearn."""
    return _sklearn_fit(y, X, "lasso", alpha)


def _sklearn_fit(y, X, reg_type, alpha) -> RegressionResult:
    """Fit Ridge or Lasso using sklearn and wrap the result."""
    from sklearn.linear_model import Ridge, Lasso

    has_const = "const" in X.columns
    X_fit = X.drop(columns=["const"]) if has_const else X

    if reg_type == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=has_const)
    else:
        model = Lasso(alpha=alpha, fit_intercept=has_const, max_iter=10000)

    model.fit(X_fit.values, y.values)

    fitted = model.predict(X_fit.values)
    residuals = y.values - fitted

    # Build coefficient series matching statsmodels convention
    coef_dict = {}
    if has_const:
        coef_dict["const"] = model.intercept_
    for i, col in enumerate(X_fit.columns):
        coef_dict[col] = model.coef_[i]

    return RegressionResult(
        model=model,
        backend="sklearn",
        regression_type=reg_type,
        y=y,
        X=X,
        fitted_values=fitted,
        residuals=residuals,
        coefficients=pd.Series(coef_dict),
        n_obs=len(y),
        n_predictors=X_fit.shape[1],
    )
