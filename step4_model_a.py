"""
step4_model_a.py - Model A: Gradient Boosted Tree Baseline
============================================================
Trains LightGBM on 5 years of technical (and optionally external) features.

Key design decisions:
  - TimeSeriesSplit CV (never shuffle financial data)
  - Excludes the last 65 days (news window) from training so that
    residuals used by Model B are truly out-of-sample
  - Early stopping on a held-out set to prevent overfitting
"""

import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from config import TECHNICAL_FEATURES, TARGET_COL, NEWS_WINDOW_DAYS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data preparation
# ---------------------------------------------------------------------------
def _prepare_data(df, tech_features, news_window):
    """Split data into training set (excl. news window) and return mask."""
    X = df[tech_features].copy()
    y = df[TARGET_COL].copy()

    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    n_train = len(X) - news_window
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]

    return X, y, X_train, y_train, mask, n_train


# ---------------------------------------------------------------------------
# LightGBM trainer
# ---------------------------------------------------------------------------
def _train_lightgbm(X_train, y_train, X_all):
    """Train LightGBM with TimeSeriesSplit CV + early stopping."""
    tscv = TimeSeriesSplit(n_splits=5)

    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05,
        max_depth=4, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.01, reg_lambda=0.01,
        min_child_samples=10,
        random_state=42, n_jobs=-1, verbose=-1,
    )

    cv_rmse = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        cv_rmse.append(rmse)
        print(f"    Fold {fold+1} RMSE: {rmse:.6f}  (iter={model.best_iteration_})")

    print(f"    Mean CV RMSE: {np.mean(cv_rmse):.6f} +/- {np.std(cv_rmse):.6f}")

    # Final fit with early stopping
    n = len(X_train)
    holdout_size = max(50, n // 10)
    X_fit,  X_hold = X_train.iloc[:-holdout_size], X_train.iloc[-holdout_size:]
    y_fit,  y_hold = y_train.iloc[:-holdout_size], y_train.iloc[-holdout_size:]

    model.fit(
        X_fit, y_fit,
        eval_set=[(X_hold, y_hold)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    print(f"    Final: {len(X_fit)} train + {len(X_hold)} holdout, "
          f"iter={model.best_iteration_}")

    importance = pd.Series(model.feature_importances_, index=X_all.columns.tolist())
    return model, np.mean(cv_rmse), np.std(cv_rmse), importance


# ---------------------------------------------------------------------------
# XGBoost trainer
# ---------------------------------------------------------------------------
def _train_xgboost(X_train, y_train, X_all):
    """Train XGBoost with TimeSeriesSplit CV + early stopping."""
    tscv = TimeSeriesSplit(n_splits=5)

    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05,
        max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.01, reg_lambda=0.01,
        min_child_weight=10,
        early_stopping_rounds=50,
        random_state=42, n_jobs=-1, verbosity=0,
    )

    cv_rmse = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = getattr(model, "best_iteration", model.n_estimators)
        rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        cv_rmse.append(rmse)
        print(f"    Fold {fold+1} RMSE: {rmse:.6f}  (iter={best_iter})")

    print(f"    Mean CV RMSE: {np.mean(cv_rmse):.6f} +/- {np.std(cv_rmse):.6f}")

    # Final fit with early stopping
    n = len(X_train)
    holdout_size = max(50, n // 10)
    X_fit, X_hold = X_train.iloc[:-holdout_size], X_train.iloc[-holdout_size:]
    y_fit, y_hold = y_train.iloc[:-holdout_size], y_train.iloc[-holdout_size:]

    model.fit(
        X_fit, y_fit,
        eval_set=[(X_hold, y_hold)],
        verbose=False,
    )
    best_iter = getattr(model, "best_iteration", model.n_estimators)
    print(f"    Final: {len(X_fit)} train + {len(X_hold)} holdout, "
          f"iter={best_iter}")

    importance = pd.Series(model.feature_importances_, index=X_all.columns.tolist())
    return model, np.mean(cv_rmse), np.std(cv_rmse), importance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train_model_a(df, tech_features=None, news_window=NEWS_WINDOW_DAYS):
    """
    Train LightGBM and XGBoost, compare, and return both.

    Returns:
        results: dict with keys 'lgbm' and 'xgb', each containing:
                   model, cv_rmse_mean, cv_rmse_std, importance
                 Also includes 'best' key pointing to the better model.
        mask:    Boolean mask of rows with no NaN in features
    """
    if tech_features is None:
        tech_features = [f for f in TECHNICAL_FEATURES if f in df.columns]

    print("\n" + "=" * 60)
    print("STEP 4: Model A - Training LightGBM + XGBoost")
    print("=" * 60)

    X, y, X_train, y_train, mask, n_train = _prepare_data(
        df, tech_features, news_window
    )
    print(f"  Training rows (excl. {news_window}-day news window): {n_train:,}")
    print(f"  Held-out news window rows: {news_window}")

    # --- LightGBM ---------------------------------------------------------
    print("\n  --- LightGBM ---")
    lgbm_model, lgbm_mean, lgbm_std, lgbm_imp = _train_lightgbm(
        X_train, y_train, X
    )

    # --- XGBoost ----------------------------------------------------------
    print("\n  --- XGBoost ---")
    xgb_model, xgb_mean, xgb_std, xgb_imp = _train_xgboost(
        X_train, y_train, X
    )

    # --- Comparison -------------------------------------------------------
    print("\n  --- Model A Comparison ---")
    print(f"    LightGBM  CV RMSE: {lgbm_mean:.6f} +/- {lgbm_std:.6f}")
    print(f"    XGBoost   CV RMSE: {xgb_mean:.6f} +/- {xgb_std:.6f}")

    if lgbm_mean <= xgb_mean:
        best = "lgbm"
        print(f"    Winner: LightGBM (lower RMSE by {xgb_mean - lgbm_mean:.6f})")
    else:
        best = "xgb"
        print(f"    Winner: XGBoost (lower RMSE by {lgbm_mean - xgb_mean:.6f})")

    results = {
        "lgbm": {
            "model": lgbm_model,
            "cv_rmse_mean": lgbm_mean,
            "cv_rmse_std": lgbm_std,
            "importance": lgbm_imp,
        },
        "xgb": {
            "model": xgb_model,
            "cv_rmse_mean": xgb_mean,
            "cv_rmse_std": xgb_std,
            "importance": xgb_imp,
        },
        "best": best,
    }
    return results, mask
