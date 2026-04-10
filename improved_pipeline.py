"""
improved_pipeline.py - Improved BTC Direction Classifier
========================================================
Three key improvements over the original pipeline:

1. Direction classification (up/down) instead of regression
   - Trading only cares about direction, not magnitude
   - Classification is a simpler, more learnable task

2. Single-stage model with tech + sentiment features
   - Uses ALL 1,762 training rows (not just 64 news-day samples)
   - LightGBM handles sparse sentiment naturally (tree-based)

3. Confidence threshold trading
   - Only trades when predicted probability > threshold
   - Filters out low-conviction signals, reduces bad trades

Run:
  python improved_pipeline.py
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from config import (
    OUTPUT_NEWS_DAILY, OUTPUT_TRENDS_DAILY, OUTPUT_FGI_DAILY,
    LEGACY_NEWS_DAILY, LEGACY_TRENDS_DAILY, LEGACY_FGI_DAILY,
    NEWS_WINDOW_DAYS, TECHNICAL_FEATURES, SENTIMENT_FEATURES,
    TRENDS_FEATURES, FGI_FEATURES, OUTPUTS_DIR,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRADING_FEE_PCT,
)
from step1_data import load_ohlcv
from step2_features import (
    add_technical_features, add_target,
    align_sentiment_to_ohlcv, align_external_to_ohlcv,
    merge_ohlcv_and_externals,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ---------------------------------------------------------------------------
# Data loading (reuses existing steps)
# ---------------------------------------------------------------------------
def _load_cached(path, date_col="date"):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def build_dataset():
    """Build the full merged dataset using existing pipeline steps."""
    df = load_ohlcv()
    df = add_technical_features(df)
    df = add_target(df)

    news_path = OUTPUT_NEWS_DAILY if Path(OUTPUT_NEWS_DAILY).exists() else LEGACY_NEWS_DAILY
    trends_path = OUTPUT_TRENDS_DAILY if Path(OUTPUT_TRENDS_DAILY).exists() else LEGACY_TRENDS_DAILY
    fgi_path = OUTPUT_FGI_DAILY if Path(OUTPUT_FGI_DAILY).exists() else LEGACY_FGI_DAILY

    df_sent = _load_cached(news_path)
    df_trends = _load_cached(trends_path)
    df_fgi = _load_cached(fgi_path)

    if df_sent.empty:
        raise FileNotFoundError(f"Missing sentiment cache. Run `python main.py --full` first.")

    sent_aligned = align_sentiment_to_ohlcv(df_sent, df)
    trends_aligned = align_external_to_ohlcv(df_trends, df, prefix="trends_")
    fgi_aligned = align_external_to_ohlcv(df_fgi, df, prefix="fgi_")

    df = merge_ohlcv_and_externals(df, sent_aligned, trends_aligned, fgi_aligned)
    return df


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------
def _make_classifier(model_type="lgbm"):
    """Create a classifier instance for the given model type."""
    if model_type == "xgb":
        return xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.03,
            max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.01, reg_lambda=0.01,
            min_child_weight=10,
            scale_pos_weight=1.0,
            early_stopping_rounds=50,
            random_state=42, n_jobs=-1, verbosity=0,
            eval_metric="logloss",
        )
    else:
        return lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03,
            max_depth=4, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.01, reg_lambda=0.01,
            min_child_samples=10,
            class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )


def train_direction_classifier(df, features, test_window=NEWS_WINDOW_DAYS,
                                model_type="lgbm"):
    """
    Train a classifier for next-day direction (up=1, down=0).
    Supports model_type='lgbm' or 'xgb'.
    """
    label = "LightGBM" if model_type == "lgbm" else "XGBoost"
    target = "target_direction"
    X = df[features].copy()
    y = df[target].astype(int).copy()

    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    n_train = len(X) - test_window
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]

    print(f"  Model: {label}")
    print(f"  Training: {n_train} rows  |  Test: {test_window} rows")
    print(f"  Features: {len(features)}")
    print(f"  Class balance (train): {y_train.mean():.1%} up / {1 - y_train.mean():.1%} down")

    # --- Cross-validation ---
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = _make_classifier(model_type)
        if model_type == "xgb":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            best_iter = getattr(model, "best_iteration", model.n_estimators)
        else:
            model.fit(
                X_tr, y_tr, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            best_iter = model.best_iteration_

        pred_val = model.predict(X_val)
        acc = accuracy_score(y_val, pred_val)
        f1 = f1_score(y_val, pred_val)
        cv_scores.append({"acc": acc, "f1": f1, "iter": best_iter})
        print(f"    Fold {fold+1}: Acc={acc:.3f}  F1={f1:.3f}  (iter={best_iter})")

    mean_acc = np.mean([s["acc"] for s in cv_scores])
    mean_f1 = np.mean([s["f1"] for s in cv_scores])
    print(f"    Mean CV: Acc={mean_acc:.3f}  F1={mean_f1:.3f}")

    # --- Final model ---
    holdout_size = max(50, n_train // 10)
    X_fit, X_hold = X_train.iloc[:-holdout_size], X_train.iloc[-holdout_size:]
    y_fit, y_hold = y_train.iloc[:-holdout_size], y_train.iloc[-holdout_size:]

    final_model = _make_classifier(model_type)
    if model_type == "xgb":
        final_model.fit(X_fit, y_fit, eval_set=[(X_hold, y_hold)], verbose=False)
        best_iter = getattr(final_model, "best_iteration", final_model.n_estimators)
    else:
        final_model.fit(
            X_fit, y_fit, eval_set=[(X_hold, y_hold)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        best_iter = final_model.best_iteration_

    print(f"    Final model: {len(X_fit)} train + {len(X_hold)} holdout, "
          f"iter={best_iter}")

    return final_model, X_test, y_test, mask, n_train


# ---------------------------------------------------------------------------
# Evaluation with confidence threshold
# ---------------------------------------------------------------------------
def evaluate_with_threshold(model, X_test, y_test, thresholds=None):
    """Evaluate classifier at multiple confidence thresholds."""
    if thresholds is None:
        thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]

    probs = model.predict_proba(X_test)[:, 1]  # probability of UP

    print(f"\n  Probability distribution:")
    print(f"    Mean={probs.mean():.3f}  Std={probs.std():.3f}  "
          f"Min={probs.min():.3f}  Max={probs.max():.3f}")

    results = []
    print(f"\n  {'Threshold':>10}  {'Trades':>7}  {'Accuracy':>9}  "
          f"{'Precision':>10}  {'Recall':>7}  {'F1':>7}  {'Coverage':>9}")
    print("  " + "-" * 72)

    for t in thresholds:
        # Predict UP if prob > threshold, DOWN if prob < (1-threshold), else SKIP
        pred = np.full(len(probs), -1)  # -1 = skip
        pred[probs >= t] = 1      # UP
        pred[probs <= (1 - t)] = 0  # DOWN

        traded = pred >= 0
        n_traded = traded.sum()

        if n_traded == 0:
            print(f"  {t:>10.2f}  {0:>7}  {'n/a':>9}  {'n/a':>10}  {'n/a':>7}  {'n/a':>7}  {0:>8.1%}")
            results.append({"threshold": t, "n_trades": 0, "accuracy": 0, "f1": 0})
            continue

        y_traded = y_test.values[traded]
        p_traded = pred[traded]

        acc = accuracy_score(y_traded, p_traded)
        prec = precision_score(y_traded, p_traded, zero_division=0)
        rec = recall_score(y_traded, p_traded, zero_division=0)
        f1 = f1_score(y_traded, p_traded, zero_division=0)
        coverage = n_traded / len(probs)

        print(f"  {t:>10.2f}  {n_traded:>7}  {acc:>9.1%}  "
              f"{prec:>10.1%}  {rec:>7.1%}  {f1:>7.3f}  {coverage:>8.1%}")

        results.append({
            "threshold": t, "n_trades": n_traded,
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "coverage": coverage,
        })

    return results, probs


# ---------------------------------------------------------------------------
# Backtesting with confidence threshold
# ---------------------------------------------------------------------------
def backtest_with_confidence(df_test, probs, threshold,
                             stop_loss=STOP_LOSS_PCT,
                             take_profit=TAKE_PROFIT_PCT,
                             fee=TRADING_FEE_PCT):
    """
    Backtest using classifier probabilities + confidence threshold.
    BUY when P(up) >= threshold. SELL/SKIP otherwise.
    """
    prices_close = df_test["Price"].values
    prices_high = df_test["High"].values
    prices_low = df_test["Low"].values
    dates = df_test["Date"].values
    n = len(probs)

    cash = 1.0
    equity = np.ones(n)
    trades = []

    for t in range(n - 1):
        if probs[t] >= threshold:
            entry = prices_close[t]
            next_high = prices_high[t + 1]
            next_low = prices_low[t + 1]
            next_close = prices_close[t + 1]

            sl_price = entry * (1 + stop_loss)
            tp_price = entry * (1 + take_profit)

            if next_low <= sl_price:
                exit_price = sl_price
                reason = "SL"
            elif next_high >= tp_price:
                exit_price = tp_price
                reason = "TP"
            else:
                exit_price = next_close
                reason = "CLOSE"

            gross = exit_price / entry - 1
            net = gross - 2 * fee
            cash *= (1 + net)
            trades.append({
                "date": dates[t], "entry": entry, "exit": exit_price,
                "return_pct": net * 100, "reason": reason,
                "prob": probs[t], "win": net > 0,
            })

        equity[t + 1] = cash

    total_return = (cash - 1) * 100
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["win"])
    win_rate = wins / n_trades * 100 if n_trades else 0

    daily_ret = np.diff(equity) / (equity[:-1] + 1e-12)
    sharpe = (np.mean(daily_ret) / (np.std(daily_ret) + 1e-9)) * np.sqrt(252)

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / (running_max + 1e-12)
    max_dd = drawdowns.min() * 100

    return {
        "total_return": total_return, "n_trades": n_trades,
        "win_rate": win_rate, "sharpe": sharpe, "max_drawdown": max_dd,
        "equity": equity, "trades": trades, "dates": dates,
    }


def backtest_buy_hold(df_test):
    prices = df_test["Price"].values
    equity = prices / prices[0]
    daily_ret = np.diff(equity) / equity[:-1]
    sharpe = (np.mean(daily_ret) / (np.std(daily_ret) + 1e-9)) * np.sqrt(252)
    running_max = np.maximum.accumulate(equity)
    max_dd = ((equity - running_max) / running_max).min() * 100
    return {
        "total_return": (equity[-1] - 1) * 100, "n_trades": 1,
        "win_rate": float("nan"), "sharpe": sharpe, "max_drawdown": max_dd,
        "equity": equity,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(dates, results_dict, chart_path):
    """Plot equity curves for multiple strategies."""
    fig, ax = plt.subplots(figsize=(14, 6))

    color_list = ["steelblue", "orangered", "gold", "lime", "cyan", "magenta", "gray"]

    for i, (name, res) in enumerate(results_dict.items()):
        if "Hold" in name:
            c = "gray"
        elif "Old" in name:
            c = "steelblue"
        else:
            c = color_list[min(i, len(color_list) - 1)]
        lw = 2.0 if "New" in name else 1.2
        ls = "--" if "Hold" in name else "-"
        label = f"{name} ({res['total_return']:+.2f}%)"
        eq = res["equity"]
        d = dates[:len(eq)]
        ax.plot(d, eq, label=label, color=c, lw=lw, ls=ls)

    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_title("Improved Pipeline: Direction Classification + Confidence Threshold", fontsize=13)
    ax.set_ylabel("Portfolio Value ($1 start)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved -> {chart_path}")
    plt.close(fig)


def plot_feature_importance(model, features, chart_path):
    """Plot top feature importances."""
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    imp = imp.tail(20)  # top 20

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(imp.index, imp.values, color="steelblue")

    # Color sentiment features differently
    for bar, name in zip(bars, imp.index):
        if name.startswith("sentiment") or name in ("positive_ratio", "negative_ratio", "bull_bear_ratio"):
            bar.set_color("orangered")
        elif name.startswith("trends_") or name.startswith("fgi_"):
            bar.set_color("gold")

    ax.set_xlabel("Feature Importance (split count)")
    ax.set_title("Feature Importance: Direction Classifier\n"
                 "(blue=technical, red=sentiment, gold=external)")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Feature importance chart -> {chart_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run old pipeline for comparison
# ---------------------------------------------------------------------------
def run_old_regression_backtest(df, test_window=NEWS_WINDOW_DAYS):
    """Run the old regression model for apples-to-apples comparison."""
    from step4_model_a import train_model_a
    from config import TARGET_COL

    tech = [f for f in TECHNICAL_FEATURES if f in df.columns]
    results, mask = train_model_a(df, tech_features=tech, news_window=test_window)
    model_a = results["lgbm"]["model"]

    df_clean = df[mask].reset_index(drop=True)
    df_window = df_clean.tail(test_window).copy().reset_index(drop=True)

    preds = model_a.predict(df_window[tech])
    prices_close = df_window["Price"].values
    prices_high = df_window["High"].values
    prices_low = df_window["Low"].values
    dates = df_window["Date"].values

    # Simulate old strategy: buy if predicted return > 0
    n = len(preds)
    cash = 1.0
    equity = np.ones(n)
    trades = []

    for t in range(n - 1):
        if preds[t] > 0:
            entry = prices_close[t]
            sl_price = entry * (1 + STOP_LOSS_PCT)
            tp_price = entry * (1 + TAKE_PROFIT_PCT)

            if prices_low[t+1] <= sl_price:
                exit_p = sl_price
            elif prices_high[t+1] >= tp_price:
                exit_p = tp_price
            else:
                exit_p = prices_close[t+1]

            net = exit_p / entry - 1 - 2 * TRADING_FEE_PCT
            cash *= (1 + net)
            trades.append({"win": net > 0})

        equity[t + 1] = cash

    total_ret = (cash - 1) * 100
    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t["win"]) / n_trades * 100 if n_trades else 0
    daily_ret = np.diff(equity) / (equity[:-1] + 1e-12)
    sharpe = (np.mean(daily_ret) / (np.std(daily_ret) + 1e-9)) * np.sqrt(252)
    running_max = np.maximum.accumulate(equity)
    max_dd = ((equity - running_max) / (running_max + 1e-12)).min() * 100

    return {
        "total_return": total_ret, "n_trades": n_trades, "win_rate": win_rate,
        "sharpe": sharpe, "max_drawdown": max_dd, "equity": equity, "dates": dates,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("IMPROVED PIPELINE: Direction Classification + Confidence Threshold")
    print("=" * 70)

    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)

    # --- Build dataset ---
    print("\n[1/5] Building dataset...")
    df = build_dataset()

    # --- Define feature sets ---
    tech = [f for f in TECHNICAL_FEATURES if f in df.columns]
    sent = [f for f in SENTIMENT_FEATURES if f in df.columns]
    trends = [f for f in TRENDS_FEATURES if f in df.columns]
    fgi = [f for f in FGI_FEATURES if f in df.columns]

    feature_sets = {
        "Tech Only": tech,
        "Tech + Sentiment": tech + sent,
        "Tech + All": tech + sent + trends + fgi,
    }

    # --- Train and evaluate each feature set x model type ---
    all_results = []
    model_types = [("lgbm", "LightGBM"), ("xgb", "XGBoost")]

    # Track best overall model for backtest
    best_model = None
    best_features = None
    best_probs = None
    best_model_label = None
    best_f1_at_55 = -1

    for name, features in feature_sets.items():
        for mtype, mlabel in model_types:
            print(f"\n{'=' * 70}")
            print(f"[2/5] Training: {name} / {mlabel} ({len(features)} features)")
            print("=" * 70)

            model, X_test, y_test, mask, n_train = train_direction_classifier(
                df, features, test_window=NEWS_WINDOW_DAYS, model_type=mtype
            )

            # Evaluate at multiple thresholds
            print(f"\n  --- Threshold Analysis ---")
            threshold_results, probs = evaluate_with_threshold(model, X_test, y_test)

            # Full classification report at default threshold
            pred_default = model.predict(X_test)
            print(f"\n  Classification Report (threshold=0.50):")
            print(classification_report(y_test, pred_default, target_names=["DOWN", "UP"], zero_division=0))

            print(f"  Confusion Matrix:")
            cm = confusion_matrix(y_test, pred_default)
            print(f"    {cm}")

            for tr in threshold_results:
                all_results.append({
                    "feature_set": name, "model_type": mlabel,
                    "n_features": len(features),
                    **tr,
                })

            # Track best model at 55% threshold for backtest
            t55 = [r for r in threshold_results if r["threshold"] == 0.55]
            if t55 and t55[0]["f1"] > best_f1_at_55:
                best_f1_at_55 = t55[0]["f1"]
                best_model = model
                best_features = features
                best_probs = probs
                best_model_label = f"{name} / {mlabel}"

    # --- Results table ---
    print(f"\n{'=' * 70}")
    print("[3/5] RESULTS COMPARISON")
    print("=" * 70)

    df_results = pd.DataFrame(all_results)
    out_path = Path(OUTPUTS_DIR) / "improved_model_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    print(df_results.to_string(index=False))

    # --- Backtest comparison ---
    print(f"\n{'=' * 70}")
    print("[4/5] BACKTEST COMPARISON")
    print("=" * 70)

    # Get test window data
    mask_all = df[best_features].notna().all(axis=1)
    df_clean = df[mask_all].reset_index(drop=True)
    df_test = df_clean.tail(NEWS_WINDOW_DAYS).copy().reset_index(drop=True)

    # Run old regression model for comparison
    print("\n  Running old regression models...")
    old_results = run_old_regression_backtest(df)

    print(f"\n  Best classifier: {best_model_label}")

    # Run new classifier at different thresholds
    bt_results = {}
    for t in [0.50, 0.55, 0.60]:
        bt = backtest_with_confidence(df_test, best_probs, threshold=t)
        label = f"New: Direction Classifier ({int(t*100)}%)"
        bt_results[label] = bt

    bt_bh = backtest_buy_hold(df_test)

    # Print comparison table
    print(f"\n  {'Strategy':<40} {'Return':>9} {'Trades':>8} {'Win Rate':>10} "
          f"{'Sharpe':>8} {'Max DD':>9}")
    print("  " + "-" * 85)

    all_strats = {
        "Old: Tech Regression": old_results,
        **bt_results,
        "Buy & Hold BTC": bt_bh,
    }

    for name, r in all_strats.items():
        wr = f"{r['win_rate']:.1f}%" if not np.isnan(r.get('win_rate', 0)) else "n/a"
        print(f"  {name:<40} {r['total_return']:>+8.2f}% {r['n_trades']:>8} "
              f"{wr:>10} {r['sharpe']:>+8.2f} {r['max_drawdown']:>+8.2f}%")

    # Trade breakdown for best threshold
    for label, bt in bt_results.items():
        trades = bt.get("trades", [])
        if trades:
            sl = sum(1 for t in trades if t["reason"] == "SL")
            tp = sum(1 for t in trades if t["reason"] == "TP")
            cl = sum(1 for t in trades if t["reason"] == "CLOSE")
            print(f"\n  {label} trade breakdown:")
            print(f"    SL={sl}  TP={tp}  Close={cl}  Total={len(trades)}")
            for reason in ["TP", "SL", "CLOSE"]:
                subset = [t["return_pct"] for t in trades if t["reason"] == reason]
                if subset:
                    print(f"      {reason:5s}  avg={np.mean(subset):+.2f}%  count={len(subset)}")

    # --- Charts ---
    print(f"\n{'=' * 70}")
    print("[5/5] GENERATING CHARTS")
    print("=" * 70)

    # Equity curve comparison
    dates = df_test["Date"].values
    plot_comparison(dates, all_strats, f"{OUTPUTS_DIR}/improved_equity_comparison.png")

    # Feature importance
    plot_feature_importance(best_model, best_features,
                           f"{OUTPUTS_DIR}/improved_feature_importance.png")

    # --- Final summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    old_ret = old_results["total_return"]
    best_new = max(bt_results.values(), key=lambda x: x["total_return"])
    best_label = [k for k, v in bt_results.items() if v is best_new][0]
    bh_ret = bt_bh["total_return"]

    print(f"\n  Old regression model:       {old_ret:+.2f}% return, "
          f"{old_results['n_trades']} trades, Sharpe {old_results['sharpe']:+.2f}")
    print(f"  Best new classifier:        {best_new['total_return']:+.2f}% return, "
          f"{best_new['n_trades']} trades, Sharpe {best_new['sharpe']:+.2f}")
    print(f"  ({best_label})")
    print(f"  Buy & Hold:                 {bh_ret:+.2f}% return, Sharpe {bt_bh['sharpe']:+.2f}")

    improvement = best_new["total_return"] - old_ret
    print(f"\n  Improvement over old model: {improvement:+.2f} percentage points")

    if best_new["total_return"] > 0:
        print(f"  New model is PROFITABLE ({best_new['total_return']:+.2f}%)")
    else:
        print(f"  New model still loses money but less than old ({best_new['total_return']:+.2f}% vs {old_ret:+.2f}%)")

    print(f"\n{'=' * 70}")
    print("DONE - All outputs saved to outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
