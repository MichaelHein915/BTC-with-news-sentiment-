# BTC/USD Next-Day Return Prediction with News Sentiment

Predicting next-day Bitcoin log returns using a two-stage model that combines technical analysis with FinBERT-scored news sentiment.

## Research Question

Does lagged news sentiment improve next-day BTC/USD return prediction compared to a technical-only baseline?

## Data Sources

- **Bitcoin OHLCV** (~5 years daily): Downloaded from [Investing.com](https://www.investing.com/crypto/bitcoin/historical-data) (`Bitcoin Historical Data.csv`)
- **Crypto news articles**: Pre-scraped from CoinDesk, CoinTelegraph, Decrypt, Bitcoin Magazine, CryptoSlate, The Block (`news_prescraped_seed.csv`), plus live RSS feeds
- **Google Trends**: Search interest for "bitcoin", "btc", "bitcoin price" via pytrends
- **Fear & Greed Index**: From [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/)

## How to Run

```bash
pip install -r requirements.txt

# First time: scrapes news, runs FinBERT, caches everything
python main.py --full

# After that: uses cached sentiment, just trains and evaluates
python main.py

# Ablation comparison (A0 tech-only vs A1 tech+sentiment vs others)
python compare_models.py
```

## Project Structure

```
├── config.py                  # All paths, feature lists, hyperparameters
├── main.py                    # Main pipeline entry point
├── step1_data.py              # Load OHLCV + collect news articles
├── step2_features.py          # Technical indicators + sentiment aggregation
├── step3_sentiment.py         # FinBERT scoring (cached after first run)
├── step4_model_a.py           # LightGBM/XGBoost trained on technical features
├── step5_model_b.py           # ElasticNet residual corrector using sentiment
├── step6_evaluate.py          # Walk-forward evaluation with expanding window
├── step7_backtest.py          # Simulated trading with stop-loss/take-profit
├── compare_models.py          # Ablation study (A0-A4 + TwoStage)
├── direction_model_comparison.py  # Direction classification benchmark
├── improved_pipeline.py       # Direction classifier with confidence thresholds
├── presentation_summary.py    # Console summary for slides
│
├── Bitcoin Historical Data.csv    # Raw OHLCV input
├── news_prescraped_seed.csv       # Pre-scraped news articles
│
├── cache/                     # Cached FinBERT results, trends, FGI
│   ├── btc_news_daily_sentiment.csv
│   ├── btc_news_raw.csv
│   ├── btc_google_trends_daily.csv
│   └── btc_fear_greed_daily.csv
│
└── outputs/                   # All results and visualizations
    ├── model_comparison.csv           # Full ablation results table
    ├── ablation_summary.csv           # A0 vs A1 sentiment lift
    ├── direction_model_comparison.csv # Classification results
    ├── btc_merged_dataset.csv         # Final merged dataset
    ├── btc_two_stage_bundle.pkl       # Saved model bundle
    ├── btc_two_stage_evaluation_*.png # Prediction vs actual charts
    ├── btc_backtest_equity_curve_*.png # Equity curve charts
    └── BTC_Sentiment_Prediction_Presentation.pptx
```

## Model Architecture

**Stage 1 - Model A (LightGBM):** Trained on ~5 years of 16 technical indicators (returns, moving averages, RSI, MACD, Bollinger Bands, ATR, volume metrics) to predict `target_next_log_return`.

**Stage 2 - Model B (ElasticNetCV):** Trained on the last 65 days to predict Model A's residuals using 11 sentiment features (daily sentiment mean/std/range, positive/negative ratios, 3-day and 7-day momentum).

**Final prediction = Model A output + Model B correction**

All external features are lagged by 1 day to prevent look-ahead bias.

## Key Results

| Variant | Dir. Accuracy | Backtest Return | Sharpe |
|---------|:---:|:---:|:---:|
| A0 Tech Only (XGBoost) | 48% | +6.0% | 1.07 |
| A1 Tech+Sentiment (XGBoost) | 51% | +4.9% | 0.90 |
| **Two-Stage (XGBoost)** | **52%** | **+1.8%** | **0.99** |

The two-stage architecture achieves the highest directional accuracy and ~25% lower RMSE than single-stage models, with the smallest max drawdown (-1.9%).
