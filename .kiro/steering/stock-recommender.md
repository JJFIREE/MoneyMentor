---
inclusion: auto
---

# Stock Recommender System — Architecture & Implementation Guide

This document describes the AI Stock Recommender feature in the MoneyMentor app, located at `user_pages/stock_recommender.py`.

## Overview

The stock recommender is an ML-powered pipeline that predicts which Indian stocks (NSE-listed) are likely to go up in the next 5 trading days. It uses a soft-voting ensemble of Random Forest, Gradient Boosting, and Extra Trees classifiers trained on technical indicators, with news sentiment applied as a post-prediction ranking boost.

## Production Pipeline

```
SEBI Shareholding Data + Yahoo Finance Market Data + Economic Times News + Technical Indicators
    ↓
Feature Engineering Layer (19 features)
    ↓
Ensemble Classifier (RF + GB + ET soft voting)
    ↓
Post-Prediction News Boost (±10% from Groq sentiment)
    ↓
Ranking Engine (Top 5 Stocks)
    ↓
Groq / LLM Explanation Layer
    ↓
User Final Output
```

## Key Files

- `user_pages/stock_recommender.py` — Main recommender code (all phases)
- `models/shareholding_data.csv` — SEBI shareholding pattern data (2310+ companies, columns: COMPANY, PROMOTER & PROMOTER GROUP (A), PUBLIC (B))
- `models/stock_recommender_rf.pickle` — Trained model (saved after training)
- `models/live_stock_data.pickle` — Cached live stock data (pre-fetched during training for instant recommendations)
- `.streamlit/secrets.toml` — API keys (Groq at `REST.GROQ_API_KEY`, Gemini at `GOOGLE.GEMINI_API_KEY`)

## Data Sources

1. **SEBI Shareholding Data** (`models/shareholding_data.csv`): Promoter holding % used as institutional confidence signal. Higher promoter holding = more stable. Loaded via `load_shareholding_data()`.

2. **Yahoo Finance** (via `yfinance`): Historical price, volume, high/low data. Used for both training (2 years) and live features (6 months). Rate-limited — uses custom User-Agent header.

3. **Economic Times RSS**: `https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms` — Latest 30 market headlines. Fetched once, then scored per-stock via Groq.

4. **Groq API** (`llama-3.1-8b-instant`): Used for batch news sentiment scoring and final stock explanation generation. Key stored at `st.secrets['REST']['GROQ_API_KEY']`.

## Stock Universe

Built dynamically from the shareholding CSV. Only companies with a known NSE ticker mapping in `TICKER_MAP` + `EXTENDED_TICKER_MAP` are included (~200 mappings). The mapping is necessary because NSE ticker symbols don't follow a predictable pattern from company names (e.g., "Avenue Supermarts Limited" → `DMART.NS`).

To add more stocks: add entries to `EXTENDED_TICKER_MAP` dict in the format `"Company Name As In CSV": "TICKER.NS"`.

## Feature Engineering (19 Features)

| Feature | Description |
|---------|-------------|
| `institutional_score` | Promoter holding % / 100 (from SEBI CSV) |
| `holding_weight` | Same as institutional_score |
| `return_1d/5d/10d/20d` | Price returns over 1, 5, 10, 20 days |
| `price_vs_sma10/20/50` | Price position relative to moving averages |
| `sma10_vs_sma20` | SMA crossover signal (10 vs 20) |
| `sma20_vs_sma50` | SMA crossover signal (20 vs 50) |
| `volatility_20d` | 20-day rolling standard deviation of returns |
| `volume_ratio` | Recent 5-day volume / 20-day average volume |
| `rsi` | 14-period Relative Strength Index |
| `macd_hist` | MACD histogram (MACD line - signal line) |
| `momentum_10d` | 10-day rate of change |
| `atr` | Average True Range (14-period, normalized by price) |
| `bb_position` | Position within Bollinger Bands (0=lower, 1=upper) |
| `consecutive_days` | Net consecutive up/down days (last 5 days) |

Pretty display names are in `FEATURE_DISPLAY_NAMES` dict.

## Target Variable

- Binary classification: 1 = stock goes up >1% in 5 days, 0 = stock goes down >1%
- Samples with returns between -1% and +1% are dropped (ambiguous zone)
- This threshold filtering gives the model clearer signal to learn from

## Model Architecture

Soft-voting ensemble of 3 classifiers:

1. **RandomForestClassifier**: 500 trees, max_depth=10, balanced class weights
2. **GradientBoostingClassifier**: 300 trees, learning_rate=0.05, max_depth=5, subsample=0.8
3. **ExtraTreesClassifier**: 500 trees, max_depth=12, balanced class weights

The ensemble averages predicted probabilities from all 3 models. This reduces individual model errors and typically gives 3-5% better test accuracy than any single model.

## News Sentiment (Post-Prediction Boost)

News is NOT a model feature (historical news data unavailable). Instead:

1. During training: headlines fetched from Economic Times RSS
2. All headlines + all stock names sent to Groq in one batch API call
3. Groq returns sentiment scores per stock (-1.0 to +1.0)
4. After model predicts `prob_up`, news adjusts ranking: `final_score = prob_up + (news_score × 0.10)`
5. This shifts confidence by up to ±10% based on current news sentiment

## Two-Phase Architecture

**Training phase** (slow, done once/weekly):
- Fetches 2 years of historical data for all stocks
- Generates training samples with technical indicators
- Trains the ensemble model
- Pre-fetches current live data for all stocks
- Fetches and scores news sentiment
- Caches everything to disk (pickle files)

**Recommendation phase** (instant):
- Loads cached model + cached live data from disk
- Runs predictions (no API calls needed)
- Applies news boost
- Ranks and displays top 5
- Sends top 5 to Groq for explanation

## UI Structure

Two tabs in Streamlit:
- **📊 Get Recommendations**: Loads cached data, runs instant predictions, shows top 5 with reasons + Groq explanation
- **🔧 Train / Retrain Model**: Full pipeline — fetch data, train model, cache live data

Each recommended stock shows:
- Rank, company name, ticker, price
- Confidence % (model + news boost)
- Human-readable reason (generated from `generate_stock_reason()` using RSI, momentum, news, promoter data)

"View All Stock Scores" shows simplified table: Rank, Company, Ticker, Confidence %, Price.

Disclaimer at bottom: "This is not financial advice. Do your own research."

## Expected Accuracy

- Train: ~65-75%
- Test: ~55-62%
- For 5-day stock prediction, >55% test accuracy is useful (market is inherently noisy)
- The ensemble approach reduces the train/test gap (less overfitting)

## Dependencies

Key packages: `yfinance`, `scikit-learn`, `groq`, `beautifulsoup4`, `lxml`, `streamlit`, `pandas`, `numpy`

All listed in `requirements.txt`. Python 3.9 compatible (note: `@dataclass(kw_only=True)` in `ChatBot/chatbot.py` was changed to `@dataclass` for 3.9 compatibility).

## Common Issues

- **Yahoo Finance rate limiting**: Uses custom User-Agent. If data fetch fails, wait and retry.
- **Price showing N/A**: Fallback logic tries `stock.info['currentPrice']` if history close is NaN.
- **Groq API errors**: Returns neutral score (0.0) on failure, doesn't break the pipeline.
- **New stocks**: Must add company→ticker mapping to `EXTENDED_TICKER_MAP` manually.
