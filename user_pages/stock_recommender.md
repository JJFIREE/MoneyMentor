# Stock Recommender

## Overview
The Stock Recommender is an ML-powered module that generates buy/hold recommendations for NSE-listed Indian stocks. It combines technical analysis, SEBI shareholding data, and Groq AI-driven news sentiment analysis to rank stocks by confidence.

## Architecture

### Data Sources
| Source | What it provides |
|---|---|
| **Yahoo Finance** (yfinance) | Historical price data, volume, OHLC for 250+ NSE stocks |
| **SEBI Shareholding CSV** (`models/shareholding_data.csv`) | Promoter and institutional holding percentages |
| **RSS Feeds** (Economic Times, MoneyControl) | Market headlines from the last 15 days |
| **Groq AI** (LLaMA 3.1 8B) | News sentiment scoring and stock-to-headline mapping |

### ML Model
An ensemble of three classifiers trained via majority voting:
- **Random Forest** (500 estimators)
- **Gradient Boosting** (300 estimators)
- **Extra Trees** (500 estimators)

**Target:** Binary classification — will the stock go up >1% or down >1% in the next 5 trading days.

### Feature Set (19 features)
| Category | Features |
|---|---|
| **Returns** | 1-day, 5-day, 10-day, 20-day returns |
| **Moving Averages** | Price vs SMA10/20/50, SMA10 vs SMA20, SMA20 vs SMA50 |
| **Momentum** | RSI, MACD Histogram, 10-day momentum |
| **Volatility** | 20-day volatility, Average True Range, Bollinger Band position |
| **Volume** | Volume ratio (5-day vs 20-day average) |
| **Trend** | Consecutive up/down days |
| **Fundamentals** | Institutional score, holding weight (from SEBI data) |

### Scoring Formula
The final confidence score is a weighted blend:

```
final_score = 0.50 * model_confidence + 0.50 * news_normalized
```

- `model_confidence`: Probability of upward movement from the ensemble model (0 to 1)
- `news_normalized`: Groq sentiment score mapped from [-1, 1] to [0, 1]

News sentiment is the single most influential signal since the model's 50% share is distributed across 19 features.

## News Sentiment Pipeline

1. **Fetch** — Headlines are pulled from 4 RSS feeds (Economic Times, MoneyControl) filtered to the last 15 days. Duplicates are removed.
2. **Analyze** — Headlines are sent to Groq AI in batches (25 stocks per call, 25 headlines per call) with an educational-purpose prompt. Groq identifies which stocks are affected (directly or via sector/market impact) and assigns sentiment scores.
3. **Map** — Groq returns headline numbers per stock, which are mapped back to actual headline text for display.
4. **Display** — A dedicated "Stocks Affected by Recent News" section shows all stocks with matched headlines, sorted by impact strength, with positive/negative indicators.

## User Flow

### Train / Retrain Model Tab
1. Loads SEBI shareholding data
2. Fetches market headlines from RSS feeds
3. Sends headlines to Groq AI for sentiment analysis (batched)
4. Generates training data from 2 years of Yahoo Finance history
5. Trains the ensemble classifier
6. Pre-fetches and caches live data for all stocks (6 months of history + news scores)

### Get Recommendations Tab
1. Loads the cached model and live data
2. Runs predictions using the ensemble model + news sentiment blend
3. Displays top 5 recommended stocks with confidence scores and AI-generated reasons
4. Shows a separate "Stocks Affected by Recent News" section with matched headlines
5. Provides a "View All Stock Scores" expandable table
6. Generates an AI analysis summary via Groq

## File Structure
```
user_pages/stock_recommender.py   # Main module
models/stock_recommender_rf.pickle # Trained model (generated, gitignored)
models/live_stock_data.pickle      # Cached live data (generated, gitignored)
models/shareholding_data.csv       # SEBI shareholding input data
```

## Dependencies
- `yfinance` — Yahoo Finance API
- `scikit-learn` — ML classifiers
- `groq` — Groq AI client (LLaMA 3.1 8B)
- `beautifulsoup4` — RSS feed parsing
- `streamlit` — UI framework
- `pandas`, `numpy` — Data processing

## Notes
- Pickle files are gitignored — the model and live data are regenerated via the Train tab.
- The stock universe is built dynamically from `shareholding_data.csv` matched against a known ticker map of 250+ NSE companies.
- Groq API calls use batching (25 stocks per call) to stay within token limits.
- This tool is for **educational purposes only** and is not financial advice.
