import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from groq import Groq
import requests
from bs4 import BeautifulSoup
import json
import os
import pickle
import logging
from datetime import datetime, timedelta

# Suppress verbose logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- Rate limit fix ---
import yfinance.utils
yfinance.utils.get_user_agent = lambda: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# --- CONFIG ---
MODEL_PATH = "models/stock_recommender_rf.pickle"
LIVE_DATA_PATH = "models/live_stock_data.pickle"
SHAREHOLDING_CSV_PATH = "models/shareholding_data.csv"
TOP_N = 5
FORECAST_DAYS = 5

# --- KNOWN TICKER MAPPINGS ---
TICKER_MAP = {
    "Reliance Industries Limited": "RELIANCE.NS",
    "Tata Consultancy Services Limited": "TCS.NS",
    "HDFC Bank Limited": "HDFCBANK.NS",
    "Infosys Limited": "INFY.NS",
    "ICICI Bank Limited": "ICICIBANK.NS",
    "Hindustan Unilever Limited": "HINDUNILVR.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel Limited": "BHARTIARTL.NS",
    "ITC Limited": "ITC.NS",
    "Kotak Mahindra Bank Limited": "KOTAKBANK.NS",
    "Larsen & Toubro Limited": "LT.NS",
    "Axis Bank Limited": "AXISBANK.NS",
    "Bajaj Finance Limited": "BAJFINANCE.NS",
    "Maruti Suzuki India Limited": "MARUTI.NS",
    "Asian Paints Limited": "ASIANPAINT.NS",
    "Titan Company Limited": "TITAN.NS",
    "Sun Pharmaceutical Industries Limited": "SUNPHARMA.NS",
    "Wipro Limited": "WIPRO.NS",
    "Adani Enterprises Limited": "ADANIENT.NS",
    "Tata Motors Limited": "TATAMOTORS.NS",
    "Power Grid Corporation of India Limited": "POWERGRID.NS",
    "NTPC Limited": "NTPC.NS",
    "HCL Technologies Limited": "HCLTECH.NS",
    "Bajaj Finserv Limited": "BAJAJFINSV.NS",
    "Tech Mahindra Limited": "TECHM.NS",
    "Aditya Birla Capital Limited": "ABCAPITAL.NS",
    "Adani Green Energy Limited": "ADANIGREEN.NS",
    "Adani Ports and Special Economic Zone Limited": "ADANIPORTS.NS",
    "Adani Power Limited": "ADANIPOWER.NS",
    "Cipla Limited": "CIPLA.NS",
    "Dr. Reddy's Laboratories Limited": "DRREDDY.NS",
    "Divi's Laboratories Limited": "DIVISLAB.NS",
    "Eicher Motors Limited": "EICHERMOT.NS",
    "Coal India Limited": "COALINDIA.NS",
    "Bharat Electronics Limited": "BEL.NS",
    "Bharat Petroleum Corporation Limited": "BPCL.NS",
    "Canara Bank": "CANBK.NS",
    "Dabur India Limited": "DABUR.NS",
    "DLF Limited": "DLF.NS",
    "Dixon Technologies (India) Limited": "DIXON.NS",
    "Biocon Limited": "BIOCON.NS",
    "Colgate Palmolive (India) Limited": "COLPAL.NS",
    "Bosch Limited": "BOSCHLTD.NS",
    "ABB India Limited": "ABB.NS",
    "Ashok Leyland Limited": "ASHOKLEY.NS",
    "Apollo Hospitals Enterprise Limited": "APOLLOHOSP.NS",
    "Britannia Industries Limited": "BRITANNIA.NS",
    "Cochin Shipyard Limited": "COCHINSHIP.NS",
    "Container Corporation of India Limited": "CONCOR.NS",
    "Coromandel International Limited": "COROMANDEL.NS",
    "Crompton Greaves Consumer Electricals Limited": "CROMPTON.NS",
    "Cummins India Limited": "CUMMINSIND.NS",
    "Deepak Nitrite Limited": "DEEPAKNTR.NS",
    "Avenue Supermarts Limited": "DMART.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Aurobindo Pharma Limited": "AUROPHARMA.NS",
    "Balkrishna Industries Limited": "BALKRISIND.NS",
    "Bandhan Bank Limited": "BANDHANBNK.NS",
    "Cholamandalam Investment and Finance Company Limited": "CHOLAFIN.NS",
    "Blue Star Limited": "BLUESTARCO.NS",
    "Alkem Laboratories Limited": "ALKEM.NS",
    "Astral Limited": "ASTRAL.NS",
}

# --- EXTENDED TICKER MAP: Auto-generated for common NSE-listed companies ---
EXTENDED_TICKER_MAP = {
    "Tata Steel Limited": "TATASTEEL.NS",
    "Tata Power Company Limited": "TATAPOWER.NS",
    "Tata Consumer Products Limited": "TATACONSUM.NS",
    "Tata Communications Limited": "TATACOMM.NS",
    "Tata Chemicals Limited": "TATACHEM.NS",
    "Tata Elxsi Limited": "TATELXSI.NS",
    "Vedanta Limited": "VEDL.NS",
    "JSW Steel Limited": "JSWSTEEL.NS",
    "JSW Energy Limited": "JSWENERGY.NS",
    "Hindalco Industries Limited": "HINDALCO.NS",
    "Grasim Industries Limited": "GRASIM.NS",
    "UltraTech Cement Limited": "ULTRACEMCO.NS",
    "Shree Cement Limited": "SHREECEM.NS",
    "Ambuja Cements Limited": "AMBUJACEM.NS",
    "ACC Limited": "ACC.NS",
    "IndusInd Bank Limited": "INDUSINDBK.NS",
    "Federal Bank Limited": "FEDERALBNK.NS",
    "IDFC First Bank Limited": "IDFCFIRSTB.NS",
    "Punjab National Bank": "PNB.NS",
    "Bank of India": "BANKINDIA.NS",
    "Indian Bank": "INDIANB.NS",
    "Union Bank of India": "UNIONBANK.NS",
    "Central Bank of India": "CENTRALBK.NS",
    "IDBI Bank Limited": "IDBI.NS",
    "Indian Overseas Bank": "IOB.NS",
    "Godrej Consumer Products Limited": "GODREJCP.NS",
    "Godrej Properties Limited": "GODREJPROP.NS",
    "Godrej Industries Limited": "GODREJIND.NS",
    "Pidilite Industries Limited": "PIDILITIND.NS",
    "Havells India Limited": "HAVELLS.NS",
    "Voltas Limited": "VOLTAS.NS",
    "Whirlpool of India Limited": "WHIRLPOOL.NS",
    "Page Industries Limited": "PAGEIND.NS",
    "Jubilant FoodWorks Limited": "JUBLFOOD.NS",
    "Zomato Limited": "ZOMATO.NS",
    "Info Edge (India) Limited": "NAUKRI.NS",
    "Persistent Systems Limited": "PERSISTENT.NS",
    "Mphasis Limited": "MPHASIS.NS",
    "L&T Technology Services Limited": "LTTS.NS",
    "L&T Finance Limited": "LTF.NS",
    "Mindtree Limited": "LTIM.NS",
    "Polycab India Limited": "POLYCAB.NS",
    "KEI Industries Limited": "KEI.NS",
    "Siemens Limited": "SIEMENS.NS",
    "Honeywell Automation India Limited": "HONAUT.NS",
    "Torrent Pharmaceuticals Limited": "TORNTPHARM.NS",
    "Lupin Limited": "LUPIN.NS",
    "Glenmark Pharmaceuticals Limited": "GLENMARK.NS",
    "Laurus Labs Limited": "LAURUSLABS.NS",
    "Ipca Laboratories Limited": "IPCALAB.NS",
    "Granules India Limited": "GRANULES.NS",
    "Natco Pharma Limited": "NATCOPHARM.NS",
    "Metropolis Healthcare Limited": "METROPOLIS.NS",
    "Max Healthcare Institute Limited": "MAXHEALTH.NS",
    "Fortis Healthcare Limited": "FORTIS.NS",
    "SBI Life Insurance Company Limited": "SBILIFE.NS",
    "HDFC Life Insurance Company Limited": "HDFCLIFE.NS",
    "ICICI Lombard General Insurance Company Limited": "ICICIGI.NS",
    "ICICI Prudential Life Insurance Company Limited": "ICICIPRULI.NS",
    "General Insurance Corporation of India": "GICRE.NS",
    "New India Assurance Company Limited": "NIACL.NS",
    "Bajaj Auto Limited": "BAJAJ-AUTO.NS",
    "Hero MotoCorp Limited": "HEROMOTOCO.NS",
    "TVS Motor Company Limited": "TVSMOTOR.NS",
    "Mahindra & Mahindra Limited": "M&M.NS",
    "Motherson Sumi Wiring India Limited": "MSUMI.NS",
    "MRF Limited": "MRF.NS",
    "Apollo Tyres Limited": "APOLLOTYRE.NS",
    "Bharat Forge Limited": "BHARATFORG.NS",
    "Tube Investments of India Limited": "TIINDIA.NS",
    "Schaeffler India Limited": "SCHAEFFLER.NS",
    "Oil and Natural Gas Corporation Limited": "ONGC.NS",
    "Indian Oil Corporation Limited": "IOC.NS",
    "Hindustan Petroleum Corporation Limited": "HINDPETRO.NS",
    "GAIL (India) Limited": "GAIL.NS",
    "Petronet LNG Limited": "PETRONET.NS",
    "Adani Total Gas Limited": "ATGL.NS",
    "Indraprastha Gas Limited": "IGL.NS",
    "Mahanagar Gas Limited": "MGL.NS",
    "Gujarat Gas Limited": "GUJGASLTD.NS",
    "NHPC Limited": "NHPC.NS",
    "Tata Power Company Limited": "TATAPOWER.NS",
    "Adani Transmission Limited": "ADANITRANS.NS",
    "Torrent Power Limited": "TORNTPOWER.NS",
    "CESC Limited": "CESC.NS",
    "Oberoi Realty Limited": "OBEROIRLTY.NS",
    "Prestige Estates Projects Limited": "PRESTIGE.NS",
    "Brigade Enterprises Limited": "BRIGADE.NS",
    "Phoenix Mills Limited": "PHOENIXLTD.NS",
    "Sobha Limited": "SOBHA.NS",
    "Sunteck Realty Limited": "SUNTECK.NS",
    "Dalmia Bharat Limited": "DALBHARAT.NS",
    "Ramco Cements Limited": "RAMCOCEM.NS",
    "JK Cement Limited": "JKCEMENT.NS",
    "Star Cement Limited": "STARCEMENT.NS",
    "UPL Limited": "UPL.NS",
    "PI Industries Limited": "PIIND.NS",
    "Bayer CropScience Limited": "BAYERCROP.NS",
    "Rallis India Limited": "RALLIS.NS",
    "Chambal Fertilizers & Chemicals Limited": "CHAMBLFERT.NS",
    "Coromandel International Limited": "COROMANDEL.NS",
    "Marico Limited": "MARICO.NS",
    "Emami Limited": "EMAMILTD.NS",
    "Trent Limited": "TRENT.NS",
    "Aditya Birla Fashion and Retail Limited": "ABFRL.NS",
    "Shoppers Stop Limited": "SHOPERSTOP.NS",
    "V-Mart Retail Limited": "VMART.NS",
    "Indian Railway Catering and Tourism Corporation Limited": "IRCTC.NS",
    "InterGlobe Aviation Limited": "INDIGO.NS",
    "Container Corporation of India Limited": "CONCOR.NS",
    "Delhivery Limited": "DELHIVERY.NS",
    "Solar Industries India Limited": "SOLARINDS.NS",
    "Bharat Dynamics Limited": "BDL.NS",
    "Hindustan Aeronautics Limited": "HAL.NS",
    "Mazagon Dock Shipbuilders Limited": "MAZDOCK.NS",
    "Garden Reach Shipbuilders & Engineers Limited": "GRSE.NS",
    "Data Patterns (India) Limited": "DATAPATTNS.NS",
    "Paras Defence and Space Technologies Limited": "PARAS.NS",
    "Manappuram Finance Limited": "MANAPPURAM.NS",
    "Muthoot Finance Limited": "MUTHOOTFIN.NS",
    "Shriram Finance Limited": "SHRIRAMFIN.NS",
    "Poonawalla Fincorp Limited": "POONAWALLA.NS",
    "LIC Housing Finance Limited": "LICHSGFIN.NS",
    "PNB Housing Finance Limited": "PNBHOUSING.NS",
    "Aavas Financiers Limited": "AAVAS.NS",
    "AU Small Finance Bank Limited": "AUBANK.NS",
    "Ujjivan Small Finance Bank Limited": "UJJIVANSFB.NS",
    "Equitas Small Finance Bank Limited": "EQUITASBNK.NS",
    "Multi Commodity Exchange of India Limited": "MCX.NS",
    "BSE Limited": "BSE.NS",
    "CRISIL Limited": "CRISIL.NS",
    "Computer Age Management Services Limited": "CAMS.NS",
    "Angel One Limited": "ANGELONE.NS",
    "HDFC Asset Management Company Limited": "HDFCAMC.NS",
    "Nippon Life India Asset Management Limited": "NAM-INDIA.NS",
    "SBI Cards and Payment Services Limited": "SBICARD.NS",
    "Varun Beverages Limited": "VBL.NS",
    "United Spirits Limited": "UNITDSPR.NS",
    "United Breweries Limited": "UBL.NS",
    "Radico Khaitan Limited": "RADICO.NS",
    "Nestle India Limited": "NESTLEIND.NS",
    "Procter & Gamble Hygiene and Health Care Limited": "PGHH.NS",
    "3M India Limited": "3MINDIA.NS",
    "Atul Limited": "ATUL.NS",
    "SRF Limited": "SRF.NS",
    "Navin Fluorine International Limited": "NAVINFLUOR.NS",
    "Clean Science and Technology Limited": "CLEAN.NS",
    "Deepak Fertilizers and Petrochemicals Corporation Limited": "DEEPAKFERT.NS",
    "Gujarat Fluorochemicals Limited": "FLUOROCHEM.NS",
    "Apar Industries Limited": "APARINDS.NS",
    "Thermax Limited": "THERMAX.NS",
    "Cummins India Limited": "CUMMINSIND.NS",
    "Kirloskar Brothers Limited": "KIRLOSBROS.NS",
    "Elgi Equipments Limited": "ELGIEQUIP.NS",
    "Triveni Turbine Limited": "TRITURBINE.NS",
    "KEC International Limited": "KEC.NS",
    "Kalpataru Projects International Limited": "KPIL.NS",
    "Ahluwalia Contracts (India) Limited": "AHLUCONT.NS",
    "NCC Limited": "NCC.NS",
    "PNC Infratech Limited": "PNCINFRA.NS",
    "IRB Infrastructure Developers Limited": "IRB.NS",
    "Zee Entertainment Enterprises Limited": "ZEEL.NS",
    "PVR INOX Limited": "PVRINOX.NS",
    "Sun TV Network Limited": "SUNTV.NS",
    "Saregama India Limited": "SAREGAMA.NS",
    "Nazara Technologies Limited": "NAZARA.NS",
    "Happiest Minds Technologies Limited": "HAPPSTMNDS.NS",
    "KPIT Technologies Limited": "KPITTECH.NS",
    "Zensar Technologies Limited": "ZENSARTECH.NS",
    "Cyient Limited": "CYIENT.NS",
    "Tata Technologies Limited": "TATATECH.NS",
    "Kaynes Technology India Limited": "KAYNES.NS",
    "Dixon Technologies (India) Limited": "DIXON.NS",
    "Amber Enterprises India Limited": "AMBER.NS",
    "Syrma SGS Technology Limited": "SYRMA.NS",
    "Avalon Technologies Limited": "AVALON.NS",
    "Campus Activewear Limited": "CAMPUS.NS",
    "Go Fashion (India) Limited": "GOCOLORS.NS",
    "Ethos Limited": "ETHOSLTD.NS",
    "Kalyan Jewellers India Limited": "KALYANKJIL.NS",
    "Titan Company Limited": "TITAN.NS",
    "Senco Gold Limited": "SENCO.NS",
    "CarTrade Tech Limited": "CARTRADE.NS",
    "Nykaa Limited": "NYKAA.NS",
    "Paytm Limited": "PAYTM.NS",
    "PB Fintech Limited": "POLICYBZR.NS",
    "Fino Payments Bank Limited": "FINOPB.NS",
    "RateGain Travel Technologies Limited": "RATEGAIN.NS",
    "EaseMyTrip Limited": "EASEMYTRIP.NS",
    "Affle (India) Limited": "AFFLE.NS",
    "IndiaMart InterMesh Limited": "INDIAMART.NS",
    "Just Dial Limited": "JUSTDIAL.NS",
    "Tanla Platforms Limited": "TANLA.NS",
    "Route Mobile Limited": "ROUTE.NS",
    "Gland Pharma Limited": "GLAND.NS",
    "Syngene International Limited": "SYNGENE.NS",
    "JB Chemicals & Pharmaceuticals Limited": "JBCHEPHARM.NS",
    "Ajanta Pharma Limited": "AJANTPHARM.NS",
    "Aarti Industries Limited": "AARTIIND.NS",
    "Balrampur Chini Mills Limited": "BALRAMCHIN.NS",
    "Dhampur Sugar Mills Limited": "DHAMPURSUG.NS",
    "Triveni Engineering & Industries Limited": "TRIVENI.NS",
    "EID Parry India Limited": "EIDPARRY.NS",
    "Castrol India Limited": "CASTROLIND.NS",
    "Gujarat State Petronet Limited": "GSPL.NS",
    "Oil India Limited": "OIL.NS",
    "Coal India Limited": "COALINDIA.NS",
    "NMDC Limited": "NMDC.NS",
    "Hindustan Zinc Limited": "HINDZINC.NS",
    "National Aluminium Company Limited": "NATIONALUM.NS",
    "MOIL Limited": "MOIL.NS",
    "Mishra Dhatu Nigam Limited": "MIDHANI.NS",
    "Rashtriya Chemicals and Fertilizers Limited": "RCF.NS",
    "Gujarat Narmada Valley Fertilizers & Chemicals Limited": "GNFC.NS",
    "Fertilisers And Chemicals Travancore Limited": "FACT.NS",
    "Bharat Heavy Electricals Limited": "BHEL.NS",
    "Engineers India Limited": "ENGINERSIN.NS",
    "RITES Limited": "RITES.NS",
    "IRCON International Limited": "IRCON.NS",
    "Rail Vikas Nigam Limited": "RVNL.NS",
    "Jupiter Wagons Limited": "JWL.NS",
    "Titagarh Rail Systems Limited": "TITAGARH.NS",
    "Texmaco Rail & Engineering Limited": "TEXRAIL.NS",
}

# Merge extended map into TICKER_MAP
TICKER_MAP.update(EXTENDED_TICKER_MAP)


def build_stock_universe_from_csv(csv_path=SHAREHOLDING_CSV_PATH):
    """Build STOCK_UNIVERSE dynamically from the shareholding CSV."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        companies = df['COMPANY'].dropna().unique().tolist()
        universe = {}
        for company in companies:
            company_clean = company.strip()
            if company_clean in TICKER_MAP:
                # Use known mapping first
                if TICKER_MAP[company_clean] not in universe.values():
                    universe[company_clean] = TICKER_MAP[company_clean]
        return universe
    except FileNotFoundError:
        return TICKER_MAP


STOCK_UNIVERSE = build_stock_universe_from_csv()


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():
    """Get Groq client using API key from secrets."""
    try:
        api_key = st.secrets['REST']['GROQ_API_KEY']
        return Groq(api_key=api_key)
    except Exception:
        return None


# ============================================================
# TECHNICAL INDICATORS (Better features for the model)
# ============================================================

def compute_rsi(series, period=14):
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    """Compute MACD line and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_features_for_index(hist, i):
    """Compute all technical features at a given index in the history DataFrame."""
    close = hist['Close']
    volume = hist['Volume']
    high = hist['High'] if 'High' in hist.columns else close
    low = hist['Low'] if 'Low' in hist.columns else close

    # Basic returns
    return_1d = (close.iloc[i] / close.iloc[i - 1] - 1) if i >= 1 else 0
    return_5d = (close.iloc[i] / close.iloc[i - 5] - 1) if i >= 5 else 0
    return_10d = (close.iloc[i] / close.iloc[i - 10] - 1) if i >= 10 else 0
    return_20d = (close.iloc[i] / close.iloc[i - 20] - 1) if i >= 20 else 0

    # Moving averages
    sma_10 = close.iloc[max(0, i - 10):i + 1].mean()
    sma_20 = close.iloc[max(0, i - 20):i + 1].mean()
    sma_50 = close.iloc[max(0, i - 50):i + 1].mean()
    price = close.iloc[i]

    price_vs_sma10 = (price / sma_10 - 1) if sma_10 > 0 else 0
    price_vs_sma20 = (price / sma_20 - 1) if sma_20 > 0 else 0
    price_vs_sma50 = (price / sma_50 - 1) if sma_50 > 0 else 0

    # SMA crossover signals
    sma10_vs_sma20 = (sma_10 / sma_20 - 1) if sma_20 > 0 else 0
    sma20_vs_sma50 = (sma_20 / sma_50 - 1) if sma_50 > 0 else 0

    # Volatility
    returns_window = close.iloc[max(0, i - 20):i + 1].pct_change().dropna()
    volatility_20d = returns_window.std() if len(returns_window) > 1 else 0

    # Volume signal
    vol_recent = volume.iloc[max(0, i - 5):i + 1].mean()
    vol_avg = volume.iloc[max(0, i - 20):i + 1].mean()
    volume_ratio = (vol_recent / vol_avg) if vol_avg > 0 else 1

    # RSI
    rsi_series = compute_rsi(close)
    rsi = rsi_series.iloc[i] if i < len(rsi_series) and not np.isnan(rsi_series.iloc[i]) else 50

    # MACD
    macd_line, signal_line = compute_macd(close)
    macd_val = macd_line.iloc[i] if i < len(macd_line) else 0
    macd_signal = signal_line.iloc[i] if i < len(signal_line) else 0
    macd_hist = macd_val - macd_signal

    # Momentum (rate of change)
    momentum_10d = (close.iloc[i] / close.iloc[i - 10] - 1) if i >= 10 else 0

    # Average True Range (volatility measure)
    if i >= 14:
        tr_list = []
        for j in range(max(1, i - 13), i + 1):
            tr = max(
                high.iloc[j] - low.iloc[j],
                abs(high.iloc[j] - close.iloc[j - 1]),
                abs(low.iloc[j] - close.iloc[j - 1])
            )
            tr_list.append(tr)
        atr = np.mean(tr_list) / price if price > 0 else 0
    else:
        atr = 0

    # Bollinger Band position (where is price relative to bands)
    if len(returns_window) > 1:
        bb_mid = sma_20
        bb_std = close.iloc[max(0, i - 20):i + 1].std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_position = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    else:
        bb_position = 0.5

    # Consecutive up/down days
    consecutive = 0
    for j in range(i, max(0, i - 5), -1):
        if j >= 1 and close.iloc[j] > close.iloc[j - 1]:
            consecutive += 1
        elif j >= 1:
            consecutive -= 1
        else:
            break

    return {
        'return_1d': return_1d,
        'return_5d': return_5d,
        'return_10d': return_10d,
        'return_20d': return_20d,
        'price_vs_sma10': price_vs_sma10,
        'price_vs_sma20': price_vs_sma20,
        'price_vs_sma50': price_vs_sma50,
        'sma10_vs_sma20': sma10_vs_sma20,
        'sma20_vs_sma50': sma20_vs_sma50,
        'volatility_20d': volatility_20d,
        'volume_ratio': volume_ratio,
        'rsi': rsi,
        'macd_hist': macd_hist,
        'momentum_10d': momentum_10d,
        'atr': atr,
        'bb_position': bb_position,
        'consecutive_days': consecutive,
    }


FEATURE_COLS = [
    'institutional_score', 'holding_weight',
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
    'sma10_vs_sma20', 'sma20_vs_sma50',
    'volatility_20d', 'volume_ratio', 'rsi', 'macd_hist',
    'momentum_10d', 'atr', 'bb_position', 'consecutive_days'
]

# Pretty names for display
FEATURE_DISPLAY_NAMES = {
    'institutional_score': 'Institutional Score',
    'holding_weight': 'Holding Weight',
    'return_1d': '1 Day Return',
    'return_5d': '5 Day Return',
    'return_10d': '10 Day Return',
    'return_20d': '20 Day Return',
    'price_vs_sma10': 'Price vs SMA 10',
    'price_vs_sma20': 'Price vs SMA 20',
    'price_vs_sma50': 'Price vs SMA 50',
    'sma10_vs_sma20': 'SMA 10 vs SMA 20',
    'sma20_vs_sma50': 'SMA 20 vs SMA 50',
    'volatility_20d': '20 Day Volatility',
    'volume_ratio': 'Volume Ratio',
    'rsi': 'RSI',
    'macd_hist': 'MACD Histogram',
    'momentum_10d': '10 Day Momentum',
    'atr': 'Avg True Range',
    'bb_position': 'Bollinger Band Position',
    'consecutive_days': 'Consecutive Up/Down Days',
}


# ============================================================
# DATA LOADING
# ============================================================

def load_shareholding_data(csv_path=SHAREHOLDING_CSV_PATH):
    """Load SEBI shareholding pattern CSV."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        return df
    except FileNotFoundError:
        st.warning("Shareholding CSV not found.")
        return None


def get_institutional_score(company_name, shareholding_df):
    """Calculate institutional score from shareholding data."""
    if shareholding_df is None:
        return 0.5, 0.5

    match = shareholding_df[
        shareholding_df['COMPANY'].str.contains(company_name.split(' ')[0], case=False, na=False)
    ]
    if match.empty:
        return 0.5, 0.5

    row = match.iloc[0]
    promoter_pct = float(row.get('PROMOTER & PROMOTER GROUP (A)', 50))
    return promoter_pct / 100.0, promoter_pct / 100.0


# ============================================================
# NEWS LAYER
# ============================================================

def fetch_all_market_headlines():
    """Fetch all market headlines once (shared across stocks)."""
    try:
        resp = requests.get(
            "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, features='xml')
        items = soup.findAll('item')
        return [item.find('title').get_text().strip() for item in items[:30]]
    except Exception:
        return []


def get_stock_headlines(company_name, ticker, all_headlines):
    """Filter headlines relevant to a specific stock."""
    keywords = company_name.lower().split()[:2]
    ticker_clean = ticker.replace('.NS', '').replace('.BO', '').lower()

    relevant = [h for h in all_headlines
                if any(kw in h.lower() for kw in keywords) or ticker_clean in h.lower()]
    return relevant[:5] if relevant else all_headlines[:3]


def get_news_scores_batch(all_headlines, stock_list):
    """Get news scores for all stocks in one Groq API call."""
    client = get_groq_client()
    if not client or not all_headlines:
        return {name: 0.0 for name, _ in stock_list}

    stock_names = [name for name, _ in stock_list]
    try:
        prompt = f"""Analyze these Indian stock market headlines and rate sentiment for each stock listed below.

Headlines:
{chr(10).join(f'- {h}' for h in all_headlines[:20])}

Stocks to rate:
{chr(10).join(f'- {name}' for name in stock_names)}

Return ONLY a valid JSON object mapping each stock name to a sentiment score between -1.0 (very negative) and 1.0 (very positive). 0.0 means neutral.
Example: {{"Reliance Industries Limited": 0.5, "Infosys Limited": -0.2}}
No markdown, no explanation, just the JSON."""

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        scores = json.loads(text)
        return {name: float(scores.get(name, 0.0)) for name in stock_names}
    except Exception:
        return {name: 0.0 for name, _ in stock_list}


# ============================================================
# TRAINING: Generate data + train model + fetch live data
# ============================================================

def generate_training_data(shareholding_df, progress_bar=None):
    """Generate training data with technical indicators. Target: up/down in 5 days (with threshold)."""
    training_rows = []
    total = len(STOCK_UNIVERSE)

    for idx, (company, ticker) in enumerate(STOCK_UNIVERSE.items()):
        if progress_bar:
            progress_bar.progress((idx + 1) / total, text=f"Fetching {company}...")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")
            if hist.empty or len(hist) < 60:
                continue

            hist = hist.reset_index()
            # Target: 1 if price goes up >1% in 5 days, 0 if down >1%
            # Skip ambiguous samples near 0% to give clearer signal
            hist['future_return'] = hist['Close'].shift(-FORECAST_DAYS) / hist['Close'] - 1
            hist = hist.dropna(subset=['future_return'])
            hist['target'] = -1  # -1 = skip
            hist.loc[hist['future_return'] > 0.01, 'target'] = 1   # up > 1%
            hist.loc[hist['future_return'] < -0.01, 'target'] = 0  # down > 1%
            hist = hist[hist['target'] >= 0]  # drop ambiguous rows

            inst_score, holding_weight = get_institutional_score(company, shareholding_df)

            for i in range(50, len(hist) - FORECAST_DAYS, 3):
                actual_idx = hist.index[i] if i < len(hist) else None
                if actual_idx is None:
                    continue
                features = compute_features_for_index(hist, i)
                features['institutional_score'] = inst_score
                features['holding_weight'] = holding_weight
                features['target'] = int(hist['target'].iloc[i])
                training_rows.append(features)
        except Exception:
            continue

    return pd.DataFrame(training_rows)


def fetch_and_cache_live_data(shareholding_df, news_scores, progress_bar=None):
    """Fetch all live data during training and cache it for fast recommendations."""
    live_data = []
    total = len(STOCK_UNIVERSE)

    for idx, (company, ticker) in enumerate(STOCK_UNIVERSE.items()):
        if progress_bar:
            progress_bar.progress((idx + 1) / total, text=f"Fetching live data for {company}...")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if hist.empty or len(hist) < 30:
                continue

            hist = hist.reset_index()
            i = len(hist) - 1
            features = compute_features_for_index(hist, i)

            inst_score, holding_weight = get_institutional_score(company, shareholding_df)
            features['institutional_score'] = inst_score
            features['holding_weight'] = holding_weight
            features['news_score'] = news_scores.get(company, 0.0)  # used for post-prediction boost, not model input
            features['company'] = company
            features['ticker'] = ticker

            # Get current price robustly
            last_close = hist['Close'].iloc[-1]
            if last_close is not None and not pd.isna(last_close) and last_close > 0:
                features['current_price'] = round(float(last_close), 2)
            else:
                # Try from yfinance info as fallback
                try:
                    info = stock.info
                    features['current_price'] = round(float(info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or 0), 2)
                except Exception:
                    features['current_price'] = 0.0

            live_data.append(features)
        except Exception:
            continue

    # Save to disk
    os.makedirs("models", exist_ok=True)
    with open(LIVE_DATA_PATH, 'wb') as f:
        pickle.dump(live_data, f)

    return live_data


def train_random_forest(training_df):
    """Train an ensemble of models and pick the best, or stack them."""
    from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier

    X = training_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = training_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=15,
        min_samples_leaf=8, max_features='sqrt',
        random_state=42, n_jobs=-1, class_weight='balanced'
    )

    # Model 2: Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_split=20, min_samples_leaf=10,
        random_state=42
    )

    # Model 3: Extra Trees (more randomness = better generalization)
    et = ExtraTreesClassifier(
        n_estimators=500, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt',
        random_state=42, n_jobs=-1, class_weight='balanced'
    )

    # Soft voting ensemble — averages probabilities from all 3
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('et', et)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, ensemble.predict(X_train))
    test_acc = accuracy_score(y_test, ensemble.predict(X_test))

    return ensemble, train_acc, test_acc


def save_model(model):
    """Save trained model to disk."""
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': FEATURE_COLS}, f)


def load_model():
    """Load trained model from disk."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['feature_cols']
    except FileNotFoundError:
        return None, None


def load_live_data():
    """Load cached live data."""
    try:
        with open(LIVE_DATA_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


# ============================================================
# PREDICTION + RANKING
# ============================================================

def run_predictions(model, feature_cols, live_data):
    """Run predictions on cached live data, then apply news sentiment as a ranking boost."""
    results = []
    for row in live_data:
        features = np.array([[row.get(col, 0) for col in feature_cols]])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Model confidence (probability of going UP)
        prob_up = model.predict_proba(features)[0][1]

        # Post-prediction news boost: shift confidence by up to ±10%
        news_score = row.get('news_score', 0.0)
        news_boost = news_score * 0.10  # max ±0.10 shift
        final_score = np.clip(prob_up + news_boost, 0.0, 1.0)

        results.append({
            'company': row['company'],
            'ticker': row['ticker'],
            'model_confidence': prob_up,
            'news_score': news_score,
            'confidence': final_score,  # model + news combined
            'current_price': row['current_price'],
            'rsi': row.get('rsi', 50),
            'momentum_10d': row.get('momentum_10d', 0),
            'return_5d': row.get('return_5d', 0),
            'return_20d': row.get('return_20d', 0),
            'volatility_20d': row.get('volatility_20d', 0),
            'institutional_score': row.get('institutional_score', 0.5),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('confidence', ascending=False)
    return results_df


# ============================================================
# GROQ EXPLANATION LAYER
# ============================================================

def generate_stock_reason(row):
    """Generate a short reason for why a stock is recommended."""
    parts = []
    if row['confidence'] > 0.65:
        parts.append("Strong upward signal")
    elif row['confidence'] > 0.55:
        parts.append("Moderate upward signal")

    if row['momentum_10d'] > 0.03:
        parts.append("positive momentum")
    elif row['momentum_10d'] < -0.03:
        parts.append("recovering from dip")

    if row['rsi'] < 30:
        parts.append("oversold (RSI low)")
    elif row['rsi'] > 70:
        parts.append("strong demand (RSI high)")

    if row['news_score'] > 0.3:
        parts.append("positive news sentiment")
    elif row['news_score'] < -0.3:
        parts.append("negative news — watch closely")

    if row['institutional_score'] > 0.6:
        parts.append("strong promoter backing")

    return ". ".join(parts) if parts else "Balanced signals across indicators"


def generate_explanation_groq(top_stocks_df):
    """Use Groq (Llama) to explain why these stocks are recommended."""
    client = get_groq_client()
    if not client:
        return "⚠️ Groq API key not configured. Add it to `.streamlit/secrets.toml` under `[REST]`."

    stocks_info = ""
    for _, row in top_stocks_df.iterrows():
        cp = row.get('current_price', 0)
        price_str = f"₹{cp:,.2f}" if cp and cp > 0 else "N/A"
        stocks_info += f"""
- {row['company']} ({row['ticker']}):
  Price: {price_str}
  Model Confidence (up): {row['confidence']*100:.1f}%
  10-day Momentum: {row['momentum_10d']*100:.2f}%
  RSI: {row['rsi']:.1f}
  News Score: {row['news_score']:.2f}
  Promoter Holding: {row['institutional_score']*100:.0f}%
"""

    prompt = f"""You are a financial analyst. Explain why these top {len(top_stocks_df)} Indian stocks are recommended for the next 5 trading days.

Data:
{stocks_info}

For each stock give 2-3 sentences covering:
1. Key momentum/technical signals
2. Promoter confidence
3. News sentiment
4. One risk to watch

Keep it professional and concise. Use the stock name as a header.
End with a brief market outlook.
Add disclaimer: This is not financial advice. Do your own research."""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Could not generate explanation: {e}"


# ============================================================
# STREAMLIT UI
# ============================================================

def stock_recommender_page():
    st.markdown("""
    <style>
        .profile-header h1 { color: #556b3b; font-size: 60px; }
        .rec-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px; padding: 1.5rem; margin: 1rem 0;
            border: 1px solid #0f3460;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="profile-header">
        <h1 style="text-align:center;">🎯 AI Stock Recommender</h1>
        <p style="text-align:center;">ML-powered recommendations using Random Forest, Technical Indicators, News Sentiment & SEBI Data</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    tab1, tab2 = st.tabs(["📊 Get Recommendations", "🔧 Train / Retrain Model"])

    # ---- TRAIN TAB ----
    with tab2:
        st.subheader("Model Training")
        st.write("Trains the model, fetches latest news sentiment, and pre-fetches all live stock data so recommendations are instant.")
        st.info(f"Stock universe: {len(STOCK_UNIVERSE)} stocks from your shareholding CSV.")

        if st.button("🚀 Train Model & Fetch Data", type="primary"):
            # Step 1: Load shareholding
            shareholding_df = load_shareholding_data()

            # Step 2: Fetch news and compute scores
            with st.spinner("Fetching latest market news..."):
                all_headlines = fetch_all_market_headlines()
                stock_list = list(STOCK_UNIVERSE.items())
                news_scores = get_news_scores_batch(all_headlines, stock_list)
            st.success(f"✅ Fetched {len(all_headlines)} headlines. Scored {len(news_scores)} stocks.")

            # Step 3: Generate training data
            progress1 = st.progress(0, text="Generating training data...")
            training_df = generate_training_data(shareholding_df, progress1)
            progress1.empty()

            if training_df.empty:
                st.error("Could not generate training data. Check internet connection.")
                st.stop()

            st.success(f"✅ Generated {len(training_df)} training samples. (Target: up/down in {FORECAST_DAYS} days)")

            # Step 4: Train model
            with st.spinner("Training Random Forest classifier..."):
                model, train_acc, test_acc = train_random_forest(training_df)
                save_model(model)

            col1, col2 = st.columns(2)
            col1.metric("Train Accuracy", f"{train_acc*100:.1f}%")
            col2.metric("Test Accuracy", f"{test_acc*100:.1f}%")

            st.success("✅ Model trained and saved!")

            # Step 5: Feature importance (from Random Forest inside ensemble)
            st.subheader("Feature Importance")
            try:
                # Try ensemble's RF estimator first
                if hasattr(model, 'estimators_'):
                    importances = model.estimators_[0].feature_importances_
                elif hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    importances = None

                if importances is not None:
                    importance_df = pd.DataFrame({
                        'Feature': [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLS],
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    st.bar_chart(importance_df.set_index('Feature'))
            except Exception:
                st.info("Feature importance not available for this model type.")

            # Step 6: Pre-fetch live data
            progress2 = st.progress(0, text="Pre-fetching live stock data...")
            fetch_and_cache_live_data(shareholding_df, news_scores, progress2)
            progress2.empty()

            st.success("✅ Live data cached. Recommendations will be instant!")

    # ---- RECOMMENDATIONS TAB ----
    with tab1:
        st.subheader("Live Stock Recommendations")

        model, feature_cols = load_model()
        if model is None:
            st.warning("⚠️ No trained model found. Go to 'Train / Retrain Model' tab first.")
            st.stop()

        live_data = load_live_data()
        if live_data is None:
            st.warning("⚠️ No cached live data. Go to 'Train / Retrain Model' tab to fetch data.")
            st.stop()

        st.success(f"✅ Model loaded. {len(live_data)} stocks ready.")

        if st.button("🔍 Generate Recommendations", type="primary"):
            results_df = run_predictions(model, feature_cols, live_data)

            if results_df.empty:
                st.error("No predictions generated.")
                st.stop()

            top_5 = results_df.head(TOP_N)

            st.markdown("---")
            st.subheader(f"🏆 Top {TOP_N} Recommended Stocks")

            for rank, (_, row) in enumerate(top_5.iterrows(), 1):
                with st.container():
                    col1, col2, col3, col4 = st.columns([0.5, 2, 1.2, 2.5])

                    with col1:
                        st.markdown(f"### #{rank}")

                    with col2:
                        st.markdown(f"**{row['company']}**")
                        cp = row.get('current_price', 0)
                        price_display = f"₹{cp:,.2f}" if cp and cp > 0 else "Price unavailable"
                        st.caption(f"{row['ticker']} | {price_display}")

                    with col3:
                        conf_pct = row['confidence'] * 100
                        st.metric("Confidence", f"{conf_pct:.0f}%",
                                  delta=f"{'Bullish' if conf_pct > 55 else 'Neutral'}")

                    with col4:
                        reason = generate_stock_reason(row)
                        st.caption(f"💡 {reason}")

                    st.markdown("---")

            # Simplified "View All" table
            with st.expander("📋 View All Stock Scores"):
                display_df = results_df[['company', 'ticker', 'confidence', 'current_price']].copy()
                display_df['confidence'] = (display_df['confidence'] * 100).round(1)
                display_df['current_price'] = display_df['current_price'].fillna(0).round(2)
                display_df.columns = ['Company', 'Ticker', 'Confidence %', 'Price (₹)']
                display_df = display_df.reset_index(drop=True)
                display_df.index = display_df.index + 1
                display_df.index.name = 'Rank'
                st.dataframe(display_df, use_container_width=True)

            # Groq Explanation
            st.markdown("---")
            st.subheader("🤖 AI Analysis")

            with st.spinner("Generating explanation..."):
                explanation = generate_explanation_groq(top_5)

            st.markdown(explanation)

            st.session_state['last_recommendations'] = results_df
            st.session_state['last_top5'] = top_5

    # Disclaimer at bottom of page (always visible)
    st.markdown("---")
    st.caption("⚠️ Disclaimer: This is not financial advice. Do your own research.")


stock_recommender_page()
