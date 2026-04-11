# 🚀 High-Convexity "Doubler" Stock Screener v2.0

Screens the **S&P 1500** for stocks with the technical DNA to deliver **100%+ returns** in 3–6 months.

## Features
- **5-factor scoring**: Trend (30%), Relative Strength (25%), Volume (20%), Volatility Quality (15%), Catalyst (10%)
- **Market regime detection**: Auto-adjusts scores in bull/bear markets
- **Ensemble validation**: 3 model variants, consensus flag
- **Sector caps**: Max 4 stocks per GICS sector
- **Exit signals**: ATR-based trailing stops, 50 SMA stops
- **Walk-forward backtest**: Test on any historical date

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hard Filters
- Price < 200 SMA → disqualified
- Price > 2× 200 SMA → disqualified (overextended)

## Disclaimer
Educational/research only. Not financial advice.
