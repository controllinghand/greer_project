# Greer Project

A Python/SQL ETL and backtest suite for running Sean’s “Greer” scoring system on stock data.

## 🚀 Overview

- **fetch_financials.py** – pull fundamentals from SEC/EDGAR  
- **price_loader.py** – load daily OHLCV into your `prices` table  
- **greer_value_score.py** – calculate “Greer Value” per ticker  
- **greer_value_yield_score.py** – calculate dividend yield score  
- **greer_fair_value_gap.py** – detect Fair-Value-Gap direction  
- **greer_buyzone_calculator.py** – flag buy-zones  
- **greer_opportunity_periods.py** – build entry/exit periods  
- **backtest.py** – run your backtest rules  
- **run_all.py** – orchestrate them all, refresh materialized view, run backtest  

## 📦 Installation

1. Clone repo  
   ```bash
   git clone git@github.com:<you>/greer_project.git
   cd greer_project

