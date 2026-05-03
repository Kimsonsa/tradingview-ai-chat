@echo off
title TradeAI Assistant
cd /d "%~dp0"
start http://localhost:8502
streamlit run assistant.py --server.port 8502
