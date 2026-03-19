#!/bin/bash
cd "$(dirname "$0")"
export KABUAI_ALLOW_LIVE_TRADING=false
streamlit run ui/app.py --server.port 8501 --theme.base dark
