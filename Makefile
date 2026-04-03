.PHONY: ui live paper analyze test setup backtest lint

setup:
	pip install -e ".[dev]"
	python -c "import asyncio; from db.database import DatabaseManager; asyncio.run(DatabaseManager().init_db())"

ui:
	KABUAI_ALLOW_LIVE_TRADING=false streamlit run ui/app.py --server.port 8501

live:
	KABUAI_ALLOW_LIVE_TRADING=false python run_live_trade.py

paper:
	KABUAI_ALLOW_LIVE_TRADING=false python run_paper.py

analyze:
	KABUAI_ALLOW_LIVE_TRADING=false python main.py --mode analyze --date $(shell date +%Y-%m-%d)

test:
	pytest tests/ -v

backtest:
	KABUAI_ALLOW_LIVE_TRADING=false python main.py --mode backtest

lint:
	ruff check .
	ruff format --check .
