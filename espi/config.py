"""Central configuration for espi-event-analysis."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "espi-event-analysis"
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PRICES_DIR = DATA_DIR / "prices"
REPORTS_PATH = DATA_DIR / "reports.parquet"
EVENTS_PATH = DATA_DIR / "events.parquet"
UNIVERSE_PATH = ROOT_DIR / "universe.csv"
RESULTS_DIR = ROOT_DIR / "results"

# Market benchmark
MARKET_SYMBOL = os.getenv("MARKET_SYMBOL", "wig")

# LLM Inference Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning")
MAX_BODY_CHARS = int(os.getenv("MAX_BODY_CHARS", "6000"))
