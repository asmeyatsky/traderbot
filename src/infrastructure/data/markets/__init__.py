"""
Market Data Registry

Provides access to static market and stock data from JSON files.
Lazy-loads and caches data on first access for each market.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

_DATA_DIR = Path(__file__).parent
_cache: dict[str, dict] = {}

MARKET_FILES = [
    "us_nyse.json",
    "us_nasdaq.json",
    "uk_lse.json",
    "eu_euronext.json",
    "de_xetra.json",
    "jp_tse.json",
    "hk_hkex.json",
]


def _load_market(filename: str) -> dict:
    """Load and cache a single market JSON file."""
    if filename not in _cache:
        with open(_DATA_DIR / filename, "r", encoding="utf-8") as f:
            _cache[filename] = json.load(f)
    return _cache[filename]


def _load_all() -> list[dict]:
    """Load all market files."""
    return [_load_market(f) for f in MARKET_FILES]


def get_all_markets() -> list[dict]:
    """Return metadata for all available markets (no stock lists)."""
    return [
        {
            "market_code": m["market_code"],
            "market_name": m["market_name"],
            "country": m["country"],
            "currency": m["currency"],
        }
        for m in _load_all()
    ]


def get_markets_for_user(allowed_markets: List[str]) -> list[dict]:
    """Return market metadata filtered by the user's allowed_markets list."""
    return [
        m for m in get_all_markets()
        if m["market_code"] in allowed_markets
    ]


def get_stocks_by_market(market_code: str, search: Optional[str] = None) -> Optional[dict]:
    """
    Return stocks for a given market_code, optionally filtered by search term.

    Returns dict with market_code and filtered stocks list, or None if market not found.
    The search term matches against symbol and company name (case-insensitive).
    """
    for m in _load_all():
        if m["market_code"] == market_code:
            stocks = m["stocks"]
            if search:
                query = search.upper()
                stocks = [
                    s for s in stocks
                    if query in s["symbol"].upper() or query in s["name"].upper()
                ]
            return {
                "market_code": m["market_code"],
                "stocks": stocks,
            }
    return None
