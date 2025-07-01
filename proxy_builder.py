#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gpt_helpers import generate_subindustry_peers


# In[ ]:


# file: proxy_builder.py

import requests
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/stable"

def fetch_profile(ticker: str) -> dict:
    """
    Fetches normalized company profile metadata from Financial Modeling Prep (FMP)
    using the `/stable/profile` endpoint.

    Retrieves and parses the profile for a given ticker symbol, returning key
    fields needed for factor proxy mapping and classification (e.g., exchange, industry, ETF status).

    Parameters
    ----------
    ticker : str
        The stock symbol to query (e.g., "AAPL").

    Returns
    -------
    dict
        A dictionary with keys:
            - 'ticker'     : str  — confirmed symbol (e.g., "AAPL")
            - 'exchange'   : str  — primary listing exchange (e.g., "NASDAQ")
            - 'country'    : str  — country code (e.g., "US")
            - 'industry'   : str  — FMP-defined industry name (e.g., "Consumer Electronics")
            - 'marketCap'  : int  — latest market cap in USD
            - 'isEtf'      : bool — True if classified as an ETF
            - 'isFund'     : bool — True if classified as a mutual fund

    Raises
    ------
    ValueError
        If the API call fails or returns empty/malformed data.

    Notes
    -----
    • Used by proxy construction and GPT peer logic to determine asset type.
    • ETF and fund flags are useful for excluding non-operating entities from peer analysis.
    • Always returns the requested `ticker` if no symbol is present in payload.
    """
    url = f"{BASE_URL}/profile?symbol={ticker}&apikey={FMP_API_KEY}"
    resp = requests.get(url, timeout=10)
    if not resp.ok:
        raise ValueError(f"FMP API error: {resp.status_code} {resp.text}")

    data = resp.json()
    if not isinstance(data, list) or not data:
        raise ValueError(f"No profile data returned for {ticker}")

    profile = data[0]

    return {
        "ticker": profile.get("symbol", ticker),
        "exchange": profile.get("exchange"),              # e.g. "NASDAQ"
        "country": profile.get("country"),                # e.g. "US"
        "industry": profile.get("industry"),              # e.g. "Consumer Electronics"
        "marketCap": profile.get("marketCap"),            # e.g. 3T
        "isEtf": profile.get("isEtf", False),
        "isFund": profile.get("isFund", False),
    }


# In[ ]:


# file: proxy_builder.py

import yaml

def load_exchange_proxy_map(path: str = "exchange_etf_proxies.yaml") -> dict:
    """
    Load exchange-level ETF proxy mappings from a structured YAML file.
    Each exchange maps to:
      { market: ETF, momentum: ETF, value: ETF }
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def map_exchange_proxies(exchange: str, proxy_map: dict) -> dict:
    """
    Given an exchange name and loaded proxy_map, return:
        { market: ..., momentum: ..., value: ... }
    Falls back to proxy_map['DEFAULT'] if no match is found.
    """
    for key in proxy_map:
        if key != "DEFAULT" and key.lower() in exchange.lower():
            return proxy_map[key]
    return proxy_map.get("DEFAULT", {
        "market": "ACWX",
        "momentum": "IMTM",
        "value": "EFV"
    })


# In[ ]:


# file: proxy_builder.py

import yaml

def load_industry_etf_map(path: str = "industry_to_etf.yaml") -> dict:
    """
    Load industry → ETF mappings from a YAML file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def map_industry_etf(industry: str, etf_map: dict) -> str:
    """
    Map a given industry string to its corresponding ETF using the lookup map.

    Returns
    -------
    str or None
        Matching ETF ticker from the map. If industry is not found, returns None.

    Notes
    -----
    • No fallback to 'DEFAULT' — this is now handled at the call site,
      where fund/ETF detection can decide whether to skip the assignment.
    """
    return etf_map.get(industry)


# In[ ]:


# file: proxy_builder.py

def build_proxy_for_ticker(
    ticker: str,
    exchange_map: dict,
    industry_map: dict
) -> dict:
    """
    Constructs a stock_factor_proxies dictionary entry for a single ticker.

    This function:
      • Fetches the company profile from FMP using the provided ticker.
      • Maps the exchange to market/momentum/value ETFs using `exchange_map`.
      • If the stock is not an ETF or fund, maps the industry to an ETF using `industry_map`
        and initializes a placeholder subindustry list.
      • For ETFs or funds, sets both `industry` and `subindustry` to None or empty.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g., "AAPL").
    exchange_map : dict
        Dictionary loaded from `exchange_etf_proxies.yaml`, mapping exchange names
        to ETF proxies (keys: market, momentum, value).
    industry_map : dict
        Dictionary loaded from `industry_to_etf.yaml`, mapping industry names
        to representative ETFs.

    Returns
    -------
    dict
        A dictionary with the structure:
        {
            "market": str,
            "momentum": str,
            "value": str,
            "industry": Optional[str],
            "subindustry": list
        }

    Example
    -------
    {
        "market": "SPY",
        "momentum": "MTUM",
        "value": "IWD",
        "industry": "IGV",          # or empty if ETF
        "subindustry": []           # or empty if ETF
    }

    Notes
    -----
    • ETFs or funds are assigned themselves as their industry proxy 
      only if they are not already serving as the market proxy.
    • If the FMP profile fetch fails, the function returns None.
    • Unrecognized industries default to industry_map['DEFAULT'], if defined.
    """
    try:
        profile = fetch_profile(ticker)
    except Exception as e:
        print(f"⚠️ {ticker}: profile fetch failed — {e}")
        return None

    proxies = {}

    # Add exchange-based factors (always)
    proxies.update(map_exchange_proxies(profile.get("exchange", ""), exchange_map))

    # Assign industry to itself only if it's an ETF/fund AND not already used as market proxy
    # Else, add industry proxy ETF
    if profile.get("isEtf") or profile.get("isFund"):
        market_proxy = proxies.get("market", "").upper()
        if ticker.upper() != market_proxy:
            proxies["industry"] = ticker.upper()
        else:
            proxies["industry"] = ""
        proxies["subindustry"] = []
    else:
        industry = profile.get("industry")
        proxies["industry"] = map_industry_etf(industry, industry_map) if industry else ""
        proxies["subindustry"] = []

    return proxies


# In[ ]:


# file: proxy_builder.py

import yaml
from pathlib import Path

def inject_proxies_into_portfolio_yaml(path: str = "portfolio.yaml") -> None:
    """
    Populates the `stock_factor_proxies` section of a portfolio YAML file using exchange
    and industry mappings.

    For each ticker listed in `portfolio_input`, this function:
      - Retrieves the company profile via FMP.
      - Maps the exchange to market/momentum/value ETFs (via `exchange_etf_proxies.yaml`).
      - Maps the industry to an ETF (via `industry_to_etf.yaml`).
      - Adds a placeholder `subindustry: []` entry.
      - Stores results in the `stock_factor_proxies` block of the YAML.

    Parameters
    ----------
    path : str, optional
        Path to the portfolio YAML file. Defaults to "portfolio.yaml".

    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.
    ValueError
        If the YAML file does not contain a valid `portfolio_input` block.

    Side Effects
    ------------
    • Overwrites the YAML file in-place with updated `stock_factor_proxies`.
    • Prints the number of tickers updated.
    • Logs warnings for tickers that fail profile retrieval.

    Example
    -------
    >>> inject_proxies_into_portfolio_yaml("my_portfolio.yaml")
    ✅ Updated stock_factor_proxies for 4 tickers in my_portfolio.yaml
    """
    # Load YAML
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")

    with open(p, "r") as f:
        cfg = yaml.safe_load(f)

    tickers = list(cfg.get("portfolio_input", {}).keys())
    if not tickers:
        raise ValueError("portfolio_input is empty or missing")

    # Load reference maps
    exchange_map = load_exchange_proxy_map()
    industry_map = load_industry_etf_map()

    # Build proxies
    stock_proxies = {}
    for t in tickers:
        proxy = build_proxy_for_ticker(t, exchange_map, industry_map)
        if proxy:
            stock_proxies[t] = proxy
        else:
            print(f"⚠️ Skipping {t} due to missing profile")

    # Update and write back
    cfg["stock_factor_proxies"] = stock_proxies
    with open(p, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"✅ Updated stock_factor_proxies for {len(stock_proxies)} tickers in {path}")


# In[ ]:


# file: proxy_builder.py

from typing import List
import pandas as pd

from settings import PORTFOLIO_DEFAULTS          # <— central date window
from data_loader import fetch_monthly_close      # already cached disk-layer

def filter_valid_tickers(
    cands: List[str],
    target_ticker: str,
    start: pd.Timestamp | None = None,
    end:   pd.Timestamp | None = None,
) -> List[str]:
    """
    Return only those peer tickers that have *at least* as many monthly
    return observations as `target_ticker` over the same date window.

    Parameters
    ----------
    cands : list[str]
        Raw peer symbols (e.g. ['AAPL', 'XYZ', …]).
    target_ticker : str
        The stock we’re building the proxy for.  Its own data length sets
        the minimum observations all peers must match.
    start, end : pd.Timestamp | None
        Optional overrides for the analysis window.  Defaults fall back to
        PORTFOLIO_DEFAULTS.

    Returns
    -------
    list[str]
        Upper-cased symbols that satisfy the length requirement and loaded
        cleanly from `fetch_monthly_close`.
    """
    start = pd.to_datetime(start or PORTFOLIO_DEFAULTS["start_date"])
    end   = pd.to_datetime(end   or PORTFOLIO_DEFAULTS["end_date"])
    
    target_ticker = target_ticker.upper()
    target_obs = None

    # ▸ Observation count of the target
    target_prices  = fetch_monthly_close(target_ticker, start, end)
    target_obs  = len(target_prices)

    good: List[str] = []
    
    for sym in cands:
        try:
            prices  = fetch_monthly_close(sym, start, end)

            # Basic validation: ≥3 prices for returns calculation
            if not (isinstance(prices, pd.Series) and len(prices.dropna()) >= 3):
                continue

            # Enhanced validation: check observation count vs. target ticker
            if len(prices) >= target_obs:
                good.append(sym.upper())
                
        except Exception:
            # Any fetch failure (network, malformed payload, etc.) → skip
            continue

    return good


# In[ ]:


# file: proxy_builder.py

import ast

def get_subindustry_peers_from_ticker(ticker: str) -> list[str]:
    """
    Generates a cleaned list of subindustry peer tickers using GPT.

    This function:
    • Fetches company metadata from FMP using the ticker.
    • If the ticker is classified as an ETF or fund, returns an empty list immediately.
    • Otherwise, sends the company name and industry to the GPT API to generate 5–10 peer tickers.
    • Parses and validates the GPT response.
    • Filters the result to include only real, currently listed tickers (via `fetch_profile`).

    Parameters
    ----------
    ticker : str
        Stock symbol to generate peer group for.

    Returns
    -------
    list[str]
        Cleaned list of valid peer tickers (strings). Empty if parsing fails, the symbol is an ETF/fund,
        or if no valid peers are returned.

    Notes
    -----
    • Skips GPT call entirely for ETFs and mutual funds (via FMP profile check).
    • Uses `ast.literal_eval()` to safely parse GPT output.
    • Invalid responses (non-list, parse errors, delisted tickers) are silently skipped.
    • Calls `filter_valid_tickers()` to enforce only valid symbols from FMP.

    Side Effects
    ------------
    • Prints raw GPT output and any parse failures to stdout.
    • Logs skip message for ETFs/funds.
    """
    try:
        profile = fetch_profile(ticker)

        # Skip peer generation for ETFs and funds
        if profile.get("isEtf") or profile.get("isFund"):
            print(f"⏭️ Skipping GPT peers for {ticker} (ETF or fund)")
            return []
            
        name = profile.get("companyName") or profile.get("name") or ticker
        industry = profile.get("industry", "Unknown")

        raw_peers_text = generate_subindustry_peers(ticker=ticker, name=name, industry=industry)
        print(f"\nGPT peers for {ticker}:{raw_peers_text}")

        peer_list = ast.literal_eval(raw_peers_text)

        if not isinstance(peer_list, list):
            raise ValueError("Parsed object is not a list")

        return filter_valid_tickers(peer_list, target_ticker=ticker)

    except Exception as e:
        print(f"⚠️ {ticker}: failed to generate peers — {e}")
        return []


# In[ ]:


# file: proxy_builder.py

import yaml
from pathlib import Path

def inject_subindustry_peers_into_yaml(
    yaml_path: str = "portfolio.yaml",
    tickers: list[str] = None
) -> None:
    """
    Generates subindustry peer groups via GPT for tickers in a portfolio YAML file.

    This function updates the `stock_factor_proxies` section of the YAML file by adding
    a `subindustry` key for each ticker. It uses the GPT-based peer generation function
    (`get_subindustry_peers_from_ticker`) to generate peer lists based on company name
    and industry.

    Parameters
    ----------
    yaml_path : str, default "portfolio.yaml"
        Path to the portfolio YAML file. The file must contain a `portfolio_input` section.

    tickers : list[str], optional
        List of tickers to update. If None, updates all tickers in `portfolio_input`.

    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.

    Side Effects
    ------------
    • Overwrites the YAML file in place.
    • Adds or updates the `subindustry` field under `stock_factor_proxies` for each ticker.
    • Prints progress and peer count for each ticker to stdout.
    """
    # ── Load YAML
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"{yaml_path} not found")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    tickers_to_process = tickers or list(config.get("portfolio_input", {}).keys())
    stock_proxies = config.get("stock_factor_proxies", {})

    for tkr in tickers_to_process:
        peers = get_subindustry_peers_from_ticker(tkr)
        if tkr not in stock_proxies:
            stock_proxies[tkr] = {}
        stock_proxies[tkr]["subindustry"] = peers
        print(f"✅ {tkr} → {len(peers)} peers")

    config["stock_factor_proxies"] = stock_proxies

    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"\n✅ Finished writing subindustry peers to {yaml_path}")


# In[ ]:


# file: proxy_builder.py

import yaml
from pathlib import Path

def inject_all_proxies(
    yaml_path: str = "portfolio.yaml",
    use_gpt_subindustry: bool = False
) -> None:
    """
    Injects factor proxy mappings into a portfolio YAML file.

    For each ticker in `portfolio_input`, this function builds and injects:
      - Market, momentum, and value proxies based on exchange
      - Industry ETF proxy based on industry classification
      - (Optional) Subindustry peer list generated via GPT

    Parameters
    ----------
    yaml_path : str, default "portfolio.yaml"
        Path to the portfolio YAML file. The file must include a `portfolio_input` section.
    
    use_gpt_subindustry : bool, default False
        If True, sends company name + industry to GPT to generate subindustry peer tickers,
        and injects them into the `subindustry` field for each stock. Otherwise, the subindustry
        field is left empty (or populated from other means).

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    
    ValueError
        If the YAML file lacks a `portfolio_input` section.

    Side Effects
    ------------
    Overwrites the YAML file in place, populating the `stock_factor_proxies` section.
    Prints progress to stdout for each ticker processed.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"{yaml_path} not found")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    tickers = list(config.get("portfolio_input", {}).keys())
    if not tickers:
        raise ValueError("No portfolio_input found in YAML.")

    exchange_map = load_exchange_proxy_map()
    industry_map = load_industry_etf_map()
    stock_proxies = {}

    for tkr in tickers:
        proxy = build_proxy_for_ticker(tkr, exchange_map, industry_map)
        if proxy:
            stock_proxies[tkr] = proxy
        else:
            print(f"⚠️ Skipping {tkr} due to profile error.")

    config["stock_factor_proxies"] = stock_proxies

    # Optional: enrich with GPT subindustry peers
    if use_gpt_subindustry:
        from proxy_builder import get_subindustry_peers_from_ticker
        for tkr in tickers:
            peers = get_subindustry_peers_from_ticker(tkr)
            stock_proxies[tkr]["subindustry"] = peers
            print(f"✅ {tkr} → {len(peers)} GPT peers")

    # Save updated YAML
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"\n✅ All proxies injected into {yaml_path}")

