#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    Fetch and normalize company profile data from FMP's stable profile endpoint.

    Returns a dict with:
        - ticker
        - exchange
        - country
        - industry
        - marketCap

    Raises ValueError if data is missing or malformed.
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
    }


# In[3]:


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


# In[5]:


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
    Returns etf_map['DEFAULT'] if industry not found.
    """
    return etf_map.get(industry, etf_map.get("DEFAULT", "SPY"))


# In[7]:


# file: proxy_builder.py

def build_proxy_for_ticker(
    ticker: str,
    exchange_map: dict,
    industry_map: dict
) -> dict:
    """
    Builds the full stock_factor_proxies entry for a single ticker.
    Includes market, momentum, value, industry, and placeholder subindustry.
    """
    try:
        profile = fetch_profile(ticker)
    except Exception as e:
        print(f"⚠️ {ticker}: profile fetch failed — {e}")
        return None

    proxies = {}

    # Add exchange-based factors
    proxies.update(map_exchange_proxies(profile.get("exchange", ""), exchange_map))

    # Add industry ETF
    industry = profile.get("industry")
    if industry:
        proxies["industry"] = map_industry_etf(industry, industry_map)
    else:
        proxies["industry"] = industry_map.get("DEFAULT", "SPY")

    # Placeholder for subindustry
    proxies["subindustry"] = []

    return proxies


# In[9]:


# file: proxy_builder.py

import yaml
from pathlib import Path

def inject_proxies_into_portfolio_yaml(path: str = "portfolio.yaml") -> None:
    """
    Loads portfolio.yaml, builds stock_factor_proxies for each ticker in portfolio_input,
    and writes updated file in-place.
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


# In[11]:


# file: proxy_builder.py

import os
from typing import List
from dotenv import load_dotenv
import openai
import traceback

# ── Load env & set up the shared client ───────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ── Peer-generator helper ─────────────────────────────────────────────
def generate_subindustry_peers(
    ticker: str,
    name: str,
    industry: str,
    model: str = "gpt-4.1",    # any available chat model
    max_tokens: int = 200,
    temperature: float = 0.2,
) -> List[str]:
    """
    Ask GPT for 5-10 peer tickers in the same sub-industry.

    Returns a **list of tickers** (strings).  
    Empty list → either parse failure or model couldn’t comply.
    """
    prompt = f"""
You’re a fundamental equity analyst.
Given the following stock details, return 5–10 peer tickers that best
represent its subindustry or closest competitive group — ideally companies
that compete with or operate in similar business models.

Only include **currently publicly listed** equities from the U.S., Canada,
or U.K.  

Do not include companies that have been acquired, merged, or
delisted.  

Return a clean list of tickers, no explanation.

⸻

Example Input:

Ticker: NVDA
Name: NVIDIA Corporation
Industry: Semiconductors

⸻

Expected Output:

["AMD", "INTC", "AVGO", "QCOM", "TSM", "MRVL", "TXN"]
⸻

Do for this:

Ticker: {ticker}
Name: {name}
Industry: {industry}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = resp.choices[0].message.content.strip()

        # Expect something like ["AMD", "INTC", "QCOM", ...]
        return content 

    except Exception as e:
        # Log full traceback so the root cause is visible
        print(f"⚠️ generate_subindustry_peers failed for {ticker}: {e}")
        traceback.print_exc()
        return []



# In[12]:


# file: proxy_builder.py

from typing import List

def filter_valid_tickers(cands: List[str]) -> List[str]:
    """
    Keep only symbols for which `fetch_profile` succeeds.
    Falls through silently on any API / parsing error.
    """
    good = []
    for sym in cands:
        try:
            _ = fetch_profile(sym)        # raises on 4xx/empty payload
            good.append(sym.upper())
        except Exception:
            pass                          # just skip the dud symbol
    return good


# In[13]:


# file: proxy_builder.py

import ast

def get_subindustry_peers_from_ticker(ticker: str) -> list[str]:
    """
    Fetches company profile, sends details to GPT, and returns a list of parsed peer tickers.

    If GPT response is malformed or empty, returns [].
    """
    try:
        profile = fetch_profile(ticker)
        name = profile.get("companyName") or profile.get("name") or ticker
        industry = profile.get("industry", "Unknown")

        raw_peers_text = generate_subindustry_peers(ticker=ticker, name=name, industry=industry)
        print(f"\nGPT peers for {ticker}:{raw_peers_text}")

        peer_list = ast.literal_eval(raw_peers_text)

        if not isinstance(peer_list, list):
            raise ValueError("Parsed object is not a list")

        return filter_valid_tickers(peer_list)

    except Exception as e:
        print(f"⚠️ {ticker}: failed to generate peers — {e}")
        return []


# In[15]:


# file: proxy_builder.py

import yaml
from pathlib import Path

def inject_subindustry_peers_into_yaml(
    yaml_path: str = "portfolio.yaml",
    tickers: list[str] = None
) -> None:
    """
    For each ticker in portfolio_input (or provided list), generates subindustry peers via GPT
    and writes them into the 'stock_factor_proxies' section under 'subindustry'.

    Overwrites the existing YAML file.
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




