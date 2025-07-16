#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ─── File: helpers_input.py ──────────────────────────────────
"""
Helpers for ingesting *what-if* portfolio changes.

parse_delta(...)
    • Accepts a YAML file path (optional) and/or an in-memory shift dict.
    • Returns a tuple: (delta_dict, new_weights_dict_or_None).

Precedence rules
----------------
1. If YAML contains `new_weights:` → treat as full replacement; shift_dict ignored.
2. Else, build a *delta* dict:     YAML `delta:` first, then merge/override
   any overlapping keys from `shift_dict`.
3. YAML missing or empty           → use shift_dict alone.
"""

import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

def _parse_shift(txt: str) -> float:
    """
    Convert a human-friendly shift string to decimal.

    "+200bp", "-75bps", "1.5%", "-0.01"  →  0.02, -0.0075, 0.015, -0.01
    """
    t = txt.strip().lower().replace(" ", "")
    if t.endswith("%"):
        return float(t[:-1]) / 100
    if t.endswith(("bp", "bps")):
        return float(t.rstrip("ps").rstrip("bp")) / 10_000
    return float(t)                       # already decimal

def parse_delta(
    yaml_path: Optional[str] = None,
    literal_shift: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """
    Parse a what-if scenario.

    Parameters
    ----------
    yaml_path : str | None
        Path to a YAML file that may contain `new_weights:` or `delta:`.
    literal_shift : dict | None
        In-memory dict of {ticker: shift_string}.  Overrides YAML deltas.

    Returns
    -------
    (delta_dict, new_weights_dict_or_None)
    """
    # LOGGING: Log function entry with parameters yaml_path and literal_shift type/presence
    delta: Dict[str, float] = {}
    new_w: Optional[Dict[str, float]] = None

    # ── YAML branch (only if file is present) ─────────────────────────
    if yaml_path and Path(yaml_path).is_file():
        # LOGGING: Log YAML file loading attempt with file path
        cfg = yaml.safe_load(Path(yaml_path).read_text()) or {}
        
        # 1) full-replacement portfolio
        if "new_weights" in cfg:               
            # LOGGING: Log full portfolio replacement mode with new_weights count
            w = {k: float(v) for k, v in cfg["new_weights"].items()}
            from portfolio_risk import normalize_weights
            new_w = normalize_weights(w)
            # LOGGING: Log successful portfolio replacement with normalized weights summary
            return {}, new_w

        # 2) incremental tweaks
        if "delta" in cfg:                     
            # LOGGING: Log incremental delta parsing with delta keys
            delta.update({k: _parse_shift(v) for k, v in cfg["delta"].items()})

    # ── literal shift branch (CLI / notebook) ────────────────────────
    if literal_shift:
        # LOGGING: Log literal shift processing with ticker count and shift values
        delta.update({k: _parse_shift(v) for k, v in literal_shift.items()})

    # ── sanity check -------------------------------------------------------
    if not delta and new_w is None:
        # LOGGING: Log error condition - no valid input provided
        raise ValueError(
            "No delta or new_weights provided (YAML empty and literal_shift is None)"
        )

    # LOGGING: Log successful delta parsing with final delta dict summary
    return delta, new_w


# In[ ]:




