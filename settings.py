#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# settings.py  
# LOGGING: Add configuration loading logging
# LOGGING: Add critical alert for configuration errors here
# LOGGING: Add environment variable logging
# LOGGING: Add resource usage monitoring for settings initialization here
# LOGGING: Add settings validation logging
# LOGGING: Add configuration change tracking
PORTFOLIO_DEFAULTS = {
    "start_date": "2019-01-31",
    "end_date":   "2025-06-27",
    "normalize_weights": False,  # Global default for portfolio weight normalization
    "worst_case_lookback_years": 10  # Historical lookback period for worst-case scenario analysis

}

