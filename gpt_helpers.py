#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# === OPENAI API Setup ===
import openai
from io import StringIO
from contextlib import redirect_stdout
from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env into the environment

# Now access your keys like this
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# In[ ]:


# File: gpt_helpers.py

def interpret_portfolio_risk(diagnostics_text: str) -> str:
    """
    Sends raw printed diagnostics to GPT for layman interpretation.
    """
    user_prompt = (
        "You are a professional risk analyst at a hedge fund.\n"
        "I want you to help evaluate my portfolio. I will give you details of the portfolio's risk metrics.\n"
        "I want you to help interpret them for me and communicate with me in simple language.\n\n"
        f"{diagnostics_text}"
    )

    response = client.chat.completions.create(
        model="gpt-4.1",  # or "gpt-4o-mini" if desired
        messages=[
            {"role": "system", "content": "You are a portfolio risk analysis expert."},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=2000,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()

