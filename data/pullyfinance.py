import yfinance as yf
import pandas as pd

pairs = [
    ['NVDA', 'AMD'],   # NVIDIA / Advanced Micro Devices
    ['MSFT', 'GOOGL'], # Microsoft / Alphabet (Google)
    ['AAPL', 'MSFT'],  # Apple / Microsoft
    ['V', 'MA'],       # Visa / Mastercard
    ['CRM', 'ADBE'],   # Salesforce / Adobe
    ['INTC', 'QCOM'],  # Intel / Qualcomm
    ['CSCO', 'JNPR'],  # Cisco / Juniper Networks
    ['ORCL', 'SAP'],   # Oracle / SAP
    ['UBER', 'LYFT'],  # Uber / Lyft
    ['META', 'SNAP']   # Meta Platforms / Snap Inc.
]
tickers = [ticker for pair in pairs for ticker in pair]
data = yf.download(tickers, start='2020-01-01', end='2025-01-01')
print(data)
# Analyze NaNs in each column
nan_counts = data.isna().sum()
print("Number of NaNs in each column:")
print(nan_counts)
#0 NaNs, no data cleaning needed


