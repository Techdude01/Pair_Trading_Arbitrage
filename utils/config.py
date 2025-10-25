import os


def get_default_tickers():
    """Return the default list of tickers used in analysis."""
    return [
        'NVDA', 'AMD', 'MSFT', 'GOOGL', 'AAPL', 'V', 'MA', 'CRM', 'ADBE', 'INTC',
        'QCOM', 'CSCO', 'ANET', 'ORCL', 'SAP', 'UBER', 'LYFT', 'META', 'SNAP'
    ]


def get_stock_data_path():
    """Return the stock data path from environment."""
    return os.getenv('stock_data_path')


