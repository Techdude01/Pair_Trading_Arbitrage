"""Configuration utilities for StatArb analysis."""
import os


def get_default_tickers():
    """
    Return the default list of tickers used in analysis.
    
    Returns:
        list: List of ticker symbols
    """
    return [
        'NVDA', 'AMD', 'MSFT', 'GOOGL', 'AAPL', 'V', 'MA', 'CRM', 'ADBE', 'INTC',
        'QCOM', 'CSCO', 'ANET', 'ORCL', 'SAP', 'UBER', 'LYFT', 'META', 'SNAP'
    ]


def get_stock_data_path():
    """
    Return the stock data path from environment variable.
    
    Returns:
        str: Path to stock data file
    """
    return os.getenv('stock_data_path')


def get_default_criteria():
    """
    Return default trading criteria for pair selection.
    
    Returns:
        dict: Dictionary of criteria thresholds
    """
    return {
        'max_pvalue': 0.05,
        'min_adf_statistic': -2.86,
        'min_half_life': 1,
        'max_half_life': 30,
        'max_spread_mean_abs': 0.1,
        'min_spread_std': 0.001,
        'max_spread_std': 0.5,
        'min_r_squared': 0.5,
        'min_correlation': 0.3,
        'max_correlation': 0.95,
    }


