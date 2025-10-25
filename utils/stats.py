"""Statistical calculation utilities."""
import numpy as np
import statsmodels.api as sm


def calculate_half_life(spread):
    """
    Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.
    
    Args:
        spread (pd.Series): Spread time series
        
    Returns:
        float: Half-life in days (inf if no mean reversion detected)
    """
    spread_clean = spread.dropna()
    # half life is the time it takes for the spread to return to the mean
    # Use Ornstein-Uhlenbeck process
    spread_lag = spread_clean.shift(1).dropna()
    spread_diff = spread_clean.diff().dropna()
    # remove first NaN
    spread_lag = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]
    assert len(spread_lag) == len(spread_diff), "Spread lag and spread diff are not the same length"
    # Adjust for the dropped values
    X = sm.add_constant(spread_lag)
    model = sm.OLS(spread_diff, X)
    results = model.fit()
    beta = results.params.iloc[1]
    # half life is -1/beta * ln(2)
    # if beta is positive/0, return infinity
    # this is because the spread is not mean reverting
    if beta >= 0:
        return float('inf')
    return -np.log(2) / beta


def calculate_rolling_correlation(price_1, price_2, window=30):
    """
    Calculate rolling correlation of log returns between two price series.
    
    Args:
        price_1 (pd.Series): First price series
        price_2 (pd.Series): Second price series
        window (int): Rolling window size in days (default: 30)
        
    Returns:
        pd.Series: Rolling correlation values
    """
    # calculate the rolling correlation of price_1 and price_2
    # use log returns so that the correlation is more stable
    log_returns_1 = np.log(price_1 / price_1.shift(1))
    log_returns_2 = np.log(price_2 / price_2.shift(1))
    return log_returns_1.rolling(window=window).corr(log_returns_2)


