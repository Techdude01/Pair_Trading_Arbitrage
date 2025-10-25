"""Spread calculation utilities."""
import numpy as np
import statsmodels.api as sm


def get_hedge_ratio(price_1, price_2):
    """
    Calculate optimal hedge ratio using OLS regression on log prices.
    
    Args:
        price_1 (pd.Series): First price series
        price_2 (pd.Series): Second price series
        
    Returns:
        tuple: (hedge_ratio, intercept)
    """
    # hedge ratio is the slope of the linear regression of price_1 vs price_2
    # y = mx + b + e
    # where e is the error term
    # log(price_1) = beta*log(price_2) + alpha + e
    # solve for beta, use OLS (least squares)
    log_price_1 = np.log(price_1)
    log_price_2 = np.log(price_2)
    X = sm.add_constant(log_price_2)
    model = sm.OLS(log_price_1, X)
    results = model.fit()
    
    alpha = results.params.iloc[0]  # Intercept
    beta = results.params.iloc[1]   # Hedge ratio
    
    return beta, alpha  # Return BOTH


def calculate_spread(price_1, price_2, hedge_ratio=None, intercept=None):
    """
    Calculate spread: log(price_1) - (intercept + hedge_ratio * log(price_2)).
    
    Args:
        price_1 (pd.Series): First price series
        price_2 (pd.Series): Second price series
        hedge_ratio (float, optional): Hedge ratio (beta). If None, will be calculated.
        intercept (float, optional): Intercept (alpha). If None, will be calculated.
        
    Returns:
        tuple: (spread, hedge_ratio, intercept)
    """
    # equation: log(price_1) - (hedge_ratio * log(price_2))
    if hedge_ratio is None or intercept is None:
        hedge_ratio, intercept = get_hedge_ratio(price_1, price_2)
        
    log_price_1 = np.log(price_1)
    log_price_2 = np.log(price_2)
    
    spread = log_price_1 - (intercept + hedge_ratio * log_price_2)
    
    return spread, hedge_ratio, intercept


