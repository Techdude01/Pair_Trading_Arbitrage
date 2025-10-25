import numpy as np
import statsmodels.api as sm


def get_hedge_ratio(price_1, price_2):
    """Calculate optimal hedge ratio using OLS regression on log prices."""
    log_price_1 = np.log(price_1)
    log_price_2 = np.log(price_2)
    X = sm.add_constant(log_price_2)
    model = sm.OLS(log_price_1, X)
    results = model.fit()
    alpha = results.params.iloc[0]
    beta = results.params.iloc[1]
    return beta, alpha


def calculate_spread(price_1, price_2, hedge_ratio=None, intercept=None):
    """Calculate spread: log(price_1) - (intercept + hedge_ratio * log(price_2))."""
    if hedge_ratio is None or intercept is None:
        hedge_ratio, intercept = get_hedge_ratio(price_1, price_2)
    log_price_1 = np.log(price_1)
    log_price_2 = np.log(price_2)
    spread = log_price_1 - (intercept + hedge_ratio * log_price_2)
    return spread, hedge_ratio, intercept


