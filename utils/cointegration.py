import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from utils.io import load_data, save_top_pairs
from utils.preprocess import get_close_cols
from utils.config import get_default_tickers, get_stock_data_path


def engle_granger_test(series1, series2):
    """
    Perform Engle-Granger cointegration test on two price series.
    
    Regression: log(series1) = alpha + beta * log(series2) + error
    The residuals are tested for stationarity using ADF test by caller.
    
    Args:
        series1 (pd.Series): First price series
        series2 (pd.Series): Second price series
        
    Returns:
        statsmodels.regression.linear_model.RegressionResults: OLS regression results
    """
    # Log transform FIRST, then add constant
    log_series1 = np.log(series1)
    log_series2 = np.log(series2)
    
    # Add constant to LOG prices
    X = sm.add_constant(log_series2)
    
    # Regress: log(series1) = alpha + beta * log(series2) + error
    model = sm.OLS(log_series1, X)
    results = model.fit()
    
    return results


def find_cointegrated_pairs(significance=0.05, save_top_n=5):
    """
    Find cointegrated stock pairs using Engle-Granger method and save top N.
    
    Args:
        significance (float): Significance level for ADF test (default: 0.05)
        save_top_n (int): Number of top pairs to save (default: 5)
        
    Returns:
        list: List of cointegrated pair dictionaries
    """
    file_path = get_stock_data_path()
    df = load_data(file_path)
    df = get_close_cols(df)
    # drop na
    tickers = get_default_tickers()

    """Engle Granger Test"""
    copairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            ticker1 = tickers[i]
            ticker2 = tickers[j]
            results = engle_granger_test(df[f"Close__{ticker1}"], df[f"Close__{ticker2}"])
            # get the residuals from the engle granger test and run the adf test
            adf_fuller_results = adfuller(results.resid)
            # if the signifiance level is less than the pvalue reject the null hypothesis and do not use the pair
            if adf_fuller_results[1] < significance:
                copairs.append({
                    'pvalue': adf_fuller_results[1],
                    'adf_statistic': adf_fuller_results[0],  # More negative = stronger
                    'tickers': (ticker1, ticker2),
                    'intercept': results.params.iloc[0],     # Alpha from regression
                    'hedge_ratio': results.params.iloc[1],   # Beta from regression
                    'r_squared': results.rsquared            # Regression fit quality
                })

    copairs.sort(key=lambda x: x['pvalue'])
    print("Top 5 cointegrated pairs:")
    print(f"{'Pair':<15} {'P-value':<12} {'ADF Stat':<10} {'Hedge Ratio':<12} {'R²':<8}")
    print("=" * 70)
    for i in range(min(5, len(copairs))):
        pair = copairs[i]
        t1, t2 = pair['tickers']
        print(f"{t1}-{t2:<12} {pair['pvalue']:<12.6f} {pair['adf_statistic']:<10.4f} "
              f"{pair['hedge_ratio']:<12.4f} {pair['r_squared']:<8.4f}")
    # save only the top N pairs via utils
    save_top_pairs(copairs, top_n=save_top_n, filename='cointegrated_pairs.pkl')
    return copairs


