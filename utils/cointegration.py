import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from utils.io import load_data, save_top_pairs
from utils.preprocess import get_close_cols
from utils.config import get_default_tickers, get_stock_data_path


def engle_granger_test(series1, series2, significance=0.05, alpha=0.05):
    """
    Perform Engle-Granger cointegration test on two price series.

    Regression: log(series1) = alpha + beta * log(series2) + error
    The residuals are tested for stationarity using ADF test by caller.
    """
    log_series1 = np.log(series1)
    log_series2 = np.log(series2)
    X = sm.add_constant(log_series2)
    model = sm.OLS(log_series1, X)
    results = model.fit()
    return results


def find_cointegrated_pairs(significance=0.05, alpha=0.05, save_top_n=5):
    """
    Find cointegrated stock pairs using Engle-Granger method and save top N.
    """
    file_path = get_stock_data_path()
    df = load_data(file_path)
    df = get_close_cols(df)
    tickers = get_default_tickers()

    copairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            ticker1 = tickers[i]
            ticker2 = tickers[j]
            results = engle_granger_test(df[f"Close__{ticker1}"], df[f"Close__{ticker2}"])
            adf_fuller_results = adfuller(results.resid)
            if adf_fuller_results[1] < significance:
                copairs.append({
                    'pvalue': adf_fuller_results[1],
                    'adf_statistic': adf_fuller_results[0],
                    'tickers': (ticker1, ticker2),
                    'intercept': results.params.iloc[0],
                    'hedge_ratio': results.params.iloc[1],
                    'r_squared': results.rsquared
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

    save_top_pairs(copairs, top_n=save_top_n, filename='cointegrated_pairs.pkl')
    return copairs


