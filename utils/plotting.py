"""Plotting utilities for visualization."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from utils.io import load_pairs
from utils.spread import calculate_spread
from utils.stats import calculate_rolling_correlation


def plot_pair_analysis(df, pair_results):
    """
    Create a 4-panel plot for cointegration analysis of a trading pair.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data with Date column
        pair_results (dict): Dictionary containing pair analysis results
    """
    ticker1, ticker2 = pair_results['Pair'].split('-')
    close_col1 = f"Close__{ticker1}"
    close_col2 = f"Close__{ticker2}"

    hedge_ratio = pair_results['Hedge Ratio']
    half_life = pair_results['Half Life']
    intercept = pair_results.get('Intercept', 0)
    spread, _, _ = calculate_spread(df[close_col1], df[close_col2], hedge_ratio, intercept)
    rolling_correlation = calculate_rolling_correlation(df[close_col1], df[close_col2])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{ticker1} vs {ticker2} Cointegration Analysis')

    axes[0,0].plot(df['Date'], df[close_col1]/df[close_col1].iloc[0], label=ticker1, alpha=0.7)
    axes[0,0].plot(df['Date'], df[close_col2]/df[close_col2].iloc[0], label=ticker2, alpha=0.7)
    axes[0,0].set_title('Price Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True)

    axes[0,1].plot(df['Date'], spread, color='red', alpha=0.7)
    axes[0,1].axhline(y=spread.mean(), color='black', linestyle='--', alpha=0.5)
    axes[0,1].set_title(f'Spread (Half-life: {half_life:.1f} days)')
    axes[0,1].grid(True)

    axes[1,0].plot(df['Date'], rolling_correlation, color='green', alpha=0.7)
    axes[1,0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    axes[1,0].set_title('Rolling Correlation (30-day)')
    axes[1,0].grid(True)

    axes[1,1].hist(spread.dropna(), bins=30, alpha=0.7, color='purple')
    axes[1,1].set_title('Spread Distribution')
    axes[1,1].grid(True)

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def create_pvalue_heatmap(pairs_file='cointegrated_pairs.pkl'):
    """
    Create and show a heatmap of cointegration p-values for all pairs.
    
    Args:
        pairs_file (str): Path to pickle file containing cointegrated pairs
        
    Returns:
        tuple: (pvalue_matrix, tickers) - p-value matrix and ticker list
    """
    copairs = load_pairs(pairs_file)

    tickers = set()
    for pair in copairs:
        ticker1, ticker2 = pair['tickers']
        tickers.add(ticker1)
        tickers.add(ticker2)

    tickers = sorted(list(tickers))
    n = len(tickers)
    pvalue_matrix = np.ones((n, n))

    for pair in copairs:
        pvalue = pair['pvalue']
        ticker1, ticker2 = pair['tickers']
        idx1 = tickers.index(ticker1)
        idx2 = tickers.index(ticker2)
        pvalue_matrix[idx1, idx2] = pvalue
        pvalue_matrix[idx2, idx1] = pvalue

    np.fill_diagonal(pvalue_matrix, np.nan)

    plt.figure(figsize=(14, 12))
    mask = pvalue_matrix == 1.0
    sns.heatmap(
        pvalue_matrix,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn_r',
        xticklabels=tickers,
        yticklabels=tickers,
        cbar_kws={'label': 'P-value'},
        vmin=0,
        vmax=0.1,
        linewidths=0.5,
        linecolor='gray',
        mask=mask
    )

    plt.title('Cointegration P-value Heatmap\n(Lower p-values = Stronger cointegration)', fontsize=16, pad=20)
    plt.xlabel('Ticker', fontsize=12)
    plt.ylabel('Ticker', fontsize=12)
    plt.tight_layout()

    valid_pvalues = pvalue_matrix[~np.isnan(pvalue_matrix) & (pvalue_matrix < 1.0)]
    print(f"\nP-value Statistics:")
    print(f"Total pairs tested: {len(copairs)}")
    print(f"Significant pairs (p < 0.05): {np.sum(valid_pvalues < 0.05)}")
    print(f"Highly significant pairs (p < 0.01): {np.sum(valid_pvalues < 0.01)}")
    print(f"Very highly significant pairs (p < 0.001): {np.sum(valid_pvalues < 0.001)}")
    print(f"\nMin p-value: {np.min(valid_pvalues):.6f}")
    print(f"Max p-value: {np.max(valid_pvalues):.6f}")
    print(f"Median p-value: {np.median(valid_pvalues):.6f}")

    return pvalue_matrix, tickers


