"""Analysis utilities for pair statistics and selection."""
import pandas as pd
from utils.io import load_data, load_pairs
from utils.preprocess import get_close_cols
from utils.spread import calculate_spread
from utils.stats import calculate_half_life, calculate_rolling_correlation
from utils.config import get_stock_data_path, get_default_criteria


def run_analysis(pairs_file='cointegrated_pairs.pkl'):
    """
    Analyze all cointegrated pairs and print summary statistics.
    
    Args:
        pairs_file (str): Path to pickle file containing cointegrated pairs
        
    Returns:
        list: List of pair analysis result dictionaries
    """
    file_path = get_stock_data_path()
    df = load_data(file_path)
    df = get_close_cols(df)
    # calculate statistics for pairs:
    copairs = load_pairs(pairs_file)
    # =
    pair_results = []
    
    for pair in copairs:
        ticker1, ticker2 = pair['tickers']
        hedge_ratio = pair['hedge_ratio']
        intercept = pair['intercept']

        close_col1 = f"Close__{ticker1}"
        close_col2 = f"Close__{ticker2}"

        spread, _, _ = calculate_spread(
            df[close_col1],
            df[close_col2],
            hedge_ratio=hedge_ratio,
            intercept=intercept
        )
        half_life = calculate_half_life(spread)
        rolling_correlation = calculate_rolling_correlation(df[close_col1], df[close_col2])
        pair_results.append({
            'Pair': f'{ticker1}-{ticker2}',
            'Hedge Ratio': hedge_ratio,
            'Intercept': intercept,  # Store for reference
            'Half Life': half_life,
            'Spread Mean': spread.mean(),  # Should now be ≈ 0
            'Spread Std': spread.std(), 
            'Rolling Correlation': rolling_correlation.mean(),
            'Cointegration P-value': pair['pvalue'],
            'ADF Statistic': pair['adf_statistic'],  # For ranking
            'R Squared': pair['r_squared']  # Regression quality
        })

    pair_results.sort(key=lambda x: x['Rolling Correlation'], reverse=True)
    summary_df = pd.DataFrame(pair_results)
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    return pair_results


def select_good_pairs(criteria=None, pairs_file='cointegrated_pairs.pkl'):
    """
    Select pairs that meet trading criteria for statistical arbitrage.
    
    Args:
        criteria (dict, optional): Custom criteria thresholds. If None, uses defaults.
        pairs_file (str): Path to pickle file containing cointegrated pairs
        
    Returns:
        tuple: (good_pairs, all_pair_results) - filtered and all results
    """
    if criteria is None:
        criteria = get_default_criteria()

    pair_results = run_analysis(pairs_file)
    good_pairs = []

    if not pair_results or not isinstance(pair_results, list):
        print(f"ERROR: pair_results is {type(pair_results)}, expected list")
        return [], []

    for pair_result in pair_results:
        if not isinstance(pair_result, dict):
            print(f"SKIPPING: Invalid data type {type(pair_result)}")
            continue

        # Check all criteria
        meets_criteria = (
            # Primary: Cointegration p-value and ADF statistic
            pair_result['Cointegration P-value'] < criteria['max_pvalue'] and
            pair_result['ADF Statistic'] < criteria['min_adf_statistic'] and
            
            # Primary: Mean reversion speed
            criteria['min_half_life'] <= pair_result['Half Life'] <= criteria['max_half_life'] and
            
            # Secondary: Spread properties
            abs(pair_result['Spread Mean']) < criteria['max_spread_mean_abs'] and
            criteria['min_spread_std'] <= pair_result['Spread Std'] <= criteria['max_spread_std'] and
            
            # Secondary: Regression quality
            pair_result['R Squared'] >= criteria['min_r_squared'] and
            
            # Tertiary: Correlation (pre-screening only)
            criteria['min_correlation'] <= pair_result['Rolling Correlation'] <= criteria['max_correlation']
        )

        if meets_criteria:
            good_pairs.append(pair_result)
            print(f"✓ {pair_result['Pair']}")
            print(f"  P-value: {pair_result['Cointegration P-value']:.6f}, ADF: {pair_result['ADF Statistic']:.2f}")
            print(f"  Half-life: {pair_result['Half Life']:.1f}d, R²: {pair_result['R Squared']:.3f}")
            print(f"  Spread: μ={pair_result['Spread Mean']:.4f}, σ={pair_result['Spread Std']:.3f}")
            print(f"  Correlation: {pair_result['Rolling Correlation']:.3f}")
            print()

    print("\nCriteria used:")
    print(f"- Rolling correlation > {criteria['min_correlation']}")
    print(f"- Cointegration p-value < {criteria['max_pvalue']}")
    print(f"- Spread mean is close to 0 (< {criteria['max_spread_mean_abs']})")
    print(f"- Spread std is relatively low (< {criteria['max_spread_std']})")
    print(f"- Half life is reasonable ({criteria['min_half_life']}-{criteria['max_half_life']} days)")

    return good_pairs, pair_results


