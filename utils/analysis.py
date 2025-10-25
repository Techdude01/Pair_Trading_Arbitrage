import pickle
import pandas as pd
from utils.io import load_data
from utils.preprocess import get_close_cols
from utils.spread import calculate_spread
from utils.stats import calculate_half_life, calculate_rolling_correlation
from utils.config import get_stock_data_path


def run_analysis():
    """Analyze all cointegrated pairs and print summary statistics."""
    file_path = get_stock_data_path()
    df = load_data(file_path)
    df = get_close_cols(df)

    with open('cointegrated_pairs.pkl', 'rb') as f:
        copairs = pickle.load(f)

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
            'Intercept': intercept,
            'Half Life': half_life,
            'Spread Mean': spread.mean(),
            'Spread Std': spread.std(),
            'Rolling Correlation': rolling_correlation.mean(),
            'Cointegration P-value': pair['pvalue'],
            'ADF Statistic': pair['adf_statistic'],
            'R Squared': pair['r_squared']
        })

    pair_results.sort(key=lambda x: x['Rolling Correlation'], reverse=True)
    summary_df = pd.DataFrame(pair_results)
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    return pair_results


def select_good_pairs(criteria=None):
    """Select pairs that meet trading criteria for statistical arbitrage."""
    if criteria is None:
        criteria = {
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

    pair_results = run_analysis()
    good_pairs = []

    if not pair_results or not isinstance(pair_results, list):
        print(f"ERROR: pair_results is {type(pair_results)}, expected list")
        return [], []

    for pair_result in pair_results:
        if not isinstance(pair_result, dict):
            print(f"SKIPPING: Invalid data type {type(pair_result)}")
            continue

        meets_criteria = (
            pair_result['Cointegration P-value'] < criteria['max_pvalue'] and
            pair_result['ADF Statistic'] < criteria['min_adf_statistic'] and
            criteria['min_half_life'] <= pair_result['Half Life'] <= criteria['max_half_life'] and
            abs(pair_result['Spread Mean']) < criteria['max_spread_mean_abs'] and
            criteria['min_spread_std'] <= pair_result['Spread Std'] <= criteria['max_spread_std'] and
            pair_result['R Squared'] >= criteria['min_r_squared'] and
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


