"""Data preprocessing utilities."""


def get_close_cols(df):
    """
    Extract Date and all Close price columns from DataFrame.
    
    Args:
        df (pd.DataFrame): Input dataframe with stock data
        
    Returns:
        pd.DataFrame: DataFrame with Date and Close price columns only
    """
    close_cols = [col for col in df.columns if 'Close' in col]
    df_close = df[['Date'] + close_cols]
    return df_close


