
def get_close_cols(df):
    """Extract Date and all Close price columns from DataFrame."""
    close_cols = [col for col in df.columns if 'Close' in col]
    df_close = df[['Date'] + close_cols]
    return df_close


