"""I/O utilities for loading and saving data."""
import pickle
import pandas as pd


def load_data(file_path):
    """
    Load stock data from parquet or CSV file.
    
    Args:
        file_path (str): Path to data file
        
    Returns:
        pd.DataFrame: Loaded stock data
        
    Raises:
        ValueError: If file type is not supported
    """
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path, engine='fastparquet')
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    raise ValueError(f'Unsupported file type: {file_path}')


def load_pairs(filename='cointegrated_pairs.pkl'):
    """
    Load cointegrated pairs from pickle file.
    
    Args:
        filename (str): Path to pickle file
        
    Returns:
        list: List of cointegrated pair dictionaries
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_top_pairs(pairs, top_n=5, filename='cointegrated_pairs.pkl'):
    """
    Save top N cointegrated pairs to a pickle file.
    
    Args:
        pairs (list): List of pair dictionaries
        top_n (int): Number of top pairs to save
        filename (str): Output pickle file path
    """
    if not pairs:
        print('No pairs to save.')
        return
    pairs_sorted = sorted(pairs, key=lambda x: x['pvalue'])
    top_pairs = pairs_sorted[:top_n]
    with open(filename, 'wb') as f:
        pickle.dump(top_pairs, f)
    print(f'Saved top {len(top_pairs)} pairs to {filename}')


