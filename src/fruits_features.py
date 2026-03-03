"""
FRUITS Feature Extraction for Time Series Data

This module provides feature extraction using the FRUITS (Feature extraction from 
Regularly and Irregularly sampled Time Series) library for financial time series data.
"""

import numpy as np
import pandas as pd
import fruits
from typing import Optional


def compute_fruits_features(
    data: pd.DataFrame,
    sequence_length: int = 21,  # 24h * 21 days for hourly, or 21 days for daily
    price_column: str = 'close',
    standardize: bool = True,
    remove_duplicates: bool = True,
    word_weights: list = [1, 2, 3, 4],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute FRUITS features from time series data.
    
    This function extracts features from time series using the FRUITS methodology,
    which captures linear and non-linear temporal patterns through iterated-sums 
    signatures (ISS) with different word weights.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with datetime index and OHLCV columns
    sequence_length : int, default=504
        Length of sequences to extract features from (e.g., 504 = 24h * 21 days for hourly data)
    price_column : str, default='close'
        Column name to use for feature extraction
    standardize : bool, default=True
        Whether to standardize sequences before feature extraction
    remove_duplicates : bool, default=True
        Whether to remove duplicate features based on correlation
    word_weights : list, default=[1, 2, 3, 4]
        List of word weights to use for ISS feature extraction
        - weight 1: Linear patterns
        - weight 2: Quadratic patterns
        - weight 3: Cubic patterns
        - weight 4: Higher-order patterns
    verbose : bool, default=True
        Whether to print progress information
        
    Returns
    -------
    pd.DataFrame
        Original data with FRUITS features added as new columns
        
    Notes
    -----
    - The function computes log returns from the price column
    - Features are extracted using sliding windows of length `sequence_length`
    - The first `sequence_length - 1` rows will be NaN (insufficient data for features)
    - Each word weight captures different aspects of temporal patterns
    
    Examples
    --------
    >>> data = pd.read_csv('price_data.csv', parse_dates=['date'], index_col='date')
    >>> data_with_features = compute_fruits_features(data, sequence_length=504)
    >>> # Remove NaN rows from the beginning
    >>> data_with_features = data_with_features.dropna()
    """
    
    if verbose:
        print("=" * 60)
        print("FRUITS FEATURE EXTRACTION")
        print("=" * 60)
    
    # Validate input
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in data. Available columns: {data.columns.tolist()}")
    
    if len(data) < sequence_length:
        raise ValueError(f"Data length ({len(data)}) is less than sequence_length ({sequence_length})")
    
    # Step 1: Compute log returns
    if verbose:
        print(f"\n1. Computing log returns from '{price_column}' column...")
    
    data = data.copy()
    data['log_return'] = np.log(data[price_column] / data[price_column].shift(1))
    
    # Remove first NaN from log return calculation
    data = data.dropna(subset=['log_return'])
    
    if verbose:
        print(f"   Log return stats: mean={data['log_return'].mean():.6f}, std={data['log_return'].std():.6f}")
    
    # Step 2: Create sequences
    if verbose:
        print(f"\n2. Creating sequences of length {sequence_length}...")
    
    X_values = data['log_return'].values
    n_sequences = len(X_values) - sequence_length + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Not enough data points ({len(X_values)}) for sequence_length ({sequence_length})")
    
    X_sequences = np.array([X_values[i:i+sequence_length] for i in range(n_sequences)])
    X_sequences = X_sequences.reshape(n_sequences, sequence_length, 1)
    
    if verbose:
        print(f"   Created {n_sequences} sequences of shape {X_sequences.shape}")
    
    # Step 3: Standardize sequences
    if standardize:
        if verbose:
            print("\n3. Standardizing sequences...")
        
        X_std = np.zeros_like(X_sequences)
        for i in range(X_sequences.shape[0]):
            seq = X_sequences[i, :, 0]
            mean = np.mean(seq)
            std = np.std(seq)
            if std > 0:
                X_std[i, :, 0] = (seq - mean) / std
            else:
                X_std[i, :, 0] = 0
        
        if verbose:
            print(f"   Standardized data: mean={np.mean(X_std):.6f}, std={np.std(X_std):.6f}")
    else:
        X_std = X_sequences
    
    # Step 4: Create and fit FRUITS pipeline
    if verbose:
        print("\n4. Creating FRUITS pipeline...")
        print(f"   Using word weights: {word_weights}")
    
    fruit = _create_optimized_fruit(word_weights)
    
    if verbose:
        print("\n5. Fitting and transforming with FRUITS...")
    
    try:
        fruit.fit(X_std)
        X_features = fruit.transform(X_std)
    except Exception as e:
        if verbose:
            print(f"   Error with optimized pipeline: {e}")
            print("   Falling back to simple pipeline...")
        
        # Fallback to simple pipeline
        fruit = _create_simple_fruit(word_weights[:3])  # Use only first 3 weights
        fruit.fit(X_std)
        X_features = fruit.transform(X_std)
    
    if verbose:
        print(f"   Features shape: {X_features.shape}")
        print(f"   Non-zero features: {np.sum(np.any(X_features != 0, axis=0))}/{X_features.shape[1]}")
    
    # Step 5: Remove duplicate features
    if remove_duplicates:
        if verbose:
            print("\n6. Removing duplicate features...")
        
        X_features, unique_indices = _remove_duplicate_features(X_features, verbose=verbose)
        
        if verbose:
            print(f"   Final features shape: {X_features.shape}")
    
    # Step 6: Add features to DataFrame
    if verbose:
        print("\n7. Adding features to DataFrame...")
    
    # Create feature columns (aligned with the end of each sequence)
    n_features = X_features.shape[1]
    feature_names = [f'FRUITS_Feature_{i}' for i in range(n_features)]
    
    # Initialize feature columns with NaN
    for feature_name in feature_names:
        data[feature_name] = np.nan
    
    # Fill in feature values (starting from sequence_length-1 index)
    start_idx = sequence_length - 1
    for i, feature_name in enumerate(feature_names):
        data.iloc[start_idx:start_idx + len(X_features), data.columns.get_loc(feature_name)] = X_features[:, i]
    
    if verbose:
        print(f"   Added {n_features} FRUITS features to DataFrame")
        print(f"   First {sequence_length - 1} rows will be NaN (insufficient data)")
        print("\n" + "=" * 60)
        print("FRUITS FEATURE EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"\nFeature columns added: {feature_names}")
        print(f"Total rows: {len(data)}")
        print(f"Rows with features: {len(data) - (sequence_length - 1)}")
        print(f"\n⚠️  Remember to drop NaN rows: data = data.dropna()")
    
    return data


def _create_optimized_fruit(word_weights: list = [1, 2, 3, 4]):
    """
    Create optimized FRUITS pipeline based on data characteristics.
    
    Parameters
    ----------
    word_weights : list
        List of word weights to use for feature extraction
        
    Returns
    -------
    fruits.Fruit
        Configured FRUITS pipeline
    """
    fruit = fruits.Fruit("Optimized Fruit")
    
    # Create ISS with multiple word weights
    words = None
    for weight in word_weights:
        if words is None:
            words = fruits.words.of_weight(weight, dim=1)
        else:
            words = words + fruits.words.of_weight(weight, dim=1)
    
    iss_main = fruits.ISS(words, mode=fruits.ISSMode.EXTENDED)
    fruit.add(iss_main)
    
    # Add END sieve (works reliably)
    fruit.add(fruits.sieving.END)
    
    # Try to add MIN/MAX sieves if available
    try:
        fruit.cut()
        iss_minmax = fruits.ISS(
            fruits.words.of_weight(1, dim=1) + fruits.words.of_weight(2, dim=1),
            mode=fruits.ISSMode.EXTENDED,
        )
        fruit.add(iss_minmax)
        fruit.add(fruits.sieving.MIN)
        fruit.add(fruits.sieving.MAX)
    except AttributeError:
        # MIN/MAX not available, just use END
        pass
    
    return fruit


def _create_simple_fruit(word_weights: list = [1, 2, 3]):
    """
    Create simple but effective FRUITS pipeline (fallback).
    
    Parameters
    ----------
    word_weights : list
        List of word weights to use for feature extraction
        
    Returns
    -------
    fruits.Fruit
        Configured FRUITS pipeline
    """
    fruit = fruits.Fruit("Simple Fruit")
    
    # Create ISS with multiple word weights
    words = None
    for weight in word_weights:
        if words is None:
            words = fruits.words.of_weight(weight, dim=1)
        else:
            words = words + fruits.words.of_weight(weight, dim=1)
    
    iss = fruits.ISS(words, mode=fruits.ISSMode.EXTENDED)
    fruit.add(iss)
    
    # Add END sieve
    fruit.add(fruits.sieving.END)
    
    return fruit


def _remove_duplicate_features(X: np.ndarray, correlation_threshold: float = 0.99, verbose: bool = True) -> tuple:
    """
    Remove duplicate features based on correlation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    correlation_threshold : float, default=0.99
        Correlation threshold above which features are considered duplicates
    verbose : bool, default=True
        Whether to print information about removed features
        
    Returns
    -------
    tuple
        (X_unique, unique_indices) where X_unique is the deduplicated feature matrix
        and unique_indices is the list of kept feature indices
    """
    # Get only non-zero features first
    nonzero_mask = np.any(X != 0, axis=0)
    nonzero_indices = np.where(nonzero_mask)[0]
    
    if len(nonzero_indices) == 0:
        if verbose:
            print("   Warning: No non-zero features found!")
        return X, list(range(X.shape[1]))
    
    # Work only with non-zero features
    X_nonzero = X[:, nonzero_indices]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X_nonzero.T)
    
    # Find highly correlated features
    duplicate_indices_relative = set()
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > correlation_threshold:
                duplicate_indices_relative.add(j)
    
    # Keep only unique features (relative to nonzero indices)
    unique_indices_relative = [i for i in range(len(nonzero_indices)) if i not in duplicate_indices_relative]
    
    # Map back to original indices
    unique_indices = [nonzero_indices[i] for i in unique_indices_relative]
    
    X_unique = X[:, unique_indices]
    
    if verbose:
        print(f"   Found {len(nonzero_indices)} non-zero features")
        print(f"   Removed {len(duplicate_indices_relative)} duplicate features")
        print(f"   Kept {len(unique_indices)} unique features at indices: {unique_indices}")
    
    return X_unique, unique_indices


def analyze_fruits_features(data: pd.DataFrame, feature_prefix: str = 'FRUITS_Feature_'):
    """
    Analyze and print statistics about FRUITS features.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing FRUITS features
    feature_prefix : str, default='FRUITS_Feature_'
        Prefix used for FRUITS feature column names
        
    Returns
    -------
    pd.DataFrame
        Summary statistics for FRUITS features
    """
    feature_cols = [col for col in data.columns if col.startswith(feature_prefix)]
    
    if len(feature_cols) == 0:
        print(f"No features found with prefix '{feature_prefix}'")
        return None
    
    print("=" * 60)
    print("FRUITS FEATURE ANALYSIS")
    print("=" * 60)
    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"\nFeature statistics:")
    
    stats = data[feature_cols].describe()
    print(stats)
    
    # Correlation analysis
    print(f"\nFeature correlations:")
    corr_matrix = data[feature_cols].corr()
    print(corr_matrix)
    
    return stats