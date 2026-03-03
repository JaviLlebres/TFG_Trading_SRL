"""
NetF (Network-based Features) Feature Extraction for Time Series Data

This module provides feature extraction using network-based methods for financial 
time series data. It constructs visibility graphs and quantile graphs from sequences
and extracts topological features from these networks.
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import time
import warnings
from typing import Optional, Tuple

warnings.filterwarnings('ignore')


class NetF:
    """
    Network-based Features (NetF) - Optimized single-threaded version.
    Works in Jupyter notebooks without multiprocessing issues.
    
    Creates three types of networks from time series:
    1. WNVG (Weighted Natural Visibility Graph)
    2. WHVG (Weighted Horizontal Visibility Graph)
    3. QG (Quantile Graph)
    
    Extracts 5 measures from each network:
    - Average weighted degree
    - Average path length
    - Clustering coefficient
    - Number of communities
    - Modularity
    
    Total: 15 features per sequence
    
    Optimizations:
    - Reduced quantiles (default 15 instead of 50)
    - Skip expensive measures for long sequences
    - Efficient algorithms
    - Progress tracking
    """
    
    def __init__(self, n_quantiles=15, max_nodes_for_expensive_measures=200):
        """
        Initialize NetF extractor.
        
        Parameters
        ----------
        n_quantiles : int, default=15
            Number of quantiles for quantile graph construction
        max_nodes_for_expensive_measures : int, default=200
            Skip expensive network measures if sequence length exceeds this
        """
        self.n_quantiles = n_quantiles
        self.max_nodes_for_expensive_measures = max_nodes_for_expensive_measures
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self):
        """Generate feature names for all network measures."""
        measures = ['avg_weighted_degree', 'avg_path_length', 'clustering_coeff', 
                   'n_communities', 'modularity']
        networks = ['WNVG', 'WHVG', 'QG']
        return [f"{network}_{measure}" for network in networks for measure in measures]
    
    def _fast_natural_visibility_graph(self, ts):
        """
        Create natural visibility graph from time series.
        Two nodes are connected if they can "see" each other (no intermediate point blocks the view).
        """
        n = len(ts)
        edges = []
        
        # Add immediate neighbors
        for i in range(n-1):
            edges.append((i, i+1))
        
        # Efficient visibility checking with limited lookahead
        for i in range(n):
            for j in range(i+2, min(i+50, n)):  # Limit lookahead for speed
                if self._has_visibility_fast(ts, i, j):
                    edges.append((i, j))
        
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        return G
    
    def _has_visibility_fast(self, ts, i, j):
        """Check if point i can see point j."""
        if j - i <= 1:
            return True
        for k in range(i + 1, j):
            y_threshold = ts[j] + (ts[i] - ts[j]) * (j - k) / (j - i)
            if ts[k] >= y_threshold:
                return False
        return True
    
    def _horizontal_visibility_graph(self, ts):
        """
        Create horizontal visibility graph from time series.
        Two nodes are connected if all intermediate points are below both.
        """
        n = len(ts)
        edges = []
        
        # Add immediate neighbors
        for i in range(n-1):
            edges.append((i, i+1))
        
        # Check horizontal visibility with limited lookahead
        for i in range(n):
            for j in range(i + 2, min(i+50, n)):  # Limit lookahead
                min_height = min(ts[i], ts[j])
                visible = all(ts[k] < min_height for k in range(i + 1, j))
                if visible:
                    edges.append((i, j))
        
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        return G
    
    def _add_weights_to_graph(self, G, ts):
        """Add edge weights based on Euclidean distance in time-value space."""
        for i, j in G.edges():
            weight = 1.0 / np.sqrt((j - i)**2 + (ts[j] - ts[i])**2)
            G[i][j]['weight'] = weight
        return G
    
    def quantile_graph(self, ts):
        """
        Create quantile graph from time series.
        Discretizes the series into quantile bins and creates a transition network.
        """
        quantile_points = np.linspace(0, 1, self.n_quantiles + 1)[1:]
        quantile_values = np.percentile(ts, quantile_points * 100, method='linear')
        
        # Create transition matrix
        transition_counts = np.zeros((self.n_quantiles, self.n_quantiles), dtype=np.int32)
        bins = np.searchsorted(quantile_values, ts, side='left')
        bins = np.clip(bins, 0, self.n_quantiles - 1)
        
        for i in range(len(bins) - 1):
            transition_counts[bins[i], bins[i+1]] += 1
        
        # Create directed graph with normalized weights
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_quantiles))
        
        for i in range(self.n_quantiles):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                for j in range(self.n_quantiles):
                    if transition_counts[i, j] > 0:
                        weight = transition_counts[i, j] / row_sum
                        G.add_edge(i, j, weight=weight)
        
        return G
    
    def compute_network_measures(self, G, skip_expensive=False):
        """
        Compute network topology measures.
        
        Parameters
        ----------
        G : networkx.Graph or networkx.DiGraph
            Network to analyze
        skip_expensive : bool
            If True, skip computationally expensive measures
            
        Returns
        -------
        dict
            Dictionary of network measures
        """
        measures = {}
        
        # Always compute: Average weighted degree
        if G.number_of_edges() > 0:
            degrees = []
            for node in G.nodes():
                if G.is_directed():
                    in_degree = sum(G[u][node].get('weight', 1) for u in G.predecessors(node))
                    out_degree = sum(G[node][v].get('weight', 1) for v in G.successors(node))
                    degrees.append(in_degree + out_degree)
                else:
                    degree = sum(G[node][neighbor].get('weight', 1) for neighbor in G.neighbors(node))
                    degrees.append(degree)
            measures['avg_weighted_degree'] = np.mean(degrees)
        else:
            measures['avg_weighted_degree'] = 0.0
        
        # Skip expensive measures if requested
        if skip_expensive:
            measures['avg_path_length'] = 0.0
            measures['clustering_coeff'] = 0.0
            measures['n_communities'] = 1
            measures['modularity'] = 0.0
            return measures
        
        # Average path length
        try:
            if G.is_directed():
                measures['avg_path_length'] = nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) else 0.0
            else:
                measures['avg_path_length'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else 0.0
        except:
            measures['avg_path_length'] = 0.0
        
        # Clustering coefficient
        try:
            G_undirected = G.to_undirected() if G.is_directed() else G
            measures['clustering_coeff'] = nx.transitivity(G_undirected)
        except:
            measures['clustering_coeff'] = 0.0
        
        # Communities and modularity
        try:
            G_undirected = G.to_undirected() if G.is_directed() else G
            if G_undirected.number_of_edges() > 0:
                communities = community.louvain_communities(G_undirected)
                measures['n_communities'] = len(communities)
                measures['modularity'] = community.modularity(G_undirected, communities)
            else:
                measures['n_communities'] = G.number_of_nodes()
                measures['modularity'] = 0.0
        except:
            measures['n_communities'] = 1
            measures['modularity'] = 0.0
        
        return measures
    
    def extract_features_from_sequence(self, sequence):
        """
        Extract all network features from a single sequence.
        
        Parameters
        ----------
        sequence : np.ndarray
            1D array representing a time series sequence
            
        Returns
        -------
        np.ndarray
            Array of 15 features (5 measures × 3 networks)
        """
        if len(sequence) < 3:
            return np.zeros(15)
        
        features = []
        skip_expensive = len(sequence) > self.max_nodes_for_expensive_measures
        
        # WNVG (Weighted Natural Visibility Graph)
        try:
            nvg = self._fast_natural_visibility_graph(sequence)
            wnvg = self._add_weights_to_graph(nvg, sequence)
            wnvg_measures = self.compute_network_measures(wnvg, skip_expensive)
            features.extend([
                wnvg_measures['avg_weighted_degree'],
                wnvg_measures['avg_path_length'],
                wnvg_measures['clustering_coeff'],
                wnvg_measures['n_communities'],
                wnvg_measures['modularity']
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # WHVG (Weighted Horizontal Visibility Graph)
        try:
            hvg = self._horizontal_visibility_graph(sequence)
            whvg = self._add_weights_to_graph(hvg, sequence)
            whvg_measures = self.compute_network_measures(whvg, skip_expensive)
            features.extend([
                whvg_measures['avg_weighted_degree'],
                whvg_measures['avg_path_length'],
                whvg_measures['clustering_coeff'],
                whvg_measures['n_communities'],
                whvg_measures['modularity']
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # QG (Quantile Graph)
        try:
            qg = self.quantile_graph(sequence)
            qg_measures = self.compute_network_measures(qg, skip_expensive=False)
            features.extend([
                qg_measures['avg_weighted_degree'],
                qg_measures['avg_path_length'],
                qg_measures['clustering_coeff'],
                qg_measures['n_communities'],
                qg_measures['modularity']
            ])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def fit(self, X):
        """Fit method (for sklearn compatibility)."""
        print(f"   NetF initialized with {self.n_quantiles} quantiles")
        print(f"   Sequence length: {X.shape[1]}")
        if X.shape[1] > self.max_nodes_for_expensive_measures:
            print(f"   Note: Skipping expensive measures due to long sequences")
        return self
    
    def transform(self, X):
        """
        Transform sequences into network features.
        
        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_sequences, sequence_length) or (n_sequences, sequence_length, 1)
            
        Returns
        -------
        np.ndarray
            Array of shape (n_sequences, 15) with extracted features
        """
        n_sequences = X.shape[0]
        features_matrix = np.zeros((n_sequences, 15))
        
        print(f"   Processing {n_sequences} sequences...")
        start_time = time.time()
        
        if X.ndim == 3:
            sequences = X[:, :, 0]
        else:
            sequences = X
        
        for i in range(n_sequences):
            features_matrix[i] = self.extract_features_from_sequence(sequences[i])
            
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (n_sequences - i - 1) / rate
                print(f"   Processed {i+1}/{n_sequences} ({100*(i+1)/n_sequences:.1f}%) - "
                      f"Rate: {rate:.1f} seq/s - ETA: {remaining/60:.1f} min")
        
        total_time = time.time() - start_time
        print(f"   Completed in {total_time:.1f}s ({total_time/60:.1f} min)")
        
        return features_matrix


def downsample_sequences(X: np.ndarray, target_length: int = 150) -> np.ndarray:
    """
    Downsample sequences to reduce computation time.
    
    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_sequences, sequence_length, 1)
    target_length : int
        Target sequence length after downsampling
        
    Returns
    -------
    np.ndarray
        Downsampled array
    """
    current_length = X.shape[1]
    if current_length <= target_length:
        return X
    
    step = current_length // target_length
    indices = np.arange(0, current_length, step)[:target_length]
    
    return X[:, indices, :]


def _remove_duplicate_features(X: np.ndarray, correlation_threshold: float = 0.99, 
                               verbose: bool = True) -> Tuple[np.ndarray, list]:
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
    unique_indices_relative = [i for i in range(len(nonzero_indices)) 
                               if i not in duplicate_indices_relative]
    
    # Map back to original indices
    unique_indices = [nonzero_indices[i] for i in unique_indices_relative]
    
    X_unique = X[:, unique_indices]
    
    if verbose:
        print(f"   Found {len(nonzero_indices)} non-zero features")
        print(f"   Removed {len(duplicate_indices_relative)} duplicate features")
        print(f"   Kept {len(unique_indices)} unique features at indices: {unique_indices}")
    
    return X_unique, unique_indices


def compute_netf_features(
    data: pd.DataFrame,
    sequence_length: int = 21,
    price_column: str = 'close',
    downsample_to: Optional[int] = 150,
    n_quantiles: int = 15,
    remove_duplicates: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute NetF (Network-based Features) from time series data.
    
    This function extracts network topology features from time series by creating
    visibility graphs and quantile graphs, then analyzing their structural properties.
    
    Networks created:
    1. WNVG (Weighted Natural Visibility Graph): Captures overall temporal patterns
    2. WHVG (Weighted Horizontal Visibility Graph): Captures local extrema patterns
    3. QG (Quantile Graph): Captures distribution and transition patterns
    
    Features extracted per network (15 total):
    - Average weighted degree: Node connectivity
    - Average path length: Network diameter
    - Clustering coefficient: Local connectivity
    - Number of communities: Network modularity structure
    - Modularity: Strength of community structure
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with datetime index and OHLCV columns
    sequence_length : int, default=504
        Length of sequences to extract features from (e.g., 504 = 24h * 21 days for hourly)
    price_column : str, default='close'
        Column name to use for feature extraction
    downsample_to : int or None, default=150
        Target length for downsampling sequences (speeds up computation)
        If None, no downsampling is performed
    n_quantiles : int, default=15
        Number of quantiles for quantile graph construction
    remove_duplicates : bool, default=True
        Whether to remove duplicate features based on correlation
    verbose : bool, default=True
        Whether to print progress information
        
    Returns
    -------
    pd.DataFrame
        Original data with NetF features added as new columns
        
    Notes
    -----
    - The function computes log returns from the price column
    - Features are extracted using sliding windows of length `sequence_length`
    - The first `sequence_length - 1` rows will be NaN (insufficient data for features)
    - Downsampling is recommended for long sequences to reduce computation time
    - Processing time: ~1-10 seconds per sequence depending on length and downsampling
    
    Examples
    --------
    >>> data = pd.read_csv('price_data.csv', parse_dates=['date'], index_col='date')
    >>> # For hourly data with downsampling (recommended)
    >>> data_with_features = compute_netf_features(data, sequence_length=504, downsample_to=150)
    >>> 
    >>> # For daily data without downsampling
    >>> data_with_features = compute_netf_features(data, sequence_length=21, downsample_to=None)
    >>> 
    >>> # Remove NaN rows from the beginning
    >>> data_with_features = data_with_features.dropna()
    """
    
    if verbose:
        print("=" * 60)
        print("NETF FEATURE EXTRACTION")
        print("=" * 60)
    
    # Validate input
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in data. "
                        f"Available columns: {data.columns.tolist()}")
    
    if len(data) < sequence_length:
        raise ValueError(f"Data length ({len(data)}) is less than "
                        f"sequence_length ({sequence_length})")
    
    # Step 1: Compute log returns
    if verbose:
        print(f"\n1. Computing log returns from '{price_column}' column...")
    
    data = data.copy()
    data['log_return'] = np.log(data[price_column] / data[price_column].shift(1))
    
    # Remove first NaN from log return calculation
    data = data.dropna(subset=['log_return'])
    
    if verbose:
        print(f"   Log return stats: mean={data['log_return'].mean():.6f}, "
              f"std={data['log_return'].std():.6f}")
    
    # Step 2: Create sequences
    if verbose:
        print(f"\n2. Creating sequences of length {sequence_length}...")
    
    X_values = data['log_return'].values
    n_sequences = len(X_values) - sequence_length + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Not enough data points ({len(X_values)}) for "
                        f"sequence_length ({sequence_length})")
    
    X_sequences = np.array([X_values[i:i+sequence_length] for i in range(n_sequences)])
    X_sequences = X_sequences.reshape(n_sequences, sequence_length, 1)
    
    if verbose:
        print(f"   Created {n_sequences} sequences of shape {X_sequences.shape}")
    
    # Step 3: Downsample if requested
    if downsample_to is not None and sequence_length > downsample_to:
        if verbose:
            print(f"\n3. Downsampling sequences from {sequence_length} to {downsample_to}...")
        
        X_to_process = downsample_sequences(X_sequences, target_length=downsample_to)
        
        if verbose:
            print(f"   Downsampled shape: {X_to_process.shape}")
            print(f"   Speedup factor: ~{sequence_length / downsample_to:.1f}x")
    else:
        X_to_process = X_sequences
        if verbose:
            print(f"\n3. No downsampling (sequence length: {sequence_length})")
    
    # Step 4: Create and fit NetF extractor
    if verbose:
        print(f"\n4. Creating NetF extractor (n_quantiles={n_quantiles})...")
    
    netf = NetF(
        n_quantiles=n_quantiles,
        max_nodes_for_expensive_measures=200
    )
    
    if verbose:
        print("\n5. Extracting network features...")
        est_time = n_sequences * 0.1  # Rough estimate: 0.1s per sequence
        print(f"   Estimated time: {est_time/60:.1f} minutes")
    
    netf.fit(X_to_process)
    X_features = netf.transform(X_to_process)
    
    if verbose:
        print(f"\n   Features shape: {X_features.shape}")
        print(f"   Non-zero features: {np.sum(np.any(X_features != 0, axis=0))}/{X_features.shape[1]}")
    
    # Step 5: Remove duplicate features
    if remove_duplicates:
        if verbose:
            print("\n6. Removing duplicate features...")
        
        # First get non-zero features
        nonzero_mask = np.any(X_features != 0, axis=0)
        nonzero_indices = np.where(nonzero_mask)[0]
        X_features_nonzero = X_features[:, nonzero_indices]
        
        # Then remove duplicates
        X_features_unique, unique_indices_relative = _remove_duplicate_features(
            X_features_nonzero, verbose=verbose
        )
        
        # Map back to original feature names
        unique_indices = [nonzero_indices[i] for i in unique_indices_relative]
        feature_names = [netf.feature_names[i] for i in unique_indices]
        
        X_features_final = X_features_unique
        
        if verbose:
            print(f"   Final features shape: {X_features_final.shape}")
    else:
        # Use all non-zero features
        nonzero_mask = np.any(X_features != 0, axis=0)
        nonzero_indices = np.where(nonzero_mask)[0]
        X_features_final = X_features[:, nonzero_indices]
        feature_names = [netf.feature_names[i] for i in nonzero_indices]
    
    # Step 6: Add features to DataFrame
    if verbose:
        print("\n7. Adding features to DataFrame...")
    
    n_features = X_features_final.shape[1]
    prefixed_feature_names = [f'NetF_{name}' for name in feature_names]
    
    # Initialize feature columns with NaN
    for feature_name in prefixed_feature_names:
        data[feature_name] = np.nan
    
    # Fill in feature values (starting from sequence_length-1 index)
    start_idx = sequence_length - 1
    for i, feature_name in enumerate(prefixed_feature_names):
        data.iloc[start_idx:start_idx + len(X_features_final), 
                 data.columns.get_loc(feature_name)] = X_features_final[:, i]
    
    if verbose:
        print(f"   Added {n_features} NetF features to DataFrame")
        print(f"   First {sequence_length - 1} rows will be NaN (insufficient data)")
        print("\n" + "=" * 60)
        print("NETF FEATURE EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"\nFeature columns added: {prefixed_feature_names}")
        print(f"Total rows: {len(data)}")
        print(f"Rows with features: {len(data) - (sequence_length - 1)}")
        print(f"\n⚠️  Remember to drop NaN rows: data = data.dropna()")
    
    return data


def analyze_netf_features(data: pd.DataFrame, feature_prefix: str = 'NetF_'):
    """
    Analyze and print statistics about NetF features.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing NetF features
    feature_prefix : str, default='NetF_'
        Prefix used for NetF feature column names
        
    Returns
    -------
    pd.DataFrame
        Summary statistics for NetF features
    """
    feature_cols = [col for col in data.columns if col.startswith(feature_prefix)]
    
    if len(feature_cols) == 0:
        print(f"No features found with prefix '{feature_prefix}'")
        return None
    
    print("=" * 60)
    print("NETF FEATURE ANALYSIS")
    print("=" * 60)
    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"\nFeature statistics:")
    
    stats = data[feature_cols].describe()
    print(stats)
    
    # Correlation analysis
    print(f"\nFeature correlations:")
    corr_matrix = data[feature_cols].corr()
    print(corr_matrix)
    
    # Network type breakdown
    print(f"\nFeature breakdown by network type:")
    for network in ['WNVG', 'WHVG', 'QG']:
        network_features = [col for col in feature_cols if network in col]
        print(f"  {network}: {len(network_features)} features")
    
    return stats