"""
Exploratory Data Analysis Module 

This module provides functions for calculating descriptive statistics 
and visualizing distributions of time series data, specifically designed
for preprocessing the data before LSTM modeling.

Key Features:
- Computation of relevant descriptive statistics
- Distribution visualization for precipitation pattern analysis

Author: [Beatriz Hernández León]
Date: [8/8/2025]
Version: 1.0
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional

def calculate_statistics(
    data: Union[pd.DataFrame, pd.Series],
    column: str,
    round_decimals: int = 3
) -> pd.DataFrame:
    """
    Calculate comprehensive descriptive statistics for a numeric column.
    
    Args:
        data: Input DataFrame or Series
        column: Column name to analyze (if input is DataFrame)
        round_decimals: Number of decimal places to round results
        
    Returns:
        DataFrame with statistical measures and values
        
    Raises:
        KeyError: If specified column doesn't exist
        TypeError: If column is non-numeric
        ValueError: If DataFrame is provided but no column name specified
    """
    if isinstance(data, pd.DataFrame) and column is None:
        raise ValueError("Column name must be specified when data is a DataFrame")
    
    if isinstance(data, pd.Series):
        series = data
    else:
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Column '{column}' must be numeric")
    
    stats_dict = {
        'Statistic': [
            'Count',
            'Minimum',
            'Maximum',
            'Mean',
            'Median',
            'Standard Deviation',
            'Range',
            '1st Quartile (Q1)',
            '3rd Quartile (Q3)',
            'Interquartile Range (IQR)',
        ],
        'Value': [
            series.count(),
            series.min(),
            series.max(),
            series.mean(),
            series.median(),
            series.std(),
            series.max() - series.min(),
            series.quantile(q=0.25),
            series.quantile(q=0.75),
            series.quantile(0.75) - series.quantile(0.25),
        ]
    }
    stats_df = pd.DataFrame(stats_dict)
    stats_df['Value'] = stats_df['Value'].round(round_decimals)
    
    return stats_df


def plot_distributions(
    data: Union[pd.DataFrame, pd.Series],
    column: str,
    plot_type: str = 'histogram', 
    color: str = 'royalblue',
    figsize: tuple = (8, 4),
    bins: Union[int, str] = 12,
    kde_bandwidth: float = 1.0,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """
    Visualizes data distribution using a single specified plot type.
    
    Args:
        data: Input pandas DataFrame or Series
        column: Column name to visualize (if input is DataFrame)
        plot_type: Type of distribution plot to generate
        color: Base color for the plot
        figsize: Dimensions of the output figure
        bins: Number of bins for histogram (int or 'auto')
        kde_bandwidth: Bandwidth adjustment factor for KDE plot
        return_fig: Whether to return the Figure object
        
    Returns:
        matplotlib.Figure if return_fig=True, otherwise None
        
    Raises:
        KeyError: When specified column doesn't exist in DataFrame
        TypeError: When column contains non-numeric data
        ValueError: When invalid plot_type is specified
    """
    
    # Validate plot type selection
    VALID_PLOT_TYPES = {'histogram', 'density', 'boxplot'}
    if plot_type not in VALID_PLOT_TYPES:
        raise ValueError(f"plot_type must be one of: {sorted(VALID_PLOT_TYPES)}")

    # Extract target data series
    if isinstance(data, pd.Series):
        series = data
    else:
        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        series = data[column]
    
    # Validate numeric data
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Column '{column}' must contain numeric data")

    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate specified plot type
    if plot_type == 'histogram':
        sns.histplot(
            x=series,
            color=color,
            bins=bins,
            ax=ax,
            edgecolor='white',
            linewidth=0.5  
        )
        ax.set_title('Distribution Histogram', fontweight='bold')
        plt.grid(linewidth=1, alpha=0.7)

    elif plot_type == 'density':
        sns.kdeplot(
            x=series,
            color=color,
            fill=True,
            bw_adjust=kde_bandwidth,
            ax=ax,
            alpha=0.7  # Added for better visual appeal
        )
        ax.set_title('Probability Density', fontweight='bold')
        plt.grid(linewidth=1, alpha=0.7)

    elif plot_type == 'boxplot':
        sns.boxplot(
            x=series,
            color=color,
            ax=ax,
            showfliers=True,
            showmeans=True,
            meanprops={
                'marker': 'o', 
                'markerfacecolor': 'white',
                'markeredgecolor': 'black'
            }
        )
        ax.set_title('Boxplot Distribution', fontweight='bold')
        plt.grid(linewidth=1, alpha=0.7)

    # Common formatting
    ax.set_xlabel(column.capitalize())
    plt.tight_layout()
    
    # Return figure object if requested
    if return_fig:
        return fig
    else: return None