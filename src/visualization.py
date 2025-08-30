"""
Time Series Visualization Utilities for Forecasting

This module provides specialized plotting functions for:
- Visualizing raw time series
- Comparing model predictions against actual values
- Displaying forecast evaluation metrics

Key Features:
- Publication-quality visualizations
- Custom formatting for meteorological data
- Integrated metric display
- Flexible output options
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict, Union, List, Tuple
import numpy as np


def plot_time_series(
    data: pd.DataFrame,
    column: str,
    label: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str = 'royalblue',
    save_path: Optional[str] = None,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """
    Visualize time series data with professional formatting
    
    Args:
        data (pd.DataFrame): Input DataFrame containing the time series
        column (str): Name of the column to plot
        label (str): Legend label for the plotted series
        title (str): Chart title
        x_label (str): X-axis label
        y_label (str): Y-axis label
        color (str): Line color (default: 'royalblue')
        save_path (str): Optional path to save the figure
        return_fig: Whether to return the Figure object
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    fig = plt.figure(figsize=(16, 9))

    data[column].plot(label=label, color=color, ax=plt.gca()) 
    
    # Customize chart appearance
    plt.title(title, fontsize=18, pad=10)  # Added padding for better spacing
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    
    # Configure grid and legend
    plt.grid(linewidth=1.5, alpha=0.7)
    plt.legend(fontsize=12, framealpha=0.9)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Get current figure object to return
    fig = plt.gcf()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(
            save_path, 
            bbox_inches='tight', 
            dpi=300, 
            facecolor='white',  # Ensure white background
            transparent=False
        )
    if return_fig:
        return fig
    else: None


def plot_predictions_vs_actual(
    actual_values: Union[np.ndarray, pd.Series, List[float]],
    predicted_values: Union[np.ndarray, pd.Series, List[float]],
    title: str,
    x_label: str,
    y_label: str,
    metrics: Optional[Dict[str, float]] = None,
    x_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plots actual values vs predicted values with optional metrics display.
    
    Args:
        actual_values (array-like): Array of actual/true values
        predicted_values (array-like): Array of model predictions
        title (str): Plot title
        x_label (str): X-axis label
        y_label (str): Y-axis label
        metrics (dict, optional): Dictionary of metrics to display (e.g., {'MAE': 0.5, 'RMSE': 0.7})
        x_labels (list, optional): Custom x-axis labels (e.g., months)
        figsize (tuple): Figure dimensions
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Create DataFrame for plotting
    results_df = pd.DataFrame({
        'Actual': actual_values,
        'Predicted': predicted_values
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If no x_labels provided, use numeric indices
    if x_labels is None:
        x_labels = range(len(actual_values))
    
    # Plot lines
    ax.plot(x_labels, results_df['Actual'], 
            color='royalblue', 
            linewidth=1.5, 
            label='Actual')
    
    ax.plot(x_labels, results_df['Predicted'], 
            color='#F05C44', 
            linewidth=1.5, 
            label='Predicted')
    
    # Add metrics if provided

    if metrics:
        metric_text = "\n".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
        ax.text(
            0.9,
            0.87,
            metric_text,
            transform=ax.transAxes,
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='gray',
                boxstyle='round'
            ),
            fontsize=10,
            verticalalignment='top'
        )
    # Customize plot
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.grid(linewidth=1, alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()