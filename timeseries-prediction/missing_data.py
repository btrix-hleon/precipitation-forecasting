"""
Missing Data Analysis Module

This module provides tools for analyzing and handling missing data.

Features:
- DataFrame extension for missing data analysis
- Visualization of missing data patterns
- Time-series aware imputation methods
- Shadow matrix generation for pattern analysis

Author: [Beatriz Hernández León]
Date: [Date]
Version: 1.0
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any

@pd.api.extensions.register_dataframe_accessor("missing")
class MissingDataAnalyzer:
    """Pandas DataFrame extension for comprehensive missing data analysis
    
    Provides methods to:
    - Quantify missing data
    - Analyze patterns of missingness
    - Generate shadow matrices
    - Visualize missing data distributions
    
    Example:
        >>> df = pd.read_csv('precipitation_data.csv')
        >>> df.missing.variable_summary()
    """
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

        """Initialize the missing data analyzer.
        
        Args:
            pandas_obj: DataFrame containing the data
        """
    def _validate(self, obj: pd.DataFrame):
        """Validate input DataFrame structure."""
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if len(obj) == 0:
            raise ValueError("DataFrame cannot be empty")
         
    def count_missing(self) -> int:
        """Count total missing values in DataFrame"""
        return self._obj.isna().sum().sum()
    
    def count_complete(self) -> int:
        """Count total non-missing values in DataFrame"""
        return self._obj.size - self.count_missing()
    
    def variable_summary(self) -> pd.DataFrame:
        """Generate summary statistics for missing values by variable"""
        return (
            self._obj.isnull()
            .pipe(lambda df: (
                df.sum()
                .reset_index(name="n_missing")
                .rename(columns={"index": "variable"})
                .assign(
                    n_cases=len(df),
                    pct_missing=lambda x: x.n_missing / x.n_cases * 100,
                )
                .sort_values("pct_missing", ascending=False)
            ))
        )
    
    def create_shadow_matrix(
        self,
        true_label: str = "Missing",
        false_label: str = "Present",
        only_missing_cols: bool = True
    ) -> pd.DataFrame:
        """Create shadow matrix indicating missing values
        
        Args:
            true_label: Label for missing values (default: 'Missing')
            false_label: Label for present values (default: 'Present')
            only_missing_cols: Whether to include only columns with NAs
            
        Returns:
            DataFrame with same shape as input, where each value indicates
            whether the original was missing
            
        Example:
            >>> shadow = df.missing.create_shadow_matrix()
        """
        shadow_df = (
            self._obj.isna()
            .pipe(lambda df: df[df.columns[df.any()]] if only_missing_cols else df)
            .replace({False: false_label, True: true_label})
            .add_suffix("_NA")
        )
        return shadow_df
    
    def bind_shadow_matrix(
        self,
        true_label: str = "Missing",
        false_label: str = "Present",
        only_missing_cols: bool = True
    ) -> pd.DataFrame:
        """Concatenate original DataFrame with shadow matrix.
        
        Args:
            true_label: Label for missing values
            false_label: Label for present values
            only_missing_cols: Whether to include only columns with NAs
            
        Returns:
            Original DataFrame with shadow matrix columns appended
            
        Example:
            >>> df_with_shadow = df.missing.bind_shadow_matrix()"""
        return pd.concat(
            [
                self._obj,
                self.create_shadow_matrix(true_label, false_label, only_missing_cols)
            ],
            axis=1
        )

def visualize_missing_patterns (
    df: pd.DataFrame,
    plot_type: str,
    x: str,
    value_col: str,
    palette: Optional[Dict[str, str]] = None
) -> None:
    """Visualize missing data patterns in precipitation time series.
        
        Args:
            df: Input DataFrame with precipitation data
            plot_type: Type of plot ('histogram' or 'boxplot')
            x: Column to plot on x-axis (typically time or station ID)
            value_col: Precipitation value column to analyze
            palette: Color dictionary for missing/present values
            
        Raises:
            ValueError: If invalid plot_type is specified
            
        Example:
            >>> visualize_missing_patterns(
                    df, 
                    plot_type='histplot',
                    x='date',
                    value_col='precipitation_mm'
                )
        """
    if palette is None:
        palette = {"Missing": "#F97306", "Present": "royalblue"}
    
    shadow_matrix = df.missing.bind_shadow_matrix()
    
    # Validate plot type selection
    VALID_PLOT_TYPES = {'histplot', 'boxplot'}
    if plot_type not in VALID_PLOT_TYPES:
        raise ValueError(f"plot_type must be one of: {sorted(VALID_PLOT_TYPES)}")

    if plot_type == 'histplot':
        plt.figure(figsize=(10,6))
        sns.histplot(shadow_matrix, 
                     x = x, 
                     hue=value_col+'_NA', 
                     multiple= 'stack',
                     palette=palette,
                     alpha = 0.75)
        plt.title (f'Missing values by {x}')
        plt.xlabel(f'{x}')
        plt.ylabel('Frequency')
        plt.legend(labels = ('Missing Value','Present Value'), frameon=True)

    if plot_type == 'boxplot':

        missing = shadow_matrix.copy()[shadow_matrix[value_col+'_NA']=='Missing']
        plt.figure(figsize=(10,6))
        sns.boxplot(shadow_matrix, 
                    x = value_col+'_NA',
                    y=x,
                    hue=value_col+'_NA',
                    palette=palette)
        sns.swarmplot(missing, x = value_col+'_NA',y=x,color='black')
        plt.title (f'Missing values by {x}')
        plt.xlabel('Value')
        plt.ylabel(f'{x}')
        plt.show()
        
def handle_missing_values(
    df: pd.DataFrame,
    value_col: str,
    method: str = 'interpolate',
    interpolation_method: str = 'time',
    **impute_kwargs: Any
) -> pd.DataFrame:
    """
    Handle missing values in time series data
    
    Args:
        df: Input DataFrame
        value_col: Column to impute
        method: Imputation method ('interpolate', 'fillna', 'drop')
        interpolation_method: Type of interpolation for time series
        **impute_kwargs: Additional arguments for imputation method
        
    Returns:
        DataFrame with handled missing values

    Raises:
        ValueError: If invalid method is specified
    """
    df = df.copy()
    
    if method == 'interpolate':
        df[value_col] = (
            df[value_col]
            .interpolate(method=interpolation_method, **impute_kwargs)
        )
    elif method == 'fillna':
        df[value_col] = df[value_col].fillna(**impute_kwargs)
    elif method == 'drop':
        df = df.dropna(subset=[value_col])
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: 'interpolate', 'fillna', 'drop'"
        )
    
    return df