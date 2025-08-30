"""
Data Loading and Preprocessing Module for Forecasting

This module handles:
- Loading and transforming raw precipitation data
- Creating proper datetime indices
- Scaling data for ANN models
- Inverse scaling of predictions

Key Features:
- Robust loading of various CSV formats
- Automatic month name translation (Spanish-English)
- Time-series aware scaling
- Custom inverse scaling with metric preservation

Author: [Beatriz Hernández León]
Date: [8/8/2025]
Version: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Union

def load_data (
    csv_path: str, 
    skiprows: int = 1, 
    encoding: str = 'iso-8859-1',
    year_col: str = 'Años',
    var_name='Mes',
    value_name: str = 'precipitation'
) -> pd.DataFrame:
    """
    Load and transform data from wide to long format with datetime index
    
    Args:
        csv_path: Path to CSV file
        skiprows: Number of rows to skip in CSV
        encoding: File encoding
        year_col: Name of year column in raw data
        value_name: Name for melted value column
        value_name: Name for precipitation values column (default: 'precipitation')

    Returns:
        DataFrame with:
        - Datetime index (first day of each month)
        - Columns: [year, month_name, month, precipitation]
        - Sorted chronologically
        
    Raises:
        FileNotFoundError: If CSV path is invalid
        KeyError: If required columns are missing
        ValueError: If date parsing fails
    """
    try:
        df = pd.read_csv(csv_path, skiprows=skiprows, encoding=encoding)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {csv_path}")
    
    # Validate required columns exist
    if year_col not in df.columns:
        raise KeyError(f"Year column '{year_col}' not found in DataFrame")
    
    # Transform to long format
    df_long = df.melt(
        id_vars=[year_col], 
        var_name=var_name, 
        value_name=value_name
    )

    # Translate months from Spanish to English
    meses_into_months = {
        'Enero':'Jan',
        'Febrero':'Feb',
        'Marzo':'Mar',
        'Abril':'Apr',
        'Mayo':'May',
        'Junio':'Jun',
        'Julio':'Jul',
        'Agosto':'Aug',
        'Septiembre':'Sep',
        'Octubre':'Oct',
        'Noviembre':'Nov',
        'Diciembre':'Dec'
    }

    df_long['Mes'] = df_long['Mes'].map(meses_into_months)
    df_long.rename(columns={'Mes': 'month_name'}, inplace=True)
    
    # Map month names to numbers
    month_to_num = {month: i+1 for i, month in enumerate(meses_into_months.values())}
    
    # Create datetime index
    df_long['month'] = df_long['month_name'].map(month_to_num)
    df_long.rename(columns={year_col: 'year'}, inplace=True)
    df_long['date'] = pd.to_datetime(
    df_long[['year', 'month']].assign(day=1))
    
    # Handle missing/invalid dates
    if df_long['date'].isnull().any():
        invalid_dates = df_long[df_long['date'].isnull()]
        print(f"Warning: {len(invalid_dates)} rows with invalid dates dropped")
    
    # Sort and set index
    df_long = df_long.sort_values('date').set_index('date')
    
    return df_long

def scale_data(
    train_data: Union[np.ndarray, pd.Series],
    test_data: Union[np.ndarray, pd.Series],
    feature_range: Tuple[float, float] = (-1, 1),
) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series], MinMaxScaler]:
    """
    Scale data using MinMax scaling
    
    Args:
        train_data: Training data to fit scaler
        test_data: Test data to transform
        feature_range: Desired range of transformed data (default: (-1, 1))

    Returns:
        Tuple of (scaled_train, scaled_test, scaler)

    Raises:
        ValueError: If input data contains NaN/inf values
        TypeError: If input data cannot be converted to numpy array

    """
    # Convert to numpy arrays
    try:
        train_data = np.asarray(train_data, dtype=np.float32)
        test_data = np.asarray(test_data, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise TypeError("Input data must be convertible to numpy array") from e
    
    # Check for missing/infinite values
    if np.any(~np.isfinite(train_data)) or np.any(~np.isfinite(test_data)):
        raise ValueError("Input data contains NaN or infinite values")
    
    # Initialize and fit scaler
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(train_data.reshape(-1, 1))
    return (scaler.transform(train_data.reshape(-1, 1)).reshape(len(train_data,)),
            scaler.transform(test_data.reshape(-1, 1)).reshape(len(test_data,)),
            scaler)

def inverse_scaler(    
    scaler: MinMaxScaler,
    scaled_data: Union[np.ndarray, pd.Series],
    metrics: bool = False
) -> np.ndarray:
    """Reverse scaling transformation.
    
    Provides two modes:
    - Standard inverse transform (metrics=False)
    - Metrics inverse transform (metrics=True) for metrics in the original range of data
    
    Args:
        scaler: Fitted MinMaxScaler instance
        scaled_data: Scaled data to transform back
        metrics: Whether to use metric transform (default: False)

    Returns:
        Array with values in original range of data

    Raises:
        TypeError: If input data cannot be converted to numpy array

    Example:
        >>> # For model predictions
        >>> original_scale = inverse_scaler(scaler, predictions)
        
        >>> # For error metrics
        >>> metric_errors = inverse_scaler(scaler, errors, metrics=True)
    """
    try:
        scaled_data = np.asarray(scaled_data, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise TypeError("Input data must be convertible to numpy array") from e
    
    if metrics == False:
        return scaler.inverse_transform(scaled_data.reshape (-1,1)).reshape(scaled_data.shape)
    
    if metrics == True:
        original_range = scaler.data_max_ - scaler.data_min_
        feature_range = scaler.feature_range[1] - scaler.feature_range[0]
        return scaled_data * (original_range/feature_range)


