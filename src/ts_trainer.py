"""
 Utilities for Time Series Forecasting

This module provides tools for:
- Sequence generation for LSTM models
- Time-series cross validation
- Visualization of training progress and data splits

Key Features:
- Flexible sequence generation (sliding window or block approach)
- Temporal-aware cross validation
- Custom visualization of loss functions and data splits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List, Any, Optional

def create_sequences(
    time_series: np.ndarray,
    input_length: int,
    output_length: int,
    sliding_window: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Generate input-output sequences for time series models.
    
    Args:
        time_series: NumPy array containing the time series data
        input_length: Length of the input sequence (number of time steps)
        output_length: Length of the output sequence to predict
        sliding_window: If True, uses sliding window approach. If False, uses contiguous blocks.
        
    Returns:
        Tuple (x, y) where:
        - x: Input sequences (n_samples, input_length)
        - y: Output sequences (n_samples, output_length)
    """
    if sliding_window:
        # Sliding window approach (overlapping sequences)
        x = np.array(np.lib.stride_tricks.sliding_window_view(
            time_series[:(len(time_series)-output_length)],
            window_shape=input_length,axis=0))
        
        y = np.array(np.lib.stride_tricks.sliding_window_view(
            time_series[input_length:],
            window_shape=output_length,axis=0))
    else:
        # Contiguous blocks approach (non-overlapping sequences)
        n_samples = len(time_series) // (input_length + output_length)
        x = time_series[:n_samples*input_length].reshape(n_samples, input_length)
        y = time_series[input_length:(n_samples*input_length)+output_length].reshape(n_samples, output_length)
    
    return x, y

def time_series_cross_validate(
    model: Any,
    data: np.ndarray,
    index: np.ndarray,
    input_length: int,
    output_length: int,
    n_splits: int,
    val_size: int,
    batch_size: int,
    epochs: int,
    verbose: int = 0,
    sliding_window: bool = True
) -> Tuple[List[Any], TimeSeriesSplit]:
    """
    Perform time series cross-validation for a given model.
    
    Args:
        model: Model to train (must implement fit() method)
        data: Array containing the preprocessed data for training and validation
        index: Array containg the index of the preprocessed data for training and validation
        input_length: Length of input sequences
        output_length: Length of output sequences to predict
        n_splits: Number of splits for cross-validation
        val_size: Size of the validation set for each split
        batch_size: Batch size for training
        epochs: Number of training epochs
        verbose: Verbosity level (0 = silent, 1 = progress)
        sliding_window: Whether to use sliding window approach for sequence generation
        
    Returns:
        Tuple: (training_histories, splitter) where:
        - training_histories: List of training history objects from each fold
        - splitter: TimeSeriesSplit object used for the splits
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=val_size)
    training_histories = []

    for train_index, val_index in tscv.split(index):
        # Split the data
        train_data, val_data = data[train_index], data[val_index]

        # Generate sequences
        x_train, y_train = create_sequences (train_data, input_length,output_length, sliding_window)
        x_val, y_val = create_sequences (val_data, input_length,output_length, sliding_window)
        
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=verbose
        )
        
        training_histories.append(history)
    
    return training_histories, tscv

def plot_time_series_splits(
    tscv: TimeSeriesSplit,
    time_series_data: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    x_limits: Optional[Tuple[int, int]] = None
) -> None:
    """
    Visualize the time series splitting behavior.
    
    Args:
        splitter: TimeSeriesSplit object used for the splits
        time_series_data: Data used in the splitting
        figsize: Figure dimensions (width, height) in inches
        x_limits: Tuple (x_min, x_max) for x-axis limits (optional)
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each fold
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(time_series_data)):
        ax.fill_between(train_idx, fold_idx+0.8, fold_idx+1.3, 
                        color='royalblue', edgecolor="white", label="Train" if fold_idx == 0 else "")
        ax.fill_between(val_idx, fold_idx+0.8, fold_idx+1.3, 
                        color='#F97306', edgecolor="white", label="Validation" if fold_idx == 0 else "")
    
    # Configure plot
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Fold number")
    ax.set_title("Time Series Cross-Validation Splits")
    ax.yaxis.set_major_locator(MultipleLocator(1))
    
    # Create legend
    legend_elements = [
        Patch(facecolor='royalblue', label="Training data"),
        Patch(facecolor='#F97306', label="Validation data")
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True)
    
    # Set x-axis limits
    if x_limits is not None:
        ax.set_xlim(x_limits)
    else:
        ax.set_xlim(0, len(time_series_data))
    
    plt.tight_layout()
    plt.show()

def plot_loss_fuction (
    histories: List[Any],
    folds: bool = False
) -> None:
    """Visualize training and validation loss across folds and epochs.
    
    Provides two visualization modes:
    1. Combined view showing all folds (folds=False)
    2. Individual plots for each fold (folds=True)

    Args:
        histories: List of Keras History objects from cross-validation
        folds: Whether to plot each fold separately (default: False)
    """
    # Extract loss metrics
    loss = []
    val_loss = []
    for history in histories: 
       loss.append(history.history['loss'])
       val_loss.append(history.history['val_loss'])
    
    epochs = len(loss[0]) # Epochs per fold
    n_folds = len (loss) # Total number of folds

    x_values = []
    extended_loss = []
    extended_val_loss = []
    
    for fold in range (n_folds):
        x_cycle = np.arange(epochs) + (fold * epochs)
        x_values.extend(x_cycle)
    
    if folds == True:
         # Plot each fold separately
        for i, history in enumerate (histories):
            fig, ax = plt.subplots(figsize=(18, 6))
            ax.plot (history.history['loss'], "#3458BB", label = 'Train loss')
            ax.plot (history.history['val_loss'], "#F21216", label = 'Validation loss')
            
            # Configure x-axis
            ax.set_xticks(range(0,epochs))
            ax.set_xticklabels([str(epochs+1) for epochs in range(0,epochs) ])
            ax.set_xticks([0]+ [i for i in range(int((epochs/10)-1),epochs+1,int(epochs/10))])

            plt.title (f'Loss Function in fold {i+1}', pad = 15, fontsize = 14)
            plt.xlabel ('Epochs', fontsize = 12)
            plt.ylabel ('Loss', fontsize = 12)
            ax.yaxis.set_major_locator(MultipleLocator (0.025))
            ax.legend(loc = 'upper right', frameon = True )
            plt.grid (True, linestyle = '--', alpha = 1)

    else:
        # Combine all folds into single plot
        extended_loss = np.array(loss).flatten()
        extended_val_loss = np.array(val_loss).flatten()

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot (x_values, extended_loss, "#3458BB", label = 'Train loss', alpha = 1, linewidth=1)
        ax.plot (x_values, extended_val_loss, "#F21216", label = 'Validation loss', alpha = 1,linewidth=1)

        # Configure x-axis
        ax.set_xticks(x_values)
        ax.set_xticklabels([str((i % epochs)+1) for i in x_values])
        ax.set_xticks([0]+[((epochs*i)/2)-1 for i in range (1,((n_folds*2)+1))])

        # Add fold separation lines
        for fold in range (1,fold + 1):
            ax.axvline(x=fold*epochs,color = 'gray', linestyle = '--', linewidth = 1, alpha = 0.7)
        
        for fold in range (fold+1):
            ax.text(fold*epochs + (epochs/2)-(epochs/12),
                    (max(max(extended_loss),
                    max(extended_val_loss))-0.2),
                    f'Fold {fold+1}',fontsize = 12,alpha = 0.5)

        plt.title ('Loss Function', pad = 15, fontsize = 14)
        plt.xlabel ('Epochs', fontsize = 12)
        plt.ylabel ('Loss', fontsize = 12)
        ax.yaxis.set_major_locator(MultipleLocator (0.3))
        ax.legend(loc = 'best', frameon = True )
        plt.grid (True, linestyle = '--', alpha = 1)
        plt.show