from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any, Callable, Union, Dict, Tuple
from numpy import ndarray
from pandas.core.arrays import ExtensionArray

DEFAULT_BATCH_SIZE = 20


def _plot_batch(data: pd.DataFrame,
                unique_values: Union[ExtensionArray, ndarray],
                column: str,
                title: str,
                i: int,
                axes: Tuple,
                batch_size: int,
                plot_func: Callable,
                plot_func_args: Dict[str, Any] = {}) -> None:
    # Get batch data
    batch_values = unique_values[i * batch_size: (i + 1) * batch_size]
    batch_data = data[data[column].isin(batch_values)]
    # Set title
    ax = axes[0] if i % 2 == 0 else axes[1]
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(f'{title} - Batch {i + 1}')
    plot_func(data=batch_data, x=column, ax=ax, **plot_func_args)


def _plot_categorical_batched(data: pd.DataFrame,
                              unique_values: Union[ExtensionArray, ndarray],
                              column: str,
                              batch_size: int = DEFAULT_BATCH_SIZE):
    # Define the number of values per batch
    num_batches = int(np.ceil(len(unique_values) / batch_size))

    # Loop through batches and plot two graphs side by side
    for i in range(0, num_batches, 2):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # Create a 1x2 grid for two plots
        # Plot the first
        _plot_batch(data, unique_values, column, f'Count Plot for {column}', i, axes, batch_size, sns.countplot)
        # Plot the second
        _plot_batch(data, unique_values, column, f'Count Plot for {column}', i + 1, axes, batch_size, sns.countplot)
        plt.tight_layout()
        plt.show()


def _plot_numerical_vs_categorical_batched(data: pd.DataFrame,
                                           unique_categories: Union[ExtensionArray, ndarray],
                                           numeric_col: str,
                                           categorical_col: str,
                                           batch_size: int = DEFAULT_BATCH_SIZE):
    # Compute number of batches
    num_batches = int(np.ceil(len(unique_categories) / batch_size))

    for i in range(0, num_batches, 2):
        # Create 1 x 2 grid
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        _plot_batch(data, unique_categories, categorical_col, f'{numeric_col} vs. {categorical_col}', i, axes,
                    batch_size, sns.boxplot, {'y': numeric_col})
        _plot_batch(data, unique_categories, categorical_col, f'{numeric_col} vs. {categorical_col}', i + 1, axes,
                    batch_size, sns.boxplot, {'y': numeric_col})
        plt.tight_layout()
        plt.show()

def _plot_categorical_vs_categorical_batched(data: pd.DataFrame,
                                            unique_categories: Union[ExtensionArray, ndarray],
                                            categorical_col1: str,
                                            categorical_col2: str,
                                            batch_size: int):
    # Compute number of batches
    num_batches = int(np.ceil(len(unique_categories) / batch_size))

    for i in range(0, num_batches, 2):
        # Create 1 x 2 grid
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        _plot_batch(data, unique_categories, categorical_col1, f'{categorical_col1} vs. {categorical_col2}', i, axes,
                    batch_size, sns.countplot, {'hue': categorical_col2} )
        _plot_batch(data, unique_categories, categorical_col1, f'{categorical_col1} vs. {categorical_col2}', i + 1, axes,
                    batch_size, sns.countplot, {'hue': categorical_col2} )
        plt.tight_layout()
        plt.show()


def plot_numerical(data: pd.DataFrame, column: str) -> None:
    """Plots a histogram and boxplot of numerical data

    :param data:
        - The data to be used in plotting
    :param column:
        - The name of the column to be plotted
    :return: None
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    sns.histplot(x=column, data=data, ax=ax1)
    ax1.set_title(f"Histogram of {column}")

    sns.boxplot(x=data[column], data=data, ax=ax2)
    ax2.set_title(f"Box plot of {column}")

    plt.show()


def plot_categorical(data: pd.DataFrame, column: str) -> None:
    unique_values = data[column].unique()

    if len(unique_values) > DEFAULT_BATCH_SIZE:
        _plot_categorical_batched(data, unique_values, column)
    else:
        fig, (ax) = plt.subplots(1, 1, figsize=(10, 6))
        sns.countplot(data, x=column)
        ax.tick_params(axis='x', rotation=90)
        plt.show()


def plot_numerical_vs_numerical(data: pd.DataFrame, numeric_col1: str, numeric_col2: str) -> None:
    sns.scatterplot(x=numeric_col1, y=numeric_col2, data=data)
    plt.title(f'{numeric_col1} vs. {numeric_col2}')
    plt.show()


def plot_numerical_vs_categorical(data: pd.DataFrame, numeric_col: str, categorical_col: str) -> None:
    unique_categories = data[categorical_col].unique()

    if len(unique_categories) > DEFAULT_BATCH_SIZE:
        _plot_numerical_vs_categorical_batched(data, unique_categories, numeric_col, categorical_col)
        return

    sns.boxplot(x=categorical_col, y=numeric_col, data=data)
    plt.title(f'{numeric_col} vs. {categorical_col}')
    plt.show()


def plot_categorical_vs_categorical(data: pd.DataFrame,
                                    categorical_col1: str,
                                    categorical_col2: str,
                                    batch_size: int = 20,
                                    batched: bool = True) -> None:
    if len(data[categorical_col2].unique()) > len(data[categorical_col1].unique()):
        plot_categorical_vs_categorical(data, categorical_col2, categorical_col1, batch_size)
        return

    unique_categories = data[categorical_col1].unique()

    if batched and len(unique_categories) > batch_size:
        _plot_categorical_vs_categorical_batched(data, unique_categories, categorical_col1, categorical_col2, batch_size)
        return

    plt.figure(figsize=(20, 8))  # Set figure dimensions (width=10, height=6)
    sns.countplot(data=data, x=categorical_col1, hue=categorical_col2)
    plt.title(f'{categorical_col1} vs. {categorical_col2}')
    plt.xticks(rotation=90)  # Rotate the x-axis ticks by 45 degrees
    plt.show()
