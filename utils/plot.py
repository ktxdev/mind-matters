from collections import defaultdict
from typing import Union, Callable, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ndarray
from pandas.api.extensions import ExtensionArray

DEFAULT_BATCH_SIZE = 20
DEFAULT_FIGSIZE = (10, 6)


class DataVisualizer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataVisualizer class with a DataFrame.

        Parameters:
        - data: pd.DataFrame
            The input data for visualization.
        """
        self.data = data

    @staticmethod
    def _batch_data(data: pd.DataFrame,
                    unique_values: Union[ExtensionArray, ndarray],
                    column: str,
                    batch_size: int,
                    i: int) -> pd.DataFrame:
        """
        Helper function to retrieve a batch of data based on unique values.
        """
        batch_values = unique_values[i * batch_size: (i + 1) * batch_size]
        return data[data[column].isin(batch_values)]

    @staticmethod
    def _plot_and_customize(ax: plt.Axes, title: str, xlabel: str, ylabel: str, rotation: int = 0) -> None:
        """
        Helper function to add customizations to plots.
        """
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=rotation)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    def _plot_batch(self,
                    unique_values: Union[ExtensionArray, ndarray],
                    column: str,
                    title: str,
                    i: int,
                    axes: Tuple[plt.Axes, plt.Axes],
                    batch_size: int,
                    plot_func: Callable,
                    plot_func_args=None) -> None:
        """
        Helper function to plot batched data.
        """
        if plot_func_args is None:
            plot_func_args = {}
        batch_data = self._batch_data(self.data, unique_values, column, batch_size, i)
        ax = axes[0] if i % 2 == 0 else axes[1]
        plot_func(data=batch_data, x=column, ax=ax, **plot_func_args)
        self._plot_and_customize(ax, f"{title} - Batch {i + 1}", column, "Count", rotation=90)

    def _plot_batched(self, column: str, plot_func: Callable, batch_size: int = DEFAULT_BATCH_SIZE,
                      plot_func_args=None) -> None:
        """
        Generalized function for plotting batched data.
        """
        if plot_func_args is None:
            plot_func_args = {}

        unique_values = self.data[column].unique()
        num_batches = int(np.ceil(len(unique_values) / batch_size))

        for i in range(0, num_batches, 2):
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            self._plot_batch(unique_values, column, f'Count Plot for {column}', i, axes, batch_size, plot_func, plot_func_args)
            if i + 1 < num_batches:  # Handle odd batch cases
                self._plot_batch(unique_values, column, f'Count Plot for {column}', i + 1, axes, batch_size, plot_func,
                                 plot_func_args)
            plt.tight_layout()
            plt.show()

    def _plot_categorical(self, column: str, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """
        Plot a count plot for a categorical column, with batching if necessary.
        """
        unique_values = self.data[column].unique()

        if len(unique_values) > batch_size:
            self._plot_batched(column, sns.countplot, batch_size)
        else:
            plt.figure(figsize=DEFAULT_FIGSIZE)
            sns.countplot(data=self.data, x=column)
            plt.title(f"Count Plot for {column}")
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    def _plot_numerical(self, column: str) -> None:
        """
        Plot a histogram and boxplot for a numerical column.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        sns.histplot(data=self.data, x=column, ax=axes[0], kde=True)
        sns.boxplot(data=self.data, x=column, ax=axes[1])
        self._plot_and_customize(axes[0], f"Histogram of {column}", column, "Frequency")
        self._plot_and_customize(axes[1], f"Boxplot of {column}", column, "Value")
        plt.tight_layout()
        plt.show()

    def _plot_numerical_vs_categorical(self, numeric_col: str, categorical_col: str,
                                       batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """
        Plot a boxplot of a numerical column against a categorical column.
        """
        unique_categories = self.data[categorical_col].unique()
        if len(unique_categories) > batch_size:
            self._plot_batched(categorical_col, sns.boxplot, batch_size=batch_size, plot_func_args={"y": numeric_col})
        else:
            plt.figure(figsize=DEFAULT_FIGSIZE)
            sns.boxplot(data=self.data, x=categorical_col, y=numeric_col)
            plt.title(f"{numeric_col} vs. {categorical_col}")
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    def _plot_numerical_vs_numerical(self, numeric_col1: str, numeric_col2: str) -> None:
        """
        Plot a scatter plot for two numerical columns.
        """
        plt.figure(figsize=DEFAULT_FIGSIZE)
        sns.scatterplot(data=self.data, x=numeric_col1, y=numeric_col2)
        plt.title(f"{numeric_col1} vs. {numeric_col2}")
        plt.xlabel(numeric_col1)
        plt.ylabel(numeric_col2)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

    def _plot_categorical_vs_categorical(self, categorical_col1: str, categorical_col2: str,
                                         batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """
        Plot a count plot for two categorical columns with batching if necessary.
        """
        unique_categories = self.data[categorical_col1].unique()
        if len(unique_categories) > batch_size:
            self._plot_batched(categorical_col1, sns.countplot, batch_size=batch_size,
                               plot_func_args={"hue": categorical_col2})
        else:
            plt.figure(figsize=(18, 8))
            sns.countplot(data=self.data, x=categorical_col1, hue=categorical_col2)
            plt.title(f"{categorical_col1} vs. {categorical_col2}")
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    def _check_column_exists(self, column: str) -> bool:
        if column not in self.data.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            return False

        return True

    def plot(self, column_name1: str, column_name2: str = None) -> None:
        # Check if column exists
        self._check_column_exists(column_name1)

        # Get the data type of the column
        col1_dtype = self.data[column_name1].dtype

        # Plot based on data type
        if column_name2 and self._check_column_exists(column_name2):
            col2_dtype = self.data[column_name2].dtype

            if pd.api.types.is_numeric_dtype(col1_dtype) and pd.api.types.is_numeric_dtype(col2_dtype):
                self._plot_numerical_vs_numerical(column_name1, column_name2)
            elif pd.api.types.is_numeric_dtype(col1_dtype) and (
                    isinstance(col2_dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(col2_dtype)):
                self._plot_numerical_vs_categorical(column_name1, column_name2)
            elif pd.api.types.is_numeric_dtype(col2_dtype) and (
                    isinstance(col1_dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(col1_dtype)):
                self._plot_numerical_vs_categorical(column_name2, column_name1)
            elif (isinstance(col1_dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(col1_dtype)) and (
                    isinstance(col2_dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(col2_dtype)):
                self._plot_categorical_vs_categorical(column_name1, column_name2)
        else:
            if pd.api.types.is_numeric_dtype(col1_dtype):
                self._plot_numerical(column_name1)
            elif isinstance(col1_dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(col1_dtype):
                self._plot_categorical(column_name1)
            else:
                print(f"The column '{column_name1}' has a data type not suitable for plotting.")
