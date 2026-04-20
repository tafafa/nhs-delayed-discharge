"""
eda.py
------
Exploratory Data Analysis (EDA) functions for NHS Delayed Discharge project.

Produces the visualisations documented in Chapter 4 of the dissertation:
    - Distribution of Total Delayed Bed Days
    - Average Delay by Health Board
    - Temporal Trends (2019–2024)
    - Correlation Heatmap
    - Scatter: PrevMonthDelay vs TotalDelayedBedDays
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_delay_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Histogram of Total Delayed Bed Days distribution.

    Args:
        df: Processed modelling DataFrame.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(df["totaldelayedbeddays"].dropna(), bins=30, color="#4C72B0", edgecolor="white")
    plt.title("Distribution of Total Delayed Bed Days")
    plt.xlabel("Total Delayed Bed Days")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_delay_by_health_board(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Horizontal bar chart of average delayed bed days per Health Board.

    Args:
        df: Processed modelling DataFrame.
        save_path: Optional path to save the figure.
    """
    avg = (
        df.groupby("healthboard")["totaldelayedbeddays"]
        .mean()
        .sort_values()
    )
    plt.figure(figsize=(9, 6))
    avg.plot(kind="barh", color="#55A868", edgecolor="white")
    plt.title("Average Delayed Bed Days by Health Board")
    plt.xlabel("Average Delayed Bed Days")
    plt.ylabel("Health Board")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_temporal_trend(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Line chart of monthly mean delayed bed days over time (2019–2024).

    Args:
        df: Processed modelling DataFrame with 'monthofdelay' column.
        save_path: Optional path to save the figure.
    """
    trend = df.groupby("monthofdelay")["totaldelayedbeddays"].mean()
    plt.figure(figsize=(12, 5))
    trend.plot(color="#C44E52", linewidth=2)
    plt.title("Monthly Trend in Delayed Discharges (2019–2024)")
    plt.xlabel("Month")
    plt.ylabel("Mean Delayed Bed Days")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Annotated heatmap of Pearson correlations between numeric features.

    Args:
        df: Processed modelling DataFrame.
        save_path: Optional path to save the figure.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, annot_kws={"size": 8}
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_prevmonth_vs_total(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Scatter plot showing relationship between PrevMonthDelay and TotalDelayedBedDays.

    Args:
        df: Processed modelling DataFrame.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=(7, 6))
    plt.scatter(
        df["prevmonthdelay"], df["totaldelayedbeddays"],
        alpha=0.5, edgecolors="k", linewidths=0.3, color="#8172B2"
    )
    plt.xlabel("Previous Month Delayed Bed Days")
    plt.ylabel("Total Delayed Bed Days")
    plt.title("Relationship: PrevMonthDelay vs TotalDelayedBedDays")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_descriptive_statistics(df: pd.DataFrame) -> None:
    """Print shape, data types, null counts, and descriptive statistics."""
    print("=== Shape ===")
    print(df.shape)
    print("\n=== Data Types ===")
    print(df.dtypes)
    print("\n=== Null Counts ===")
    print(df.isnull().sum())
    print("\n=== Descriptive Statistics ===")
    print(df.describe().T)
