# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def basic_eda(df: pd.DataFrame) -> None:
    """
    Display basic dataset info.
    """
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSample rows:\n", df.head())

def plot_fraud_distribution(df: pd.DataFrame) -> None:
    """
    Show fraud vs. non-fraud class distribution.
    """
    plt.figure(figsize=(5,4))
    sns.countplot(data=df, x='is_fraud', palette='viridis')
    plt.title("Fraud vs. Non-Fraud Transactions")
    plt.show()
