import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data

def basic_eda(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSample rows:\n", df.head())

def plot_class_distribution(train_df):
    """
    Plots the distribution of fraud vs non-fraud transactions.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x="is_fraud", data=train_df, palette=["#1f77b4", "#ff7f0e"])
    plt.title("Class Distribution: Fraud vs Non-Fraud")
    plt.xlabel("Transaction Type (0: Legit, 1: Fraud)")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=["Legit", "Fraud"])
    plt.show()

def plot_transaction_amount(train_df):
    """
    Plots the distribution of transaction amounts.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data=train_df, x="amt", hue="is_fraud", bins=50, kde=True, palette=["#1f77b4", "#ff7f0e"])
    plt.xlim(0, 500)  # Limiting to better visualize low-value transactions
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount ($)")
    plt.ylabel("Density")
    plt.show()

def plot_fraud_by_category(train_df):
    """
    Plots fraud distribution
    """
    fraud_counts = train_df[train_df["is_fraud"] == 1]["category"].value_counts().head(10)
    
    plt.figure(figsize=(10, 5))
    fraud_counts.plot(kind="bar", color="#ff7f0e")
    plt.title("Top 10 Merchant Categories for Fraudulent Transactions")
    plt.xlabel("Merchant Category")
    plt.ylabel("Fraud Count")
    plt.xticks(rotation=45)
    plt.show()

def plot_correlation_heatmap(train_df):
    """
    Plots a correlation heatmap
    """
    plt.figure(figsize=(10, 6))
    corr = train_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def visualize_data():
    """
    Load dataset
    """
    train_path = "./fraudTrain.csv"
    test_path = "./fraudTest.csv"
    train_df, _ = load_data(train_path, test_path)
    
    basic_eda(train_df)
    plot_class_distribution(train_df)
    plot_transaction_amount(train_df)
    plot_fraud_by_category(train_df)
    plot_correlation_heatmap(train_df)

if __name__ == "__main__":
    visualize_data()

