import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(path: str, show_plots: bool = True):
    """
    Perform exploratory data analysis on a CSV file.
    
    Args:
        path (str): Path to the CSV file
        show_plots (bool): Whether to display distribution plots
    """
    # Read the CSV file
    df = pd.read_csv(path)

    # Display basic information about the dataset
    print(f"\nAnalyzing {path}")
    print("\nDataset Info:")
    print(df.info())

    # Display first few rows
    print("\nFirst few rows:")
    print(df.head())

    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Display unique values in each column
    print("\nUnique Values per Column:")
    for column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts().head())

    if show_plots:
        # Create some basic visualizations
        plt.figure(figsize=(12, 6))
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for i, column in enumerate(numeric_columns, 1):
            plt.subplot(2, 3, i)
            sns.histplot(data=df, x=column)
            plt.title(f'Distribution of {column}')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return df


if __name__ == "__main__":
    analyze_dataset("/share/garg/scholar_inbox_datasets/data/rated_papers.csv")