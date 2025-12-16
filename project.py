"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Patrick Nyman
- Alex Zhou
- Max Pryzbyl

Dataset: VideoGames_Sales.csv
Predicting: Game sales
Features: Genre, console, critic score
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

DATA_FILE = 'VideoGames_Sales.csv'

# Read CSV into a DataFrame
df = pd.read_csv(DATA_FILE, on_bad_lines="skip")

# FIX: remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

columns_to_drop = [
    "title",
    "publisher",
    "developer",
    "na_sales(mil)",
    "jp_sales(mil)",
    "pal_sales(mil)",
    "other_sales(mil)"
]

df = df.drop(columns=columns_to_drop)

# Convert total sales to float safely
df["total_sales(mil)"] = pd.to_numeric(
    df["total_sales(mil)"], errors="coerce"
)

# Convert other values
df["critic_score"] = pd.to_numeric(df["critic_score"], errors="coerce")
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["console_code"] = df["console"].astype("category").cat.codes
df["genre_code"] = df["genre"].astype("category").cat.codes


def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    # Load the already dropped version of the CSV file
    global df
    data = df

    print("\nFirst 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print("\nBasic statistics:")
    print(data.describe())

    print("\nColumn names:")
    print(list(data.columns))

    return data

def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)

    # Use the cleaned column names
    feature_columns = ["critic_score", "console_code", "genre_code", "release_date"]
    target_column = "total_sales(mil)"

    plt.figure(figsize=(14, 10))

    for i, feature in enumerate(feature_columns):
        plt.subplot(2, 2, i + 1)
        plt.scatter(data[feature], data[target_column], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel(target_column)
        plt.title(f"{feature} vs {target_column}")

    plt.tight_layout()
    plt.savefig("feature_vs_sales.png")
    plt.show()

    print("Scatter plots saved as feature_vs_sales.png")
    print("Look for features with a clear upward or downward trend - those are MOST IMPORTANT!")

def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test

    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    """
    # Your code here
    feature_columns = ["price", "platform", "genre", "review_score", "release_date"]
    target_column = ["sales"]

    X = data = [feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
       # Print sizes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

def train_model(X_train, y_train):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # Your code here
    
    pass


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Your code here
    
    pass


def make_prediction(model):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    pass


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

