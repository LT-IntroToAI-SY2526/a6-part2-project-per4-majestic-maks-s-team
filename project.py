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
    
    # Your code here
    #data = pd.read_csv(filename)
    
    # print("=== Price Data ===")
    # print(f"\nFirst 5 rows:")
    # print(data.head())
    
    # print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # print(f"\nBasic statistics:")
    # print(data.describe())
    
    # print(f"\nColumn names: {list(data.columns)}")
    
    # return data 


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
    
    # Your code here
    # Hint: Use subplots like in Part 2!
    #fig, axes = plt.subplots 
    pass


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    # Your code here
    
    pass


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

