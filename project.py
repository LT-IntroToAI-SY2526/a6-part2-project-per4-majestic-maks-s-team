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

# Clean currency columns by removing $ and commas, then convert to numeric
currency_cols = [
    "total_sales(mil)",
    "na_sales(mil)",
    "jp_sales(mil)",
    "pal_sales(mil)",
    "other_sales(mil)",
]

for col in currency_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop unused columns (after we've cleaned the sales columns)
columns_to_drop = [
    "title",
    "publisher",
    "developer",
    "na_sales(mil)",
    "jp_sales(mil)",
    "pal_sales(mil)",
    "other_sales(mil)",
]
df = df.drop(columns=columns_to_drop)

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
    
    # Using the globally prepared df
    global df
    data = df

    print("\nFirst 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print("\nBasic statistics:")
    print(data.describe())

    print("\nColumn names:")
    print(list(data.columns))

    print("\nMissing values per column:")
    print(data.isna().sum())

    return data


def visualize_data(data):
    """
    Create 4 scatter plots (each feature vs Total Sales)

    Args:
        data: pandas DataFrame with features and total sales
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)

    # Drop rows with missing values in the columns we need for plotting
    plot_data = data.dropna(subset=["genre_code", "console_code", "critic_score",
                                    "total_sales(mil)", "release_date"])

    # Add release year column
    plot_data = plot_data.copy()
    plot_data["release_year"] = plot_data["release_date"].dt.year

    print("\nRows used for plotting:", len(plot_data))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Feature Relationships vs Total Sales', fontsize=16, fontweight='bold')

    # ---- Plot 1: Genre Code vs Sales ----
    axes[0, 0].scatter(plot_data['genre_code'], plot_data['total_sales(mil)'],
                       color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Genre Code')
    axes[0, 0].set_ylabel('Total Sales (mil)')
    axes[0, 0].set_title('Genre vs Sales')
    axes[0, 0].grid(True, alpha=0.3)

    # ---- Plot 2: Console Code vs Sales ----
    axes[0, 1].scatter(plot_data['console_code'], plot_data['total_sales(mil)'],
                       color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Console Code')
    axes[0, 1].set_ylabel('Total Sales (mil)')
    axes[0, 1].set_title('Console vs Sales')
    axes[0, 1].grid(True, alpha=0.3)

    # ---- Plot 3: Critic Score vs Sales ----
    axes[1, 0].scatter(plot_data['critic_score'], plot_data['total_sales(mil)'],
                       color='red', alpha=0.6)
    axes[1, 0].set_xlabel('Critic Score')
    axes[1, 0].set_ylabel('Total Sales (mil)')
    axes[1, 0].set_title('Critic Score vs Sales')
    axes[1, 0].grid(True, alpha=0.3)

    # ---- Plot 4: Release Year vs Sales ----
    axes[1, 1].scatter(plot_data['release_year'], plot_data['total_sales(mil)'],
                       color='orange', alpha=0.6)
    axes[1, 1].set_xlabel('Release Year')
    axes[1, 1].set_ylabel('Total Sales (mil)')
    axes[1, 1].set_title('Release Year vs Sales')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_vs_sales.png", dpi=300, bbox_inches='tight')
    print("\n✓ Scatter plots saved as 'feature_vs_sales.png'")
    plt.show()

    print("Look for features with a clear upward or downward trend - those are MOST IMPORTANT!")


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)

    # We will use genre_code, console_code, critic_score as features
    feature_columns = ["genre_code", "console_code", "critic_score"]
    target_column = "total_sales(mil)"

    # Drop rows with missing values in features or target
    clean_data = data.dropna(subset=feature_columns + [target_column])
    print("\nRows used for training/testing:", len(clean_data))

    X = clean_data[feature_columns]
    y = clean_data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nShapes:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test, feature_columns


def train_model(X_train, y_train, feature_names):
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
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print equation
    print("\nModel equation:")
    print(f"Sales = {model.intercept_:.4f}", end="")
    for name, coef in zip(feature_names, model.coef_):
        print(f" + ({coef:.4f} * {name})", end="")
    print()

    # Feature importance (by absolute coefficient)
    coef_info = list(zip(feature_names, model.coef_))
    coef_info_sorted = sorted(coef_info, key=lambda x: abs(x[1]), reverse=True)

    print("\nFeature importance (by |coefficient|):")
    for name, coef in coef_info_sorted:
        print(f"{name}: {coef:.4f}")

    return model


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
    
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\nR² score:  {r2:.4f}")
    print(f"RMSE:      {rmse:.4f}")

    # Comparison table
    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    print("\nFirst 10 predictions vs actual:")
    print(comparison.head(10))

    return predictions


def make_prediction(model, feature_names):
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

    # We'll pick a sample based on common values in the dataset
    # Example: an Action game on PS4 with critic score 9.0
    global df

    # Get codes for a specific genre/console if they exist
    sample_genre = "Action"
    sample_console = "PS4"
    sample_critic_score = 9.0

    # Safely get codes; fallback to mode if not present
    if (df["genre"] == sample_genre).any():
        genre_code = df.loc[df["genre"] == sample_genre, "genre_code"].mode()[0]
    else:
        genre_code = df["genre_code"].mode()[0]

    if (df["console"] == sample_console).any():
        console_code = df.loc[df["console"] == sample_console, "console_code"].mode()[0]
    else:
        console_code = df["console_code"].mode()[0]

    sample = pd.DataFrame(
        [[genre_code, console_code, sample_critic_score]],
        columns=feature_names
    )

    prediction = model.predict(sample)[0]

    print("Sample input:")
    print(f"Genre: {sample_genre} (code {genre_code})")
    print(f"Console: {sample_console} (code {console_code})")
    print(f"Critic score: {sample_critic_score}")
    print(f"\nPredicted total sales (mil): {prediction:.2f}")


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test, feature_names = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train, feature_names)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction
    make_prediction(model, feature_names)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")
