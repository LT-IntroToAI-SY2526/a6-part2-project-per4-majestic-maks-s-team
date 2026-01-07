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


OPTIMIZED VERSION - Reduced memory usage for Chromebook compatibility
"""


import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save memory
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import gc  # Garbage collection to free memory


DATA_FILE = 'VideoGames_Sales.csv'


def load_and_clean_data(filename):
    """Load and clean data with memory optimization"""
    print("=" * 70)
    print("LOADING AND CLEANING DATA")
    print("=" * 70)
   
    # Read only needed columns to save memory
    usecols = [
        'genre', 'console', 'critic_score', 'total_sales(mil)',
        'release_date'
    ]
   
    try:
        df = pd.read_csv(filename, on_bad_lines="skip", usecols=usecols)
    except ValueError:
        # If column names have spaces, read all and filter
        df = pd.read_csv(filename, on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        df = df[usecols]
   
    # Strip column names
    df.columns = df.columns.str.strip()
   
    # Clean total_sales column
    df['total_sales(mil)'] = (
        df['total_sales(mil)']
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df['total_sales(mil)'] = pd.to_numeric(df['total_sales(mil)'], errors='coerce')
   
    # Convert other columns
    df['critic_score'] = pd.to_numeric(df['critic_score'], errors='coerce')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
   
    # Create categorical codes efficiently
    df['console_code'] = df['console'].astype('category').cat.codes
    df['genre_code'] = df['genre'].astype('category').cat.codes
   
    # Store mappings before dropping
    genre_map = dict(enumerate(df['genre'].astype('category').cat.categories))
    console_map = dict(enumerate(df['console'].astype('category').cat.categories))
   
    print(f"\nLoaded {len(df)} rows")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
   
    return df, genre_map, console_map




def explore_data(df):
    """Print basic information about the dataset"""
    print("\n" + "=" * 70)
    print("EXPLORING DATA")
    print("=" * 70)
   
    print("\nFirst 5 rows:")
    print(df.head())
   
    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
   
    print("\nMissing values per column:")
    print(df.isna().sum())
   
    print("\nBasic statistics:")
    print(df[['critic_score', 'total_sales(mil)']].describe())




def visualize_data(df):
    """Create scatter plots with memory optimization"""
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
   
    # Sample data if too large (use max 5000 points for plotting)
    plot_df = df.dropna(subset=['genre_code', 'console_code', 'critic_score',
                                'total_sales(mil)', 'release_date'])
   
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(n=5000, random_state=42)
        print(f"\nSampled 5000 rows for visualization (from {len(df)} total)")
    else:
        print(f"\nUsing all {len(plot_df)} valid rows for visualization")
   
    plot_df = plot_df.copy()
    plot_df['release_year'] = plot_df['release_date'].dt.year
   
    # Create figure with smaller size to save memory
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Feature Relationships vs Total Sales', fontsize=14, fontweight='bold')
   
    # Plot 1: Genre Code vs Sales
    axes[0, 0].scatter(plot_df['genre_code'], plot_df['total_sales(mil)'],
                       color='blue', alpha=0.5, s=10)
    axes[0, 0].set_xlabel('Genre Code', fontsize=10)
    axes[0, 0].set_ylabel('Total Sales (mil)', fontsize=10)
    axes[0, 0].set_title('Genre vs Sales', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
   
    # Plot 2: Console Code vs Sales
    axes[0, 1].scatter(plot_df['console_code'], plot_df['total_sales(mil)'],
                       color='green', alpha=0.5, s=10)
    axes[0, 1].set_xlabel('Console Code', fontsize=10)
    axes[0, 1].set_ylabel('Total Sales (mil)', fontsize=10)
    axes[0, 1].set_title('Console vs Sales', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
   
    # Plot 3: Critic Score vs Sales
    axes[1, 0].scatter(plot_df['critic_score'], plot_df['total_sales(mil)'],
                       color='red', alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Critic Score', fontsize=10)
    axes[1, 0].set_ylabel('Total Sales (mil)', fontsize=10)
    axes[1, 0].set_title('Critic Score vs Sales', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
   
    # Plot 4: Release Year vs Sales
    axes[1, 1].scatter(plot_df['release_year'], plot_df['total_sales(mil)'],
                       color='orange', alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Release Year', fontsize=10)
    axes[1, 1].set_ylabel('Total Sales (mil)', fontsize=10)
    axes[1, 1].set_title('Release Year vs Sales', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.savefig("feature_vs_sales.png", dpi=150, bbox_inches='tight')
    print("\n✓ Scatter plots saved as 'feature_vs_sales.png'")
    plt.close(fig)  # Close figure to free memory
   
    # Free memory
    del plot_df
    gc.collect()




def prepare_and_split_data(df):
    """Prepare features and split data"""
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
   
    feature_columns = ['genre_code', 'console_code', 'critic_score']
    target_column = 'total_sales(mil)'
   
    # Drop rows with missing values
    clean_df = df.dropna(subset=feature_columns + [target_column])
    print(f"\nRows used for training/testing: {len(clean_df)}")
   
    X = clean_df[feature_columns].values  # Use .values to get numpy array
    y = clean_df[target_column].values
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
   
    print("\nShapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
   
    return X_train, X_test, y_train, y_test, feature_columns




def train_model(X_train, y_train, feature_names):
    """Train linear regression model"""
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
   
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n✓ Model training complete")
   
    # Print equation
    print("\nModel equation:")
    print(f"Sales = {model.intercept_:.4f}", end="")
    for name, coef in zip(feature_names, model.coef_):
        print(f" + ({coef:.4f} * {name})", end="")
    print()
   
    # Feature importance
    coef_info = sorted(zip(feature_names, model.coef_),
                      key=lambda x: abs(x[1]), reverse=True)
   
    print("\nFeature importance (by |coefficient|):")
    for name, coef in coef_info:
        print(f"  {name}: {coef:.4f}")
   
    return model




def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
   
    predictions = model.predict(X_test)
   
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
   
    print(f"\nR² score:  {r2:.4f}")
    print(f"RMSE:      {rmse:.4f}")
   
    # Show first 10 predictions
    print("\nFirst 10 predictions vs actual:")
    print(f"{'Actual':>10} {'Predicted':>10} {'Difference':>10}")
    print("-" * 32)
    for i in range(min(10, len(y_test))):
        diff = predictions[i] - y_test[i]
        print(f"{y_test[i]:>10.2f} {predictions[i]:>10.2f} {diff:>10.2f}")
   
    return predictions




def make_prediction(model, feature_names, genre_map, console_map):
    """Make a sample prediction"""
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
   
    # Use middle values as example
    genre_code = len(genre_map) // 2
    console_code = len(console_map) // 2
    critic_score = 8.5
   
    sample = np.array([[genre_code, console_code, critic_score]])
    prediction = model.predict(sample)[0]
   
    genre_name = genre_map.get(genre_code, f"Genre {genre_code}")
    console_name = console_map.get(console_code, f"Console {console_code}")
   
    print("\nSample input:")
    print(f"  Genre: {genre_name} (code {genre_code})")
    print(f"  Console: {console_name} (code {console_code})")
    print(f"  Critic score: {critic_score}")
    print(f"\nPredicted total sales: ${prediction:.2f} million")




if __name__ == "__main__":
    print("Starting optimized video game sales prediction model...")
    print("(Designed for low-memory environments like Chromebooks)\n")
   
    # Step 1: Load and clean
    df, genre_map, console_map = load_and_clean_data(DATA_FILE)
   
    # Step 2: Explore
    explore_data(df)
   
    # Step 3: Visualize
    visualize_data(df)
   
    # Step 4: Prepare and split
    X_train, X_test, y_train, y_test, feature_names = prepare_and_split_data(df)
   
    # Step 5: Train
    model = train_model(X_train, y_train, feature_names)
   
    # Step 6: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
   
    # Step 7: Make prediction
    make_prediction(model, feature_names, genre_map, console_map)
   
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nMemory-saving techniques used:")
    print("  ✓ Loaded only necessary columns")
    print("  ✓ Used non-interactive plotting backend")
    print("  ✓ Sampled data for visualization if needed")
    print("  ✓ Closed plots after saving")
    print("  ✓ Freed memory with garbage collection")
    print("\nYour graphs are saved as 'feature_vs_sales.png'")

