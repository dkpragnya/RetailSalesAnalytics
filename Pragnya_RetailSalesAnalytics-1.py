"""
Project: AI-Powered Retail Sales Data Analytics and Prediction
Author: D.K. Pragnya

Creates a reproducible synthetic retail dataset, explores sales patterns,
trains a baseline machine-learning model, evaluates it, and saves outputs.
Run: python Pragnya_RetailSalesAnalytics.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

def create_dataset(n=1200, seed=42):
    """Generate sample retail data for demonstration; it is not real customer data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    date = rng.choice(dates, size=n)
    categories = np.array(["Electronics", "Home", "Clothing", "Grocery", "Books"])
    category = rng.choice(categories, size=n, p=[.18, .22, .24, .25, .11])
    region = rng.choice(["North", "South", "East", "West"], size=n)
    price = np.where(category == "Electronics", rng.uniform(80, 900, n),
            np.where(category == "Home", rng.uniform(15, 250, n),
            np.where(category == "Clothing", rng.uniform(10, 140, n),
            np.where(category == "Grocery", rng.uniform(2, 60, n),
                     rng.uniform(5, 80, n)))))
    units = rng.integers(1, 15, n)
    discount = rng.choice([0, .05, .10, .15, .20], size=n, p=[.35, .15, .25, .15, .10])
    seasonality = 1 + .15 * np.sin(2 * np.pi * pd.DatetimeIndex(date).dayofyear / 365.25)
    noise = rng.normal(0, 0.08, n)
    sales = np.maximum(0, price * units * (1 - discount) * seasonality * (1 + noise))
    return pd.DataFrame({
        "date": pd.to_datetime(date), "category": category, "region": region,
        "unit_price": price.round(2), "units_sold": units,
        "discount": discount, "sales": sales.round(2)
    })

def main():
    df = create_dataset()
    df.to_csv(OUTPUT / "retail_sales_dataset.csv", index=False)

    df["month"] = df["date"].dt.month
    summary = df.groupby("category", as_index=False).agg(
        total_sales=("sales", "sum"),
        average_sales=("sales", "mean"),
        total_units=("units_sold", "sum"),
        transactions=("sales", "count")
    ).sort_values("total_sales", ascending=False)
    summary.to_csv(OUTPUT / "category_summary.csv", index=False)

    monthly = df.groupby(df["date"].dt.to_period("M"))["sales"].sum()
    plt.figure(figsize=(10, 5))
    monthly.index = monthly.index.astype(str)
    monthly.plot()
    plt.title("Monthly Retail Sales (Synthetic Demonstration Data)")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(OUTPUT / "monthly_sales.png", dpi=150)
    plt.close()

    features = ["category", "region", "unit_price", "units_sold", "discount", "month"]
    X = df[features]
    y = df["sales"]
    categorical = ["category", "region"]
    numeric = ["unit_price", "units_sold", "discount", "month"]
    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric)
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("regressor", RandomForestRegressor(n_estimators=160, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = pd.DataFrame([{
        "model": "RandomForestRegressor",
        "test_rows": len(y_test),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4)
    }])
    metrics.to_csv(OUTPUT / "model_metrics.csv", index=False)

    print("Project completed.")
    print(f"Rows: {len(df)}")
    print("Category summary:")
    print(summary.to_string(index=False))
    print("\nModel evaluation (test set):")
    print(metrics.to_string(index=False))
    print(f"\nFiles saved in: {OUTPUT.resolve()}")

if __name__ == "__main__":
    main()
