import pandas as pd
import numpy as np


def inspect_data(file_path):
    df = pd.read_csv(file_path)

    print(f"Shape of the dataset: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    target = "SalePrice"

    print("\nTarget distribution:")
    print(df[target].describe())

    print(f"\nSalePrice skewness: {df[target].skew():.4f}")

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

    print("\nColumns with missing values:")
    print(missing_values)

    print("\nUnique values:")
    print(df.nunique())

    id_columns = [col for col in df.columns if col.lower() == "id"]
    print(f"\nObvious ID columns: {id_columns}")

    numerical_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    numerical_features = [
        col
        for col in numerical_columns
        if col not in {target, "Id"}
    ]

    print(f"\nNumber of numerical feature columns: {len(numerical_features)}")
    print(f"Number of categorical feature columns: {len(categorical_columns)}")

    print("\nNumerical feature columns:")
    print(numerical_features)

    print("\nCategorical feature columns:")
    print(categorical_columns.tolist())

    print(f"\nDuplicate row count: {df.duplicated().sum()}")


if __name__ == "__main__":
    inspect_data("train.csv")