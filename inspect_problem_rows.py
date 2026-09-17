import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def inspect_problem_rows(file_path, indices):
    df = load_data(file_path)
    
    # Columns to print for each row
    row_columns = [
        'Id', 'SalePrice', 'GrLivArea', 'OverallQual', 'OverallCond',
        'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageCars', 'GarageArea'
    ]
    
    # Numeric features to calculate statistics for
    numeric_features = [
        'GrLivArea', 'OverallQual', 'OverallCond',
        'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageCars', 'GarageArea'
    ]
    
    # Calculate dataset statistics
    dataset_median = df[numeric_features].median()
    dataset_95th = df[numeric_features].quantile(0.95)
    dataset_max = df[numeric_features].max()
    
    # Print row data for each index
    for idx in indices:
        print(f"\nRow Index: {idx}")
        print("-" * 50)
        print(f"{'Column':<15} | {'Value':<15}")
        print("-" * 50)
        for col in row_columns:
            if col in df.columns:
                value = df.loc[idx, col]
                print(f"{col:<15} | {value:<15}")
    
    # Print dataset statistics for each numeric feature
    print("\n" + "=" * 50)
    print("Dataset Statistics for Numeric Features")
    print("=" * 50)
    
    for feature in numeric_features:
        print(f"\n{feature}:")
        print(f"  Median:    {dataset_median[feature]:.2f}")
        print(f"  95th Pct:  {dataset_95th[feature]:.2f}")
        print(f"  Maximum:   {dataset_max[feature]:.2f}")

if __name__ == '__main__':
    indices = [1298, 523, 1324]
    inspect_problem_rows('train.csv', indices)
