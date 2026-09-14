import pandas as pd
import numpy as np

def inspect_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Shape of the dataset
    print(f"Shape of the dataset: {df.shape}")
    
    # Columns of the dataset
    print(f"Columns of the dataset: {df.columns}")
    
    # Data types of the dataset
    print(f"Data types of the dataset:\n{df.dtypes}")
    
    # Target distribution
    target = 'SalePrice'
    print(f"Target distribution:\n{df[target].describe()}")
    
    # Missing values
    print("Columns with missing values:")
    missing_values = df.isnull().sum().sort_values(ascending=False)
    print(missing_values)
    
    # Unique values
    print(f"Unique values:\n{df.nunique()}")
    
    # Obvious ID columns
    id_columns = [col for col in df.columns if col.startswith('Id')]
    print(f"Obvious ID columns: {id_columns}")
    
    # Numerical, categorical, ordinal-looking, and suspicious columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns
    ordinal_columns = df.select_dtypes(include=[np.number]).apply(lambda x: x.nunique() < 10).index
    suspicious_columns = df.select_dtypes(include=[np.number]).apply(lambda x: x.nunique() > 100).index
    
    print(f"Numerical columns: {numerical_columns}")
    print(f"Categorical columns: {categorical_columns}")
    print(f"Ordinal columns: {ordinal_columns}")
    print(f"Suspicious columns: {suspicious_columns}")

if __name__ == '__main__':
    inspect_data('train.csv')
