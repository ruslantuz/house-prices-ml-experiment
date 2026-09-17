import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Cross-validation function
def cross_validate_random_forest():
    # Load data
    df = load_data('train.csv')
    
    # Separate features and target
    X = df.drop(columns=['SalePrice', 'Id'])
    y = df['SalePrice']
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))
            ]), numerical_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ]
    )
    
    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Set up cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    fold_results = []

    # Saved exclusively from the Fold 3 validation iteration.
    fold_3_indices = None
    fold_3_y_val = None
    fold_3_y_pred = None
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        fold_results.append({
            'fold': fold,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })

        if fold == 3:
            fold_3_indices = X_test.index.to_numpy().copy()
            fold_3_y_val = y_test.copy()
            fold_3_y_pred = y_pred.copy()
        
        # Print fold results
        print(f"Fold {fold}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.2f}")
    
    # Calculate overall metrics
    maes = [r['mae'] for r in fold_results]
    rmse_values = [r['rmse'] for r in fold_results]
    r2_values = [r['r2'] for r in fold_results]
    
    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    
    print("\nOverall Results:")
    print(f"Mean MAE ± std: {mean_mae:.2f} ± {std_mae:.2f}")
    print(f"Mean RMSE ± std: {mean_rmse:.2f} ± {std_rmse:.2f}")
    print(f"Mean R² ± std: {mean_r2:.2f} ± {std_r2:.2f}")
    
    # Fold 3 diagnostics use only the saved out-of-fold validation predictions.
    print("\nFold 3 Diagnostics:")

    for idx in [1298, 523, 1324]:
        positions = np.where(fold_3_indices == idx)[0]

        if len(positions) == 0:
            print(f"Index {idx} was not present in Fold 3 validation.")
            continue

        pos = positions[0]
        actual = fold_3_y_val.iloc[pos]
        predicted = fold_3_y_pred[pos]
        error = abs(actual - predicted)

        print(f"Index {idx}:")
        print(f"  Actual SalePrice: {actual:.2f}")
        print(f"  Predicted SalePrice: {predicted:.2f}")
        print(f"  Absolute Error: {error:.2f}")

if __name__ == '__main__':
    cross_validate_random_forest()
