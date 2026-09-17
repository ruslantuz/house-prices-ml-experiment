import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge


def load_data(file_path):
    return pd.read_csv(file_path)


def cross_validate_log_target():
    df = load_data("train.csv")

    # Separate target from features
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice", "Id"])

    numerical_columns = X.select_dtypes(include=[np.number]).columns
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    numerical_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    model = Ridge()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Calculate metrics for each fold
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    log_rmse_scores = []

    # Store validation data for fold 3 diagnosis
    fold3_data = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Transform target using log1p for training
        y_train_fold_log = np.log1p(y_train_fold)

        pipeline.fit(X_train_fold, y_train_fold_log)

        # Predict in log space
        y_pred_log = pipeline.predict(X_val_fold)

        # Convert predictions back to original scale
        y_pred = np.expm1(y_pred_log)

        # Calculate metrics on original dollar scale
        mae = mean_absolute_error(y_val_fold, y_pred)
        rmse = mean_squared_error(y_val_fold, y_pred) ** 0.5
        r2 = r2_score(y_val_fold, y_pred)

        # Calculate RMSE in log space
        y_val_fold_log = np.log1p(y_val_fold)
        rmse_log = mean_squared_error(y_val_fold_log, y_pred_log) ** 0.5

        # Store validation data for fold 3 diagnosis
        if fold == 3:
            fold3_data.append({
                'original_index': df.index[val_idx],
                'actual': y_val_fold,
                'predicted': y_pred,
                'actual_log': y_val_fold_log,
                'predicted_log': y_pred_log,
            })

        # Print metrics for this fold
        print(f"Fold {fold}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Log RMSE: {rmse_log:.4f}")

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        log_rmse_scores.append(rmse_log)

    # Calculate mean and standard deviation
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)

    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    mean_log_rmse = np.mean(log_rmse_scores)
    std_log_rmse = np.std(log_rmse_scores)

    print(f"Mean MAE: {mean_mae:.2f} ± {std_mae:.2f}")
    print(f"Mean RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
    print(f"Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Mean Log RMSE: {mean_log_rmse:.4f} ± {std_log_rmse:.4f}")

    # Print fold 3 diagnosis
    if fold3_data:
        print("\nFold 3 Diagnosis - Top 10 Largest Absolute Errors:")
        print("-" * 80)
        
        # Create dataframe for sorting
        fold3_df = pd.DataFrame(fold3_data)
        
        # Calculate absolute errors
        fold3_df['abs_error'] = fold3_df['predicted'] - fold3_df['actual']
        fold3_df['abs_error'] = fold3_df['abs_error'].abs()
        
        # Sort by absolute error descending
        fold3_df = fold3_df.sort_values('abs_error', ascending=False)
        
        # Print top 10
        for idx, row in fold3_df.head(10).iterrows():
            print(f"Index: {row['original_index']}")
            print(f"  Actual SalePrice: {row['actual']:.2f}")
            print(f"  Predicted SalePrice: {row['predicted']:.2f}")
            print(f"  Absolute Error: {row['abs_error']:.2f}")
            print(f"  Actual log1p(SalePrice): {row['actual_log']:.4f}")
            print(f"  Predicted log SalePrice: {row['predicted_log']:.4f}")
            print("-" * 80)


if __name__ == "__main__":
    cross_validate_log_target()
