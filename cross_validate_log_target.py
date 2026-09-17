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

    for train_idx, val_idx in kfold.split(X):
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


if __name__ == "__main__":
    cross_validate_log_target()
