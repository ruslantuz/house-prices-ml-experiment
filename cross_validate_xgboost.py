import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(file_path):
    return pd.read_csv(file_path)


def cross_validate_xgboost():
    df = load_data("train.csv")

    # Separate features and target
    X = df.drop(columns=["SalePrice", "Id"])
    y = df["SalePrice"]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numerical_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    # Model pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Cross-validation
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_results = []

    # Saved out-of-fold data from Fold 3
    fold_3_indices = None
    fold_3_y_val = None
    fold_3_y_pred = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Fit only on this fold's training data
        model.fit(X_train, y_train)

        # Out-of-fold predictions for this fold
        y_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        fold_results.append(
            {
                "fold": fold,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

        # Save the exact Fold 3 validation data and predictions
        if fold == 3:
            fold_3_indices = X_val.index.to_numpy().copy()
            fold_3_y_val = y_val.copy()
            fold_3_y_pred = y_pred.copy()

        print(f"Fold {fold}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.2f}")

    # Overall metrics
    maes = [result["mae"] for result in fold_results]
    rmses = [result["rmse"] for result in fold_results]
    r2_scores = [result["r2"] for result in fold_results]

    print("\nOverall Results:")
    print(f"Mean MAE ± std: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
    print(f"Mean RMSE ± std: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
    print(f"Mean R² ± std: {np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}")

    # Fold 3 diagnostics using ONLY Fold 3 out-of-fold predictions
    requested_indices = [1298, 523, 1324]

    print("\nFold 3 Diagnostics:")

    for dataframe_index in requested_indices:
        positions = np.where(fold_3_indices == dataframe_index)[0]

        if len(positions) == 0:
            print(
                f"Index {dataframe_index} was not present "
                "in the Fold 3 validation set."
            )
            continue

        pos = positions[0]

        # y_val is a pandas Series, so positional indexing requires .iloc
        actual = fold_3_y_val.iloc[pos]

        # y_pred is a NumPy array
        predicted = fold_3_y_pred[pos]

        absolute_error = abs(actual - predicted)

        print(f"Index {dataframe_index}:")
        print(f"  Actual SalePrice: {actual:.2f}")
        print(f"  Predicted SalePrice: {predicted:.2f}")
        print(f"  Absolute Error: {absolute_error:.2f}")


if __name__ == "__main__":
    cross_validate_xgboost()
