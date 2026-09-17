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


def cross_validate_xgboost_log_target():
    df = load_data("train.csv")

    # Keep the raw target for dollar-scale evaluation and diagnostics.
    X = df.drop(columns=["SalePrice", "Id"])
    y = df["SalePrice"]
    y_log = np.log1p(y)

    numerical_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

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

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_results = []

    # Saved solely from the Fold 3 out-of-fold iteration.
    fold_3_indices = None
    fold_3_y_val = None
    fold_3_y_val_log = None
    fold_3_y_pred_log = None
    fold_3_y_pred = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train_log = y_log.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_val_log = y_log.iloc[val_idx]

        # Train and predict exclusively in log space.
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        log_rmse = np.sqrt(mean_squared_error(y_val_log, y_pred_log))

        fold_results.append(
            {
                "fold": fold,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "log_rmse": log_rmse,
            }
        )

        if fold == 3:
            fold_3_indices = X_val.index.to_numpy().copy()
            fold_3_y_val = y_val.copy()
            fold_3_y_val_log = y_val_log.copy()
            fold_3_y_pred_log = y_pred_log.copy()
            fold_3_y_pred = y_pred.copy()

        print(f"Fold {fold}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.2f}")
        print(f"  Log RMSE: {log_rmse:.4f}")

    maes = [result["mae"] for result in fold_results]
    rmses = [result["rmse"] for result in fold_results]
    r2_scores = [result["r2"] for result in fold_results]
    log_rmses = [result["log_rmse"] for result in fold_results]

    print("\nOverall Results:")
    print(f"Mean MAE ± std: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
    print(f"Mean RMSE ± std: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
    print(f"Mean R² ± std: {np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}")
    print(
        f"Mean Log RMSE ± std: {np.mean(log_rmses):.4f} ± "
        f"{np.std(log_rmses):.4f}"
    )

    print("\nFold 3 Diagnostics:")
    for dataframe_index in [1298, 523, 1324]:
        positions = np.where(fold_3_indices == dataframe_index)[0]
        if len(positions) == 0:
            print(f"Index {dataframe_index} was not present in Fold 3 validation.")
            continue

        pos = positions[0]
        actual = fold_3_y_val.iloc[pos]
        actual_log = fold_3_y_val_log.iloc[pos]
        predicted_log = fold_3_y_pred_log[pos]
        predicted = fold_3_y_pred[pos]

        print(f"Index {dataframe_index}:")
        print(f"  Actual SalePrice: {actual:.2f}")
        print(f"  Predicted SalePrice: {predicted:.2f}")
        print(f"  Absolute Dollar Error: {abs(actual - predicted):.2f}")
        print(f"  Actual log1p(SalePrice): {actual_log:.4f}")
        print(f"  Predicted log SalePrice: {predicted_log:.4f}")


if __name__ == "__main__":
    cross_validate_xgboost_log_target()
