import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


OUTLIER_THRESHOLD = 4000
REQUESTED_INDICES = [1298, 523, 1324]


def build_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]
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


def build_model(model_name, numerical_cols, categorical_cols):
    if model_name == "Random Forest":
        regressor = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        regressor = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numerical_cols, categorical_cols)),
            ("regressor", regressor),
        ]
    )


def calculate_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def compare_outlier_policy():
    df = pd.read_csv("train.csv")
    X = df.drop(columns=["SalePrice", "Id"])
    y = df["SalePrice"]

    numerical_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    configurations = [
        ("Random Forest baseline", "Random Forest", False),
        ("Random Forest filtered", "Random Forest", True),
        ("XGBoost baseline", "XGBoost", False),
        ("XGBoost filtered", "XGBoost", True),
    ]
    results = {name: [] for name, _, _ in configurations}
    fold_3_predictions = {}
    fold_3_indices = None
    fold_3_y_val = None

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        train_outlier_mask = X_train["GrLivArea"] > OUTLIER_THRESHOLD
        removed_count = int(train_outlier_mask.sum())
        X_train_filtered = X_train.loc[~train_outlier_mask]
        y_train_filtered = y_train.loc[~train_outlier_mask]

        print(f"\nFold {fold} (training rows removed by rule: {removed_count}):")
        for name, model_name, use_filtered_training in configurations:
            model = build_model(model_name, numerical_cols, categorical_cols)
            if use_filtered_training:
                model.fit(X_train_filtered, y_train_filtered)
            else:
                model.fit(X_train, y_train)

            # Validation is never filtered.
            y_pred = model.predict(X_val)
            metrics = calculate_metrics(y_val, y_pred)
            results[name].append(metrics)

            print(
                f"  {name}: MAE {metrics['mae']:.2f} | "
                f"RMSE {metrics['rmse']:.2f} | R² {metrics['r2']:.2f}"
            )

            if fold == 3:
                fold_3_predictions[name] = y_pred.copy()

        if fold == 3:
            fold_3_indices = X_val.index.to_numpy().copy()
            fold_3_y_val = y_val.copy()

    print("\nOverall Comparison:")
    print(
        f"{'Configuration':<25} {'MAE (mean ± std)':>23} "
        f"{'RMSE (mean ± std)':>24} {'R² (mean ± std)':>20}"
    )
    for name, _, _ in configurations:
        metrics = results[name]
        maes = [metric["mae"] for metric in metrics]
        rmses = [metric["rmse"] for metric in metrics]
        r2_scores = [metric["r2"] for metric in metrics]
        print(
            f"{name:<25} "
            f"{np.mean(maes):.2f} ± {np.std(maes):.2f}  "
            f"{np.mean(rmses):.2f} ± {np.std(rmses):.2f}  "
            f"{np.mean(r2_scores):.2f} ± {np.std(r2_scores):.2f}"
        )

    print("\nFold 3 Diagnostics (out-of-fold validation predictions only):")
    for dataframe_index in REQUESTED_INDICES:
        positions = np.where(fold_3_indices == dataframe_index)[0]
        if len(positions) == 0:
            print(f"Index {dataframe_index} was not present in Fold 3 validation.")
            continue

        pos = positions[0]
        actual = fold_3_y_val.iloc[pos]
        print(f"Index {dataframe_index}:")
        print(f"  Actual SalePrice: {actual:.2f}")
        for name, _, _ in configurations:
            predicted = fold_3_predictions[name][pos]
            print(
                f"  {name}: {predicted:.2f} "
                f"(absolute error: {abs(actual - predicted):.2f})"
            )


if __name__ == "__main__":
    compare_outlier_policy()
