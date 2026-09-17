# Experiment Log — House Price Regression

## Project Goal

Build a reliable regression model for predicting `SalePrice` from the Ames Housing dataset.

The purpose of the experiments is not only to maximize predictive performance, but to understand:

* which modeling decisions materially improve performance,
* whether improvements are stable across validation folds,
* where and why models fail,
* how preprocessing and target transformation affect model behavior,
* and whether unusual observations require special treatment.

Experiments are intentionally changed one factor at a time where practical so that improvements or regressions can be attributed to specific decisions.

---

# 0. Dataset Audit

Before fitting models, the dataset was inspected to establish its structure and identify potential modeling issues.

## Dataset

* Rows: 1,460
* Columns: 81
* Target: `SalePrice`
* Identifier: `Id`
* Numerical feature columns: 36
* Categorical feature columns: 43
* Duplicate rows: 0

`Id` was identified as an identifier rather than a meaningful predictive feature and is excluded from model input.

## Target Distribution

`SalePrice` statistics:

* Mean: 180,921
* Median: 163,000
* Minimum: 34,900
* Maximum: 755,000
* Standard deviation: 79,443
* Skewness: approximately 1.88

The target is strongly right-skewed.

This suggested that a logarithmic target transformation might eventually be worth testing, but it was deliberately postponed until after establishing simple baselines.

## Missing Values

Several features contain substantial missing data.

Examples include:

* `PoolQC`
* `MiscFeature`
* `Alley`
* `Fence`
* `MasVnrType`
* `FireplaceQu`
* `LotFrontage`
* garage-related features
* basement-related features

At this stage, missing values were handled generically:

* numerical features: median imputation
* categorical features: most-frequent imputation

More semantic missing-value handling may be explored later if justified.

## Initial Modeling Strategy

To minimize leakage and ensure preprocessing is reproducible, preprocessing is kept inside scikit-learn `Pipeline` / `ColumnTransformer` objects.

General preprocessing:

### Numerical features

* median imputation

### Categorical features

* most-frequent imputation
* one-hot encoding with unknown-category handling

---

# 1. Dummy Regression Baseline

## Purpose

Before training a real model, establish a performance floor.

Without a naive baseline, a model score has little context: even a seemingly reasonable RMSE may provide little improvement over simply predicting a typical house price.

## Model

`DummyRegressor(strategy="median")`

## Validation

Single 80/20 train-validation split:

* `random_state=42`

## Results

| Metric |    Result |
| ------ | --------: |
| MAE    | 59,568.25 |
| RMSE   | 88,667.17 |
| R²     |   -0.0250 |

## Interpretation

The negative R² is expected for a weak constant predictor.

The important result is that we now have a floor that every useful model should beat substantially.

## Why We Moved On

The next goal was to determine whether the available features contain strong predictive signal using a simple interpretable regression model.

---

# 2. Ridge Regression — Single Holdout

## Hypothesis

A regularized linear model with one-hot encoded categorical features should capture substantially more signal than the dummy baseline without introducing complex modeling.

## Model

`Ridge`

Preprocessing remained unchanged.

Target remained raw `SalePrice`.

## Validation

Same 80/20 split:

* `random_state=42`

## Results

| Metric |     Dummy |     Ridge |
| ------ | --------: | --------: |
| MAE    | 59,568.25 | 20,572.04 |
| RMSE   | 88,667.17 | 34,565.56 |
| R²     |   -0.0250 |    0.8442 |

Ridge reduced MAE by approximately 39,000 dollars and RMSE by approximately 54,000 dollars relative to the dummy baseline.

## Interpretation

This confirmed that the dataset contains strong predictive signal and that even a relatively simple linear model can perform well.

However, this result was based on only one train-validation split.

## Why We Moved On

A single split may accidentally be unusually easy or difficult.

Before comparing more models or tuning anything, we needed to determine whether Ridge performance was stable across different subsets of the data.

---

# 3. Ridge Regression — 5-Fold Cross-Validation

## Hypothesis

If the single-split Ridge result is representative, similar performance should appear across multiple validation folds.

## Model

Same raw-target Ridge pipeline.

No modeling changes.

## Validation

```text
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

## Results

| Metric |      Mean | Standard Deviation |
| ------ | --------: | -----------------: |
| MAE    | 20,162.88 |           2,290.89 |
| RMSE   | 35,716.98 |          11,906.90 |
| R²     |    0.7654 |             0.1914 |

## Interpretation

MAE remained relatively stable and close to the original holdout result.

RMSE and R² showed substantially more variability.

This suggested that some folds contain observations where Ridge makes unusually large errors.

Because RMSE squares prediction errors, even a small number of severe misses can dominate this metric.

At the same time, the target distribution was known to be strongly right-skewed.

## Why We Moved On

A common approach for positively skewed price targets is to model:

```text
log1p(SalePrice)
```

rather than raw dollar prices.

The next experiment tested whether a log target could reduce the influence of high-price observations and improve stability.

---

# 4. Ridge Regression with Log-Transformed Target

## Hypothesis

Training Ridge on:

```text
log1p(SalePrice)
```

may produce a regression problem that is closer to linear and less dominated by expensive houses.

Predictions are converted back to dollars using:

```text
expm1(prediction)
```

before calculating dollar-scale metrics.

## Model

Same Ridge model and preprocessing.

Only target handling changed.

## Validation

Same 5-fold cross-validation configuration.

## Results

| Metric   |      Mean | Standard Deviation |
| -------- | --------: | -----------------: |
| MAE      | 19,830.14 |           6,017.55 |
| RMSE     | 63,082.79 |          72,812.27 |
| R²       |   -0.6642 |             3.1082 |
| Log RMSE |    0.1569 |             0.0464 |

At first glance, these aggregate results looked much worse than raw-target Ridge.

However, fold-level results revealed a different story.

## Fold-Level Results

| Fold |       MAE |       RMSE |      R² | Log RMSE |
| ---- | --------: | ---------: | ------: | -------: |
| 1    | 17,889.08 |  29,104.12 |  0.8896 |   0.1484 |
| 2    | 17,196.02 |  28,005.00 |  0.8847 |   0.1282 |
| 3    | 31,744.90 | 208,657.29 | -6.8806 |   0.2482 |
| 4    | 17,012.34 |  26,182.25 |  0.8908 |   0.1350 |
| 5    | 15,308.34 |  23,465.31 |  0.8947 |   0.1245 |

## Interpretation

Four out of five folds performed very well.

Fold 3 was catastrophic and completely dominated the aggregate RMSE and R².

This was a useful example of why mean cross-validation metrics should not be examined without also considering fold-level behavior.

The experiment therefore could not simply be categorized as "log transformation failed."

Instead, Fold 3 required diagnosis.

## Why We Moved On

The next step was to inspect the largest Fold 3 prediction errors rather than changing models immediately.

---

# 5. Fold 3 Error Analysis

## Purpose

Determine whether Fold 3 was generally difficult or whether a small number of extreme predictions were responsible for its poor metrics.

## Largest Errors

### Observation 1298

* Actual SalePrice: $160,000
* Predicted SalePrice: $3,543,210
* Absolute error: $3,383,210
* Actual log target: 11.9829
* Predicted log target: 15.0805

### Observation 523

* Actual SalePrice: $184,750
* Predicted SalePrice: $1,236,098
* Absolute error: $1,051,348
* Actual log target: 12.1268
* Predicted log target: 14.0275

### Observation 1324

* Actual SalePrice: $147,000
* Predicted SalePrice: $267,700
* Absolute error: $120,700

The first two observations overwhelmingly dominated Fold 3 squared error.

## Interpretation

The log model was not broadly failing across the entire fold.

Instead, a few very high log-space predictions became enormous dollar predictions after applying `expm1()`.

This revealed an important effect of inverse logarithmic transformation:

moderately excessive predictions in log space can become extreme errors on the original dollar scale.

However, this did not yet explain why Ridge produced those unusually large log predictions.

## New Hypothesis

Ridge regularization is scale-sensitive.

Numerical features such as:

* `LotArea`
* `GrLivArea`
* `YearBuilt`
* `OverallQual`

operate on very different numerical scales.

The initial preprocessing imputed numerical values but did not standardize them.

This raised the possibility that poorly scaled numerical features were contributing to unstable Ridge coefficients.

## Why We Moved On

The next experiment added numerical standardization while keeping the log-target Ridge experiment otherwise unchanged.

---

# 6. Scaled Numerical Features + Log-Target Ridge

## Hypothesis

Standardizing numerical features before Ridge should make regularization more consistent across coefficients and potentially reduce extreme extrapolation.

## Change

Numerical preprocessing became:

```text
median imputation
→ StandardScaler
```

Everything else remained unchanged:

* same Ridge model
* same log target
* same categorical preprocessing
* same cross-validation folds
* same random seed

## Results

| Metric   |      Mean | Standard Deviation |
| -------- | --------: | -----------------: |
| MAE      | 17,721.47 |           4,424.56 |
| RMSE     | 55,665.47 |          63,635.26 |
| R²       |   -0.2819 |             2.3854 |
| Log RMSE |    0.1486 |             0.0407 |

Typical-fold performance improved.

### Fold Results

| Fold |       MAE |       RMSE |      R² | Log RMSE |
| ---- | --------: | ---------: | ------: | -------: |
| 1    | 15,736.62 |  23,839.68 |  0.9259 |   0.1316 |
| 2    | 15,780.94 |  23,750.44 |  0.9170 |   0.1289 |
| 3    | 26,412.17 | 182,861.15 | -5.0525 |   0.2294 |
| 4    | 16,622.53 |  27,386.53 |  0.8806 |   0.1341 |
| 5    | 14,055.08 |  20,489.57 |  0.9197 |   0.1190 |

Scaling clearly improved most folds.

However, Fold 3 remained severely unstable.

## Fold 3 Extreme Predictions

### Observation 1298

* Actual: $160,000
* Predicted: $3,161,739
* Absolute error: $3,001,739

### Observation 523

* Actual: $184,750
* Predicted: $986,462
* Absolute error: $801,712

### Observation 1324

* Actual: $147,000
* Predicted: $281,700
* Absolute error: $134,700

## Interpretation

Scaling was beneficial.

Compared with unscaled log-target Ridge:

* mean MAE improved,
* mean log RMSE improved,
* four folds became particularly strong.

However, the same problematic observations remained.

Therefore:

> Lack of feature scaling was not the root cause of the Fold 3 failure.

Scaling reduced the severity of the problem but did not remove it.

## Why We Moved On

The next remaining major variable was the target transformation itself.

To isolate its contribution, the following experiment kept numerical scaling but returned to raw `SalePrice`.

---

# 7. Scaled Numerical Features + Raw-Target Ridge

## Hypothesis

If the catastrophic Fold 3 behavior is primarily caused by the log/exponential target transformation, training on raw prices should eliminate or greatly reduce those extreme predictions.

## Change

Kept:

* numerical median imputation
* `StandardScaler`
* categorical imputation
* one-hot encoding
* Ridge
* identical 5-fold split

Changed:

* removed `log1p`
* removed `expm1`
* trained directly on raw `SalePrice`

## Results

| Metric |      Mean | Standard Deviation |
| ------ | --------: | -----------------: |
| MAE    | 18,478.87 |             924.36 |
| RMSE   | 33,619.59 |          10,353.11 |
| R²     |    0.7937 |             0.1605 |

### Fold Results

| Fold |       MAE |      RMSE |     R² |
| ---- | --------: | --------: | -----: |
| 1    | 19,003.97 | 29,840.90 | 0.8839 |
| 2    | 18,408.04 | 30,295.44 | 0.8650 |
| 3    | 19,128.68 | 53,951.78 | 0.4731 |
| 4    | 19,143.31 | 29,207.29 | 0.8641 |
| 5    | 16,710.33 | 24,802.55 | 0.8823 |

## Interpretation

This became the strongest stable Ridge configuration so far.

Compared with unscaled raw-target Ridge:

* MAE improved from 20,163 to 18,479
* RMSE improved from 35,717 to 33,620
* R² improved from 0.765 to 0.794
* MAE variability fell substantially

This confirms that standardizing numerical features was an important improvement for Ridge.

However, Fold 3 remained notably weaker.

## Problematic Predictions

### Observation 1298

* Actual: $160,000
* Predicted: $873,260
* Error: $713,260

### Observation 523

* Actual: $184,750
* Predicted: $617,423
* Error: $432,673

### Observation 1324

* Actual: $147,000
* Predicted: $293,774
* Error: $146,774

These errors were much smaller than under the log-target model, but the same observations were still problematic.

## Conclusion

The log transformation did **not create** the underlying issue.

Instead:

> Ridge already overpredicts these observations on the raw target, and the exponential inverse transform greatly amplifies the error when using a log target.

This shifted attention away from target transformation and toward the observations themselves.

## Why We Moved On

Because the same rows repeatedly caused problems under multiple Ridge configurations, the next step was to inspect their feature values.

---

# 8. Investigation of High-Error Observations

## Purpose

Determine whether observations 1298, 523, and 1324 occupy unusual regions of feature space.

No rows were removed and no model was changed during this investigation.

## Observation 1298

* `Id`: 1299
* `SalePrice`: $160,000
* `GrLivArea`: 5,642
* `OverallQual`: 10
* `OverallCond`: 5
* `YearBuilt`: 2008
* `TotalBsmtSF`: 6,110
* `1stFlrSF`: 4,692
* `2ndFlrSF`: 950
* `GarageCars`: 2
* `GarageArea`: 1,418

## Observation 523

* `Id`: 524
* `SalePrice`: $184,750
* `GrLivArea`: 4,676
* `OverallQual`: 10
* `OverallCond`: 5
* `YearBuilt`: 2007
* `TotalBsmtSF`: 3,138
* `1stFlrSF`: 3,138
* `2ndFlrSF`: 1,538
* `GarageCars`: 3
* `GarageArea`: 884

## Observation 1324

* `Id`: 1325
* `SalePrice`: $147,000
* `GrLivArea`: 1,795
* `OverallQual`: 8
* `OverallCond`: 5
* `YearBuilt`: 2006
* `TotalBsmtSF`: 1,795
* `1stFlrSF`: 1,795
* `2ndFlrSF`: 0
* `GarageCars`: 3
* `GarageArea`: 895

## Dataset Reference Statistics

### GrLivArea

* Median: 1,464
* 95th percentile: 2,466
* Maximum: 5,642

### TotalBsmtSF

* Median: 991.5
* 95th percentile: 1,753
* Maximum: 6,110

### 1stFlrSF

* Median: 1,087
* 95th percentile: 1,831
* Maximum: 4,692

### GarageArea

* Median: 480
* 95th percentile: 850
* Maximum: 1,418

## Interpretation

Observations 1298 and 523 are extreme high-leverage observations.

Both have:

* extremely large living area,
* maximum `OverallQual`,
* recent construction years,
* very large floor/basement areas,
* substantial garage capacity,

yet relatively modest sale prices.

For example:

```text
Observation 1298:
GrLivArea = 5642
95th percentile = 2466
SalePrice = $160,000
```

From the perspective of a linear model, this observation looks like an extremely large, modern, high-quality house.

A large positive prediction is therefore understandable.

The actual sale price is unusual relative to the observed feature combination.

Observation 523 shows a similar pattern.

Observation 1324 is different. Its size is not nearly as extreme and should not automatically be grouped with the first two observations.

## Current Working Conclusion

The Fold 3 instability is primarily driven by a very small number of high-leverage observations whose feature combinations are poorly represented by their target prices.

The evidence so far suggests:

1. Ridge is capable of strong overall performance.
2. Scaling numerical features improves Ridge materially.
3. Log-target Ridge performs extremely well on typical observations.
4. Log-target predictions can become catastrophic when Ridge overestimates unusual houses because `expm1()` magnifies errors.
5. The underlying overprediction exists even without the log transformation.
6. Two specific observations are extreme in feature space and repeatedly dominate error metrics.
7. These observations should not be removed automatically simply because they hurt model performance.

---

# Current Model Comparison

| Experiment                   |              MAE |                RMSE |                R² | Notes                                  |
| ---------------------------- | ---------------: | ------------------: | ----------------: | -------------------------------------- |
| Dummy median                 |           59,568 |              88,667 |            -0.025 | Performance floor                      |
| Ridge, single holdout        |           20,572 |              34,566 |             0.844 | Strong first model                     |
| Ridge, raw target CV         |   20,163 ± 2,291 |     35,717 ± 11,907 |     0.765 ± 0.191 | Exposed instability                    |
| Ridge, log target CV         |   19,830 ± 6,018 |     63,083 ± 72,812 |    -0.664 ± 3.108 | Fold 3 catastrophic                    |
| Scaled Ridge, log target     |   17,721 ± 4,425 |     55,665 ± 63,635 |    -0.282 ± 2.385 | Strong typical folds, same outliers    |
| **Scaled Ridge, raw target** | **18,479 ± 924** | **33,620 ± 10,353** | **0.794 ± 0.161** | Best stable Ridge configuration so far |

The log-target model currently achieves the best typical-fold MAE, but its catastrophic outlier behavior makes its dollar-scale RMSE and R² unreliable.

The scaled raw-target Ridge model is currently the most stable Ridge configuration.

---

# Key Lessons So Far

## 1. Always establish a naive baseline

Without the dummy regression result, it would be difficult to quantify how much Ridge actually improved prediction quality.

## 2. A single holdout score is insufficient

The first Ridge result suggested R² ≈ 0.84.

Cross-validation revealed that performance varies considerably depending on which observations fall into the validation fold.

## 3. Aggregate CV statistics can hide the actual failure mode

The log-target Ridge model appeared disastrous from its average R².

Examining individual folds showed that four folds were strong and one fold contained catastrophic errors.

## 4. RMSE can be dominated by very few observations

Two predictions were sufficient to make an otherwise strong model appear extremely poor under RMSE.

## 5. Target transformations change error behavior

`log1p` can make the central regression problem easier, but `expm1` can turn high log-space predictions into extremely large dollar errors.

## 6. Ridge requires attention to feature scaling

Adding `StandardScaler` improved both typical performance and stability.

## 7. Do not remove outliers simply because they hurt the score

The problematic observations were investigated first.

Any later outlier policy should be:

* explicitly justified,
* documented,
* compared experimentally,
* and evaluated using the same validation framework.

## 8. Error analysis can be more informative than another model

The most useful discovery so far did not come from adding a more complex algorithm.

It came from tracing a poor aggregate metric down to:

```text
CV average
→ Fold 3
→ individual predictions
→ specific observations
→ unusual feature combinations
```

---

# Next Questions

The current analysis suggests several future experiments.

They should be tested separately rather than combined immediately.

## Nonlinear Models

Linear Ridge regression extrapolates strongly for unusual feature combinations.

Tree-based models may behave differently because they generally do not extrapolate linearly outside learned regions.

Candidate models include:

* Random Forest
* gradient boosting
* XGBoost

## Explicit Outlier Policy

A controlled experiment may compare:

* training with all observations,
* versus a clearly defined training-only outlier policy.

Rows must not be removed merely because their validation predictions are bad.

## Feature Engineering

Potential engineered features include:

* total square footage,
* house age,
* years since remodel,
* total bathrooms,
* total porch area,
* interactions between overall quality and size.

Each meaningful feature-engineering group should be tested against the same validation framework.

## Ordinal Feature Encoding

Several categorical quality features have natural order.

Treating them as ordinal rather than ordinary one-hot categories may provide useful structure.

## Model Tuning

Hyperparameter tuning should occur only after candidate model families and preprocessing approaches have been compared under reliable cross-validation.

---

# Current Status

The project has progressed from a naive constant predictor to a reasonably strong and reproducible Ridge baseline.

More importantly, cross-validation and targeted error analysis uncovered a specific failure mode that would have been invisible from a single validation score.

The current strongest stable linear baseline is:

```text
Scaled numerical features
+ categorical one-hot encoding
+ raw SalePrice target
+ Ridge regression
```

with:

```text
MAE  = 18,478.87 ± 924.36
RMSE = 33,619.59 ± 10,353.11
R²   = 0.7937 ± 0.1605
```

The next phase should determine whether nonlinear models can preserve strong typical performance while handling the extreme high-leverage observations more robustly.
