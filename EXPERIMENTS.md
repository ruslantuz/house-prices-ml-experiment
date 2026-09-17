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




# 9. Random Forest — Nonlinear Baseline (Corrected)

## Hypothesis

The Ridge experiments showed that a small number of unusual observations, particularly dataframe indices 1298 and 523, caused very large prediction errors.

Because Ridge is a linear model, it can extrapolate aggressively when validation observations lie far outside the typical feature distribution.

A Random Forest should behave differently because tree-based models partition the observed feature space rather than extrapolating linearly.

The hypothesis was:

> A nonlinear Random Forest may reduce catastrophic errors on the previously identified high-leverage observations while maintaining or improving overall cross-validation performance.

## Experiment Design

The goal was to change the **model family only**.

Target:

* raw `SalePrice`
* no log transformation
* `SalePrice` excluded from features
* `Id` removed

Cross-validation:

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Numerical preprocessing:

* median imputation
* no scaling

Categorical preprocessing:

* most-frequent imputation
* `OneHotEncoder(handle_unknown="ignore")`

Model:

```python
RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
```

No hyperparameter tuning, outlier removal, target transformation, or feature engineering was performed.

## Methodology Correction

The original Random Forest implementation accidentally retained `Id` as a numerical feature. Its section labeled “Fold 3 Diagnostics” also refit the model on the full dataset and reported in-sample predictions rather than Fold 3 out-of-fold predictions.

The corrected implementation drops `Id` and captures predictions directly during Fold 3 validation. The corrected results below replace the earlier Random Forest numbers; the earlier values are not valid comparison results.

## Results

### Fold-Level Results

| Fold |       MAE |      RMSE |   R² |
| ---- | --------: | --------: | ---: |
| 1    | 17,567.06 | 29,107.80 | 0.89 |
| 2    | 17,198.17 | 25,996.57 | 0.90 |
| 3    | 20,888.49 | 44,158.02 | 0.65 |
| 4    | 17,694.34 | 27,933.19 | 0.88 |
| 5    | 15,330.23 | 23,775.56 | 0.89 |

### Overall Results

| Metric |      Mean | Standard Deviation |
| ------ | --------: | -----------------: |
| MAE    | 17,735.66 |           1,791.40 |
| RMSE   | 30,194.23 |           7,212.48 |
| R²     |      0.84 |               0.10 |

## Comparison with Scaled Raw-Target Ridge

| Model         |               MAE |               RMSE |              R² |
| ------------- | ----------------: | -----------------: | --------------: |
| Scaled Ridge  |   18,478.87 ± 924 | 33,619.59 ± 10,353 | 0.7937 ± 0.1605 |
| Random Forest | 17,735.66 ± 1,791 |  30,194.23 ± 7,212 |     0.84 ± 0.10 |

Random Forest improved mean MAE by approximately $743 and mean RMSE by approximately $3,425.

It also reduced RMSE and R² variability.

## Fold 3 Diagnostics

### Observation 1298

* Actual: $160,000
* Scaled Ridge prediction: $873,260
* Random Forest prediction: $588,331
* Random Forest absolute error: $428,331

### Observation 523

* Actual: $184,750
* Scaled Ridge prediction: $617,423
* Random Forest prediction: $589,837
* Random Forest absolute error: $405,087

### Observation 1324

* Actual: $147,000
* Scaled Ridge prediction: $293,774
* Random Forest prediction: $290,397
* Random Forest absolute error: $143,397

## Interpretation

Random Forest still improves overall RMSE and R² relative to scaled raw-target Ridge. However, the genuine out-of-fold predictions show that it does not solve the extreme observations nearly as well as the earlier in-sample diagnostics suggested. It still substantially overpredicts indices 1298 and 523.

This supports the hypothesis that model family changes behavior on unusual feature combinations, but it does not establish Random Forest as a complete solution to the high-leverage observations.

An important tradeoff also appeared in Fold 3:

* Random Forest had slightly worse MAE than scaled Ridge.
* Random Forest had substantially better RMSE and R².

This suggests Random Forest may make somewhat larger ordinary errors on some observations while avoiding the few catastrophic errors that strongly affect RMSE.

## Conclusion

The hypothesis was supported.

Random Forest:

* improved overall CV performance,
* improved mean R²,
* reduced RMSE variability,
* but still made very large out-of-fold errors on the known high-leverage observations.

This provides evidence that choosing a model better suited to nonlinear relationships can address unusual observations without simply deleting them.

## Why We Moved On

Random Forest established a stronger nonlinear baseline than Ridge on aggregate RMSE and R².

The next question was whether a gradient-boosted tree model could improve typical prediction accuracy further while retaining similar robustness.

---

# 10. XGBoost — Raw-Target Baseline

## Hypothesis

Gradient boosting may model nonlinear relationships and feature interactions more efficiently than Random Forest.

The hypothesis was:

> A baseline XGBoost model may improve average prediction accuracy relative to Random Forest while still handling unusual feature combinations better than Ridge.

To isolate the model-family effect, no feature engineering, target transformation, outlier removal, or hyperparameter tuning was introduced.

## Experiment Design

Target:

* raw `SalePrice`
* no `log1p`
* `SalePrice` excluded from features
* `Id` removed

Cross-validation:

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Numerical preprocessing:

* median imputation
* no scaling

Categorical preprocessing:

* most-frequent imputation
* one-hot encoding with unknown-category handling

Model:

```python
XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
```

Default XGBoost hyperparameters were otherwise retained.

## Results

### Fold-Level Results

| Fold |       MAE |      RMSE |   R² |
| ---- | --------: | --------: | ---: |
| 1    | 17,192.60 | 26,803.78 | 0.91 |
| 2    | 17,080.02 | 28,989.19 | 0.88 |
| 3    | 18,879.22 | 45,093.43 | 0.63 |
| 4    | 19,128.35 | 31,099.73 | 0.85 |
| 5    | 15,472.26 | 24,706.50 | 0.88 |

### Overall Results

| Metric |      Mean | Standard Deviation |
| ------ | --------: | -----------------: |
| MAE    | 17,550.49 |           1,335.94 |
| RMSE   | 31,338.53 |           7,201.69 |
| R²     |      0.83 |               0.10 |

## Comparison with Previous Models

| Model         |                   MAE |                  RMSE |              R² |
| ------------- | --------------------: | --------------------: | --------------: |
| Scaled Ridge  |       18,478.87 ± 924 |    33,619.59 ± 10,353 | 0.7937 ± 0.1605 |
| Random Forest |     17,735.66 ± 1,791 | **30,194.23 ± 7,212** | **0.84 ± 0.10** |
| XGBoost       | **17,550.49 ± 1,336** |     31,338.53 ± 7,202 |     0.83 ± 0.10 |

XGBoost produced the best mean MAE so far.

However, Random Forest retained better:

* mean RMSE,
* mean R²,
* and aggregate RMSE/R².

This shows that the models have different error profiles rather than one being uniformly superior.

## Fold 3 Diagnostics

The diagnostic predictions were captured directly from the out-of-fold predictions produced by the Fold 3 model.

### Observation 1298

* Actual SalePrice: $160,000
* XGBoost prediction: $606,205
* Absolute error: $446,205

### Observation 523

* Actual SalePrice: $184,750
* XGBoost prediction: $671,884
* Absolute error: $487,134

### Observation 1324

* Actual SalePrice: $147,000
* XGBoost prediction: $286,745
* Absolute error: $139,745

## Comparison on Difficult Observations

| Index |   Actual | Scaled Ridge | Random Forest |  XGBoost |
| ----- | -------: | -----------: | ------------: | -------: |
| 1298  | $160,000 |     $873,260 | $588,331 | $606,205 |
| 523   | $184,750 |     $617,423 | $589,837 | $671,884 |
| 1324  | $147,000 |     $293,774 | $290,397 | **$286,745** |

Random Forest is slightly better on index 1298 and materially better on index 523. Raw XGBoost is slightly better on index 1324. Both models still produce large errors on the two most extreme observations.

## Interpretation

The hypothesis was only partially supported.

XGBoost improved typical absolute-error performance and achieved the best mean MAE so far.

However, it did not handle index 523 as effectively as Random Forest, while its result on index 1324 was slightly better.

This explains the metric tradeoff:

* XGBoost has better MAE.
* Random Forest has better RMSE and R².

MAE weights every absolute error linearly, while RMSE strongly penalizes a small number of very large misses.

The two models therefore appear to optimize different aspects of prediction quality under their current configurations.

## Conclusion

XGBoost is a promising candidate but does not currently dominate Random Forest.

At this stage:

> **XGBoost provides the best typical absolute-error performance, while Random Forest provides better aggregate RMSE and R².**

No model should yet be selected as final.

---

# Updated Model Comparison

| Experiment               |                MAE |               RMSE |              R² | Main Finding                          |
| ------------------------ | -----------------: | -----------------: | --------------: | ------------------------------------- |
| Dummy median             |             59,568 |             88,667 |          -0.025 | Performance floor                     |
| Ridge, single holdout    |             20,572 |             34,566 |           0.844 | Strong initial linear model           |
| Ridge, raw-target CV     |     20,163 ± 2,291 |    35,717 ± 11,907 |   0.765 ± 0.191 | Revealed fold instability             |
| Ridge, log-target CV     |     19,830 ± 6,018 |    63,083 ± 72,812 |  -0.664 ± 3.108 | `expm1` amplified extreme predictions |
| Scaled Ridge, log target |     17,721 ± 4,425 |    55,665 ± 63,635 |  -0.282 ± 2.385 | Scaling helped, extreme rows remained |
| Scaled Ridge, raw target |       18,479 ± 924 |    33,620 ± 10,353 |   0.794 ± 0.161 | Strongest stable Ridge                |
| Random Forest            |     17,736 ± 1,791 | **30,194 ± 7,212** | **0.84 ± 0.10** | Best unfiltered RMSE/R²               |
| XGBoost                  | **17,550 ± 1,336** |     31,339 ± 7,202 |     0.83 ± 0.10 | Best mean MAE so far                  |

---

# Updated Key Lessons

## Model family matters for outlier behavior

The same unusual observations produced very different prediction errors depending on the model family.

Ridge produced severe extrapolation.

Random Forest and XGBoost both still overpredicted the two most extreme observations.

Baseline XGBoost still produced large overpredictions despite being nonlinear.

Therefore, simply describing a model as "tree-based" or "nonlinear" is insufficient to predict how it will behave on unusual observations.

## No single metric is enough

XGBoost currently has the best MAE.

Random Forest currently has the best RMSE and R² among unfiltered models.

This difference is meaningful rather than contradictory.

MAE describes typical absolute prediction error, while RMSE is much more sensitive to rare catastrophic errors.

Both are useful for understanding model behavior.

## Do not remove difficult observations prematurely

The experiments showed that changing the model family alone dramatically changed the prediction error on the problematic observations.

This is strong evidence against automatically deleting those rows simply because Ridge performed badly on them.

An explicit outlier-removal experiment may still be justified later, but it should be evaluated separately.

---

# Current Status

The project now has three meaningful model families:

1. Ridge regression
2. Random Forest
3. XGBoost

The strongest stable Ridge baseline is:

```text
Scaled numerical features
+ categorical one-hot encoding
+ raw SalePrice
+ Ridge
```

Random Forest currently provides the best combination of RMSE, R², and robustness to extreme observations.

XGBoost currently provides the best mean MAE.

No hyperparameter tuning or feature engineering has yet been applied to either nonlinear model.

---

# 11. XGBoost — Log-Target Experiment

## Hypothesis

`SalePrice` is strongly right-skewed, and previous experiments showed that extreme prediction errors can dominate RMSE.

Training XGBoost on `log1p(SalePrice)` may reduce the influence of expensive or unusual observations and improve stability.

The hypothesis was:

> Training XGBoost in log-target space may improve overall predictive performance and reduce the extreme overpredictions observed on the difficult Fold 3 observations.

This experiment changed only the target representation.

No feature engineering, outlier removal, or hyperparameter tuning was introduced.

## Experiment Design

The experiment used the same configuration as the raw-target XGBoost baseline.

Features:

* `SalePrice` excluded from `X`
* `Id` removed
* same numerical features
* same categorical features

Cross-validation:

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Numerical preprocessing:

* median imputation
* no scaling

Categorical preprocessing:

* most-frequent imputation
* one-hot encoding with unknown-category handling

Model:

```python
XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
```

The only experimental change was target handling:

```python
y_train_log = np.log1p(y_train)
```

The model was trained and predicted in log space.

Predictions were converted back to dollars using:

```python
y_pred = np.expm1(y_pred_log)
```

Dollar-scale MAE, RMSE, and R² were calculated after converting predictions back to the original `SalePrice` scale.

Log RMSE was calculated directly in log space.

## Results

### Fold-Level Results

| Fold |       MAE |      RMSE |   R² | Log RMSE |
| ---- | --------: | --------: | ---: | -------: |
| 1    | 17,175.15 | 26,259.47 | 0.91 |   0.1447 |
| 2    | 19,072.40 | 38,749.92 | 0.78 |   0.1457 |
| 3    | 20,117.45 | 41,971.67 | 0.68 |   0.1741 |
| 4    | 18,268.72 | 29,961.79 | 0.86 |   0.1479 |
| 5    | 15,533.46 | 24,183.12 | 0.89 |   0.1219 |

### Overall Results

| Metric   |      Mean | Standard Deviation |
| -------- | --------: | -----------------: |
| MAE      | 18,033.43 |           1,578.89 |
| RMSE     | 32,225.19 |           6,970.70 |
| R²       |      0.82 |               0.08 |
| Log RMSE |    0.1469 |             0.0166 |

## Raw vs. Log-Target XGBoost

| Metric |               Raw Target |           Log Target |
| ------ | -----------------------: | -------------------: |
| MAE    | **17,550.49 ± 1,335.94** | 18,033.43 ± 1,578.89 |
| RMSE   | **31,338.53 ± 7,201.69** | 32,225.19 ± 6,970.70 |
| R²     |          **0.83 ± 0.10** |          0.82 ± 0.08 |

The log-target transformation did not improve overall dollar-scale performance.

Compared with raw-target XGBoost:

* MAE increased by approximately $483.
* RMSE increased by approximately $887.
* mean R² decreased slightly from 0.83 to 0.82.
* RMSE variability decreased slightly.

Therefore, raw-target XGBoost remained the stronger XGBoost configuration on the main dollar-scale metrics.

## Fold 3 Diagnostics

The diagnostic predictions were taken directly from the saved Fold 3 out-of-fold predictions.

### Observation 1298

* Actual SalePrice: $160,000
* Raw XGBoost prediction: $606,205
* Log-target XGBoost prediction: $495,210
* Log-target absolute error: $335,210
* Actual `log1p(SalePrice)`: 11.9829
* Predicted log SalePrice: 13.1127

### Observation 523

* Actual SalePrice: $184,750
* Raw XGBoost prediction: $671,884
* Log-target XGBoost prediction: $639,674
* Log-target absolute error: $454,924
* Actual `log1p(SalePrice)`: 12.1268
* Predicted log SalePrice: 13.3687

### Observation 1324

* Actual SalePrice: $147,000
* Raw XGBoost prediction: $286,745
* Log-target XGBoost prediction: $303,343
* Log-target absolute error: $156,343
* Actual `log1p(SalePrice)`: 11.8982
* Predicted log SalePrice: 12.6226

## Comparison on Difficult Observations

| Index |   Actual | Raw XGBoost Error | Log XGBoost Error | Random Forest Error |
| ----- | -------: | ----------------: | ----------------: | ------------------: |
| 1298  | $160,000 |          $446,205 |      **$335,210** |        $428,331 |
| 523   | $184,750 |          $487,134 |          $454,924 |    **$405,087** |
| 1324  | $147,000 |      **$139,745** |          $156,343 |        $143,397 |

The log transformation reduced the extreme XGBoost errors for indices 1298 and 523.

However, it did not solve the underlying problem, and observation 1324 became slightly worse.

Random Forest was materially better than raw XGBoost on index 523, slightly better on index 1298, and slightly worse on index 1324.

## Interpretation

The hypothesis was only partially supported.

Training XGBoost in log space reduced some of the extreme Fold 3 overpredictions.

In particular:

* the error on index 1298 decreased by approximately $111,000,
* the error on index 523 decreased by approximately $32,000.

Fold 3 also improved:

* RMSE decreased from approximately $45,093 to $41,972,
* R² increased from 0.63 to 0.68.

However, these improvements did not translate into better overall cross-validation performance.

Across all five folds, raw-target XGBoost retained better:

* MAE,
* RMSE,
* and R².

The log transformation therefore introduced a tradeoff:

> It slightly reduced sensitivity to some extreme observations, but worsened average dollar-scale predictive performance.

This behavior also differed from the earlier log-target Ridge experiment.

With Ridge, extreme log-space predictions were strongly amplified by `expm1()`, producing catastrophic multi-million-dollar predictions.

XGBoost remained much more controlled in log space, but the transformation still did not outperform the raw-target version.

## Conclusion

The log-target transformation was not adopted as the preferred XGBoost configuration.

Raw-target XGBoost remains the strongest XGBoost baseline because it provides better overall dollar-scale performance.

The experiment was still useful because it demonstrated that target transformation can change robustness independently of average predictive accuracy.

---

# Updated Model Comparison

| Experiment               |                MAE |               RMSE |              R² | Main Finding                                |
| ------------------------ | -----------------: | -----------------: | --------------: | ------------------------------------------- |
| Dummy median             |             59,568 |             88,667 |          -0.025 | Performance floor                           |
| Ridge, single holdout    |             20,572 |             34,566 |           0.844 | Strong initial linear model                 |
| Ridge, raw-target CV     |     20,163 ± 2,291 |    35,717 ± 11,907 |   0.765 ± 0.191 | Revealed fold instability                   |
| Ridge, log-target CV     |     19,830 ± 6,018 |    63,083 ± 72,812 |  -0.664 ± 3.108 | `expm1` amplified extreme predictions       |
| Scaled Ridge, log target |     17,721 ± 4,425 |    55,665 ± 63,635 |  -0.282 ± 2.385 | Scaling helped, extreme rows remained       |
| Scaled Ridge, raw target |       18,479 ± 924 |    33,620 ± 10,353 |   0.794 ± 0.161 | Strongest stable Ridge                      |
| Random Forest            |     17,736 ± 1,791 | **30,194 ± 7,212** | **0.84 ± 0.10** | Best unfiltered RMSE and R²                 |
| Raw-target XGBoost       | **17,550 ± 1,336** |     31,339 ± 7,202 |     0.83 ± 0.10 | Best mean MAE                               |
| Log-target XGBoost       |     18,033 ± 1,579 |     32,225 ± 6,971 |     0.82 ± 0.08 | More robust on some extremes, worse overall |

---

# Updated Key Lessons

## Target transformations are model-dependent

A log-target transformation does not automatically improve a regression model simply because the target is right-skewed.

For Ridge, the transformation produced severe instability when extreme log predictions were converted back with `expm1()`.

For XGBoost, the transformation was much more stable and reduced some extreme errors, but still produced worse overall dollar-scale performance.

Target transformations therefore need to be evaluated empirically for each model family.

## Model family matters for extreme observations

The difficult observations behaved very differently across models.

Ridge produced severe extrapolation.

Raw and log-target XGBoost still substantially overpredicted the two most extreme houses.

Random Forest was better on some difficult observations, but all models still substantially overpredicted the two most extreme houses.

Model architecture therefore has a large effect on behavior outside the typical feature distribution.

## No single metric is sufficient

Raw-target XGBoost currently has the best mean MAE.

Random Forest currently has the best unfiltered RMSE and R², but does not uniformly have the smallest error on the known extreme observations.

This difference is meaningful.

MAE reflects typical absolute error, while RMSE strongly penalizes rare large mistakes.

Both are required to understand the current models.

## Difficult observations should not be deleted automatically

Changing the model family and target representation substantially changed the predictions for the problematic rows.

This confirms that poor performance on these observations is partly a modeling issue rather than sufficient evidence that the observations should be removed.

Outlier removal should therefore be tested as a separate controlled experiment rather than applied retroactively because certain rows produced large errors.

---

# 12. Training-Only Outlier Policy — `GrLivArea > 4000`

## Hypothesis

The two most extreme high-error observations also have extremely large `GrLivArea` values. This experiment tests whether excluding extremely large homes from model training improves generalization without making the validation population easier.

## Rule and Methodology

The rule was:

```text
GrLivArea > 4000
```

The rule uses only a predictor, not `SalePrice`, prediction error, or specific dataframe indices. Filtering was applied only within each training fold. Validation folds remained completely unchanged, so extreme homes remained in validation and still had to be predicted.

The same folds, preprocessing, models, and hyperparameters as the corrected baselines were used. No target transformation, feature engineering, or tuning was introduced.

Rows removed from training were:

| Fold | Rows removed |
| ---: | -----------: |
| 1 | 3 |
| 2 | 3 |
| 3 | 2 |
| 4 | 4 |
| 5 | 4 |

## Fold-Level Results

### Random Forest

| Fold | Baseline MAE / RMSE / R² | Filtered-training MAE / RMSE / R² |
| ---: | ------------------------ | --------------------------------- |
| 1 | 17,567.06 / 29,107.80 / 0.89 | 17,202.78 / 29,391.68 / 0.89 |
| 2 | 17,198.17 / 25,996.57 / 0.90 | 17,309.91 / 26,579.03 / 0.90 |
| 3 | 20,888.49 / 44,158.02 / 0.65 | 20,961.19 / 41,255.79 / 0.69 |
| 4 | 17,694.34 / 27,933.19 / 0.88 | 18,051.80 / 28,161.35 / 0.87 |
| 5 | 15,330.23 / 23,775.56 / 0.89 | 15,261.47 / 23,728.05 / 0.89 |

### XGBoost

| Fold | Baseline MAE / RMSE / R² | Filtered-training MAE / RMSE / R² |
| ---: | ------------------------ | --------------------------------- |
| 1 | 17,192.60 / 26,803.78 / 0.91 | 17,844.18 / 27,568.04 / 0.90 |
| 2 | 17,080.02 / 28,989.19 / 0.88 | 17,603.02 / 27,100.54 / 0.89 |
| 3 | 18,879.22 / 45,093.43 / 0.63 | 19,365.88 / 39,651.62 / 0.72 |
| 4 | 19,128.35 / 31,099.73 / 0.85 | 18,113.98 / 29,293.74 / 0.86 |
| 5 | 15,472.26 / 24,706.50 / 0.88 | 15,166.84 / 22,599.03 / 0.90 |

## Overall Comparison

| Configuration | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Random Forest baseline | 17,735.66 ± 1,791.40 | 30,194.23 ± 7,212.48 | 0.84 ± 0.10 |
| Random Forest filtered | 17,757.43 ± 1,848.70 | 29,823.18 ± 6,022.08 | 0.85 ± 0.08 |
| XGBoost baseline | **17,550.49 ± 1,335.94** | 31,338.53 ± 7,201.69 | 0.83 ± 0.10 |
| XGBoost filtered | 17,618.78 ± 1,368.26 | **29,242.59 ± 5,654.73** | **0.85 ± 0.07** |

## Fold 3 Diagnostics

All values below are genuine Fold 3 out-of-fold predictions. The extreme homes remained in validation.

| Index | Actual | RF baseline | RF filtered | XGBoost baseline | XGBoost filtered |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1298 | 160,000 | 588,331.29 (error 428,331.29) | 514,034.08 (error 354,034.08) | 606,205.38 (error 446,205.38) | 478,351.97 (error 318,351.97) |
| 523 | 184,750 | 589,836.59 (error 405,086.59) | 508,486.67 (error 323,736.67) | 671,884.06 (error 487,134.06) | 546,355.25 (error 361,605.25) |
| 1324 | 147,000 | 290,396.65 (error 143,396.65) | 284,210.11 (error 137,210.11) | 286,744.84 (error 139,744.84) | 295,660.25 (error 148,660.25) |

## Interpretation

Random Forest filtering left MAE essentially unchanged/slightly worse, improved RMSE modestly, increased mean R² from 0.84 to 0.85, and reduced RMSE/R² variability. Fold 3 extreme errors improved but remained large.

XGBoost filtering increased MAE by about $68, but improved RMSE by about $2,096 and mean R² from 0.83 to 0.85. RMSE variability fell substantially. In Fold 3, RMSE improved from about $45,093 to $39,652 and R² from 0.63 to 0.72; errors on indices 1298 and 523 also decreased substantially.

The training-only outlier policy therefore improves robustness, especially for XGBoost, without removing difficult observations from the prediction problem. It introduces a small MAE tradeoff while substantially improving XGBoost RMSE, R², and stability.

## Methodology Lesson

Experiment audits matter. The initial Random Forest experiment appeared more robust than it really was because `Id` was accidentally retained and its diagnostic predictions were in-sample. Correcting those issues changed the interpretation.

This reinforces the need to:

* check feature consistency across baselines,
* use genuine out-of-fold predictions for diagnostics,
* and verify that comparison scripts reproduce standalone baselines.

---

# Current Status

The project has compared Ridge, scaled/log-target Ridge, corrected Random Forest, raw-target XGBoost, log-target XGBoost, and training-only `GrLivArea > 4000` filtering for Random Forest and XGBoost.

The current strongest results are:

### Lowest Mean MAE

```text
Raw-target XGBoost
MAE: 17,550.49 ± 1,335.94
```

### Lowest Mean RMSE

```text
XGBoost with training-only GrLivArea > 4000 filtering
RMSE: 29,242.59 ± 5,654.73
```

### Highest Mean R²

```text
XGBoost filtered: 0.85 ± 0.07
Random Forest filtered: 0.85 ± 0.08
```

Filtered XGBoost currently provides the strongest RMSE/stability tradeoff, while raw-target XGBoost retains the lowest MAE. No final model has been selected, and no hyperparameter tuning or feature engineering has yet been applied to the nonlinear models.
