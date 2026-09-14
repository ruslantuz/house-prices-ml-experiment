# Project: Predicting SalePrice from train.csv

## Goal
Predict the `SalePrice` from the `train.csv` dataset.

## Data Inspection
The `inspect_data.py` script is used to inspect the dataset. It provides the following information:

- **Shape**: The number of rows and columns in the dataset.
- **Columns**: The names of the columns in the dataset.
- **Data Types**: The data types of each column.
- **Target Distribution**: The statistics of the `SalePrice` column.
- **Missing Values**: The number of missing values in each column.
- **Unique Values**: The number of unique values in each column.
- **Obvious ID Columns**: Columns that start with 'Id'.
- **Numerical Columns**: Columns with numerical data types.
- **Categorical Columns**: Columns with categorical data types.
- **Ordinal Columns**: Columns with a small number of unique values.
- **Suspicious Columns**: Columns with a large number of unique values.
- **Outliers**: Boxplots of numerical columns to check for outliers.
- **Missing Values in Categorical Columns**: The number of missing values in categorical columns.

## Next Steps
1. Analyze the data to understand its characteristics.
2. Identify potential features that may be useful for predicting `SalePrice`.
3. Preprocess the data to handle missing values, outliers, and categorical variables.
4. Split the dataset into training and testing sets.
5. Choose a machine learning model and train it on the training set.
6. Evaluate the model's performance on the testing set.
7. Fine-tune the model if necessary.
8. Document the findings and the process in the `PROJECT.md` file.

## Dependencies
The following dependencies are required to run the `inspect_data.py` script:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

## How to Run
1. Ensure that the `train.csv` file is in the same directory as the `inspect_data.py` script.
2. Run the `inspect_data.py` script using Python:

