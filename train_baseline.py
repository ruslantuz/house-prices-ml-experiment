import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path):
    return pd.read_csv(file_path)

def train_baseline():
    # Load the dataset
    df = load_data('train.csv')
    
    # Drop Id from the features
    features = df.drop(columns=['Id'])
    
    # Target variable
    target = 'SalePrice'
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Define preprocessing steps
    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, features.select_dtypes(include=[np.number]).columns),
            ('cat', categorical_transformer, features.select_dtypes(include=["object", "string", "category"]).columns)
        ]
    )
    
    # Define the model
    model = DummyRegressor(strategy="median")
    
    # Create the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    # Fit the model on the training data
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model on the validation data
    y_pred = pipeline.predict(X_val)
    
    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    
    # Print the metrics
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    train_baseline()
