import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocesses the data: handle missing values, feature scaling, etc."""
    # Assuming no missing values in this dataset
    
    # Split the data into features and labels
    X = df.drop(columns=['status'])  # Assuming 'status' is the target column
    y = df['status']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the data into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
