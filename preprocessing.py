import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(train_path: str, test_path: str):
    """
    Load training and testing datasets.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(df: pd.DataFrame):
    drop_cols = ["trans_date_trans_time", "cc_num", "first", "last",
                 "street", "dob", "trans_num", "unix_time", "merchant"]

    df = df.drop(columns=drop_cols, errors="ignore")
    
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"], errors="ignore")

    return X, y

def create_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return preprocessor
