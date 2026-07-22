"""
Data preprocessing pipeline for the Heart Disease dataset.

Handles:
- Missing value imputation
- Categorical variable encoding
- Numerical feature standardization
- Train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(filepath: str = "data/Heart.csv") -> pd.DataFrame:
    """
    Load the Heart Disease dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with the unnamed index column dropped.
    """
    df = pd.read_csv(filepath)
    # Drop the unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Return basic information about the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    dict
        Dictionary containing shape, dtypes, missing counts, and descriptive stats.
    """
    info = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
        "target_distribution": df["AHD"].value_counts().to_dict() if "AHD" in df.columns else None,
    }
    return info


def get_feature_lists() -> tuple:
    """
    Return lists of numerical and categorical feature names.

    Returns
    -------
    tuple of (list, list)
        (numerical_features, categorical_features)
    """
    numerical_features = ["Age", "RestBP", "Chol", "MaxHR", "Oldpeak"]
    categorical_features = ["Sex", "ChestPain", "Fbs", "RestECG", "ExAng", "Slope", "Ca", "Thal"]
    return numerical_features, categorical_features


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build a preprocessing pipeline for numerical and categorical features.

    Numerical: median imputation + standardization.
    Categorical: most-frequent imputation + one-hot encoding.

    Returns
    -------
    ColumnTransformer
        A scikit-learn ColumnTransformer pipeline.
    """
    numerical_features, categorical_features = get_feature_lists()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    target_col: str = "AHD",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split data into train/test sets and apply preprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    target_col : str
        Name of the target column.
    test_size : float
        Proportion of data to use as test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Get the feature names after preprocessing (one-hot encoding expanded).

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.

    Returns
    -------
    list
        List of feature names after transformation.
    """
    numerical_features, categorical_features = get_feature_lists()

    # Get one-hot encoded feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_encoded_names = cat_encoder.get_feature_names_out(categorical_features).tolist()

    all_features = numerical_features + cat_encoded_names
    return all_features
