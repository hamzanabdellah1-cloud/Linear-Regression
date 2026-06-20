"""
data_loader.py
--------------
Chargement, nettoyage et préparation des données.
"""
import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def clean_data(df: pd.DataFrame, feature_col: str, target_col: str) -> pd.DataFrame:
    """
    Nettoie les données :
    1. Sélectionne les colonnes utiles
    2. Supprime les NaN
    3. Supprime les doublons
    4. Retire les outliers via méthode IQR
    """
    df = df[[feature_col, target_col]].copy()
    initial = len(df)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    for col in [feature_col, target_col]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    df = df.reset_index(drop=True)
    print(f"[INFO] Nettoyage : {initial - len(df)} lignes supprimées → {len(df)} restantes")
    return df


def prepare_arrays(df, feature_col, target_col, test_size=0.2, random_state=42):
    """
    Convertit pandas -> NumPy et sépare train/test manuellement.
    Retourne : X_train, X_test, y_train, y_test (arrays 1D)
    """
    X = df[feature_col].to_numpy()
    y = df[target_col].to_numpy()

    np.random.seed(random_state)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test
