"""
metrics.py
----------
Métriques d'évaluation pour la régression.
"""
import numpy as np


def mse(y_true, y_pred):
    """Mean Squared Error : (1/n) · Σ(ŷ - y)²"""
    return float(np.mean((y_pred - y_true) ** 2))

def rmse(y_true, y_pred):
    """Root MSE : √MSE  — même unité que y"""
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred):
    """Mean Absolute Error : (1/n) · Σ|ŷ - y|"""
    return float(np.mean(np.abs(y_pred - y_true)))

def r2_score(y_true, y_pred):
    """
    Coefficient R² = 1 - SS_res/SS_tot
    Interprétation :
        1.0  → parfait
        0.0  → prédit juste la moyenne
        < 0  → pire que la moyenne
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)

def print_metrics(y_true, y_pred, split="Test"):
    print(f"\n{'='*42}")
    print(f"  📊 Métriques — Jeu de {split}")
    print(f"{'='*42}")
    print(f"  MSE   : {mse(y_true, y_pred):>18,.2f}")
    print(f"  RMSE  : {rmse(y_true, y_pred):>18,.2f}")
    print(f"  MAE   : {mae(y_true, y_pred):>18,.2f}")
    print(f"  R²    : {r2_score(y_true, y_pred):>18.4f}")
    print(f"{'='*42}\n")
