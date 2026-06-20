"""
model.py
--------
Régression linéaire univariée - solution analytique (Normal Equation).

Formule : θ = (X^T · X)^(-1) · X^T · y
Résultat : θ = [b, a]  →  y = a·x + b
"""
import numpy as np


class LinearRegressionAnalytical:
    """
    Régression linéaire from-scratch, sans scikit-learn.

    Paramètres après fit() :
        self.a     : pente
        self.b     : ordonnée à l'origine (biais)
        self.theta : vecteur [b, a]
    """

    def __init__(self):
        self.a = None
        self.b = None
        self.theta = None

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Ajoute une colonne de 1 pour modéliser le terme constant b.

        X = [40, 80, 120]  →  X_b = [[1, 40],
                                       [1, 80],
                                       [1, 120]]
        """
        ones = np.ones((len(X), 1))
        return np.hstack([ones, X.reshape(-1, 1)])  # shape (n, 2)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entraîne le modèle avec l'équation normale :
            θ = (X^T X)^(-1) · X^T · y

        Étapes :
            1. Ajouter colonne biais → X_b  (n, 2)
            2. Calculer X^T · X            (2, 2)
            3. Inverser (X^T X)            (2, 2)
            4. Calculer X^T · y            (2,)
            5. Multiplier pour obtenir θ   (2,)
        """
        X_b = self._add_bias(X_train)            # (n, 2)
        XtX     = X_b.T @ X_b                    # (2, 2)
        XtX_inv = np.linalg.inv(XtX)             # (2, 2)
        Xty     = X_b.T @ y_train                # (2,)
        self.theta = XtX_inv @ Xty               # (2,) = [b, a]

        self.b = self.theta[0]
        self.a = self.theta[1]

        print(f"[MODEL] Entraîné !  →  {self.get_equation()}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit y = a·x + b pour de nouvelles données."""
        if self.theta is None:
            raise ValueError("Modèle non entraîné. Appelez fit() d'abord.")
        X_b = self._add_bias(np.atleast_1d(np.array(X, dtype=float)))
        return X_b @ self.theta

    def get_equation(self) -> str:
        """Retourne l'équation lisible de la droite."""
        if self.theta is None:
            return "Modèle non entraîné"
        sign = "+" if self.b >= 0 else "-"
        return f"y = {self.a:.2f}·x {sign} {abs(self.b):.2f}"
