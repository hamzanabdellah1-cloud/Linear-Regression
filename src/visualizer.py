"""
visualizer.py
-------------
Graphiques pour analyser et valider le modèle.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_data(X, y, title="Distribution des données", xlabel="X", ylabel="y", save_path=None):
    """Nuage de points brut."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, alpha=0.6, color="#3498db", edgecolors="white", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[VIZ] Sauvegardé : {save_path}")
    plt.close()


def plot_regression_line(X_train, y_train, X_test, y_test, model,
                         xlabel="X", ylabel="y", save_path=None):
    """Droite de régression sur train + test."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(X_train, y_train, alpha=0.5, color="#3498db", label="Train", s=40)
    ax.scatter(X_test,  y_test,  alpha=0.8, color="#e74c3c", label="Test",  s=50, marker="^")

    # Droite de régression
    x_line = np.linspace(min(np.concatenate([X_train, X_test])),
                         max(np.concatenate([X_train, X_test])), 200)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color="#2ecc71", linewidth=2.5, label=f"Droite : {model.get_equation()}")

    ax.set_title("Régression Linéaire — Résultat", fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[VIZ] Sauvegardé : {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, split="Test", save_path=None):
    """Graphique des résidus (erreurs) → doit être centré autour de 0."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Résidus vs valeurs prédites
    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#9b59b6", edgecolors="white", linewidth=0.5)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Résidus vs Prédictions", fontweight="bold")
    axes[0].set_xlabel("Valeurs prédites"); axes[0].set_ylabel("Résidus")
    axes[0].grid(True, alpha=0.3)

    # Distribution des résidus (doit être gaussienne)
    axes[1].hist(residuals, bins=20, color="#f39c12", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Distribution des résidus", fontweight="bold")
    axes[1].set_xlabel("Résidu"); axes[1].set_ylabel("Fréquence")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Analyse des résidus — {split}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[VIZ] Sauvegardé : {save_path}")
    plt.close()
