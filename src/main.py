"""
main.py
-------
Point d'entrée principal du projet.
Lance le pipeline complet : chargement → nettoyage → entraînement → évaluation → visualisation.

Usage :
    python src/main.py
"""

import os
import sys

# Ajouter le dossier racine au path Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, clean_data, prepare_arrays
from src.model      import LinearRegressionAnalytical
from src.metrics    import print_metrics, r2_score
from src.visualizer import plot_data, plot_regression_line, plot_residuals

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH   = "data/raw/housing.csv"
FEATURE_COL = "surface_m2"
TARGET_COL  = "prix_mad"
OUTPUT_DIR  = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Pipeline ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("  Régression Linéaire Analytique — Pipeline")
    print("="*50)

    # 1. Chargement
    df = load_data(DATA_PATH)

    # 2. Nettoyage
    df_clean = clean_data(df, FEATURE_COL, TARGET_COL)
    df_clean.to_csv("data/processed/housing_clean.csv", index=False)

    # 3. Visualiser les données brutes
    plot_data(df_clean[FEATURE_COL], df_clean[TARGET_COL],
              title="Surface vs Prix",
              xlabel="Surface (m²)", ylabel="Prix (MAD)",
              save_path=f"{OUTPUT_DIR}/01_data_scatter.png")

    # 4. Préparation des arrays NumPy
    X_train, X_test, y_train, y_test = prepare_arrays(
        df_clean, FEATURE_COL, TARGET_COL, test_size=0.2
    )

    # 5. Entraînement du modèle
    model = LinearRegressionAnalytical()
    model.fit(X_train, y_train)

    # 6. Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # 7. Métriques
    print_metrics(y_train, y_pred_train, split="Entraînement")
    print_metrics(y_test,  y_pred_test,  split="Test")

    # 8. Visualisations
    plot_regression_line(X_train, y_train, X_test, y_test, model,
                         xlabel="Surface (m²)", ylabel="Prix (MAD)",
                         save_path=f"{OUTPUT_DIR}/02_regression_line.png")

    plot_residuals(y_test, y_pred_test, split="Test",
                   save_path=f"{OUTPUT_DIR}/03_residuals.png")

    # 9. Résumé final
    r2 = r2_score(y_test, y_pred_test)
    print(f"\n✅ Pipeline terminé !")
    print(f"   Équation apprise  : {model.get_equation()}")
    print(f"   R² sur test       : {r2:.4f}  ({'Excellent' if r2 > 0.9 else 'Bon' if r2 > 0.7 else 'À améliorer'})")
    print(f"   Graphiques → ./{OUTPUT_DIR}/\n")

    # 10. Exemple de prédiction
    exemple = 100
    pred = model.predict([exemple])[0]
    print(f"   Exemple : surface = {exemple} m²  →  prix prédit = {pred:,.0f} MAD")
    print()


if __name__ == "__main__":
    main()
