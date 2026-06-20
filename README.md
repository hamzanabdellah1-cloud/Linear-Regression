# Régression Linéaire Analytique — From Scratch

> Implémentation d'un modèle de régression linéaire univariée **sans framework ML**, en utilisant uniquement NumPy et Pandas.

---

## 🎯 Objectif

Estimer la relation **y = a·x + b** entre deux variables (ex: superficie → prix immobilier) en calculant les paramètres `a` et `b` via la **solution analytique (Normal Equation)** :

```
θ = (XᵀX)⁻¹ · Xᵀ · y
```

---

## 📁 Structure du projet

```
linear-regression-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/housing.csv           ← données brutes
│   └── processed/housing_clean.csv ← données nettoyées
│
├── outputs/                      ← graphiques générés
│
└── src/
    ├── data_loader.py   ← chargement, nettoyage, split train/test
    ├── model.py         ← modèle (Normal Equation) ← CŒUR DU PROJET
    ├── metrics.py       ← MSE, RMSE, MAE, R²
    ├── visualizer.py    ← graphiques matplotlib
    └── main.py          ← pipeline complet
```

---

## 🚀 Installation & Utilisation

```bash
# 1. Cloner le repo
git clone https://github.com/TON_USERNAME/linear-regression-from-scratch.git
cd linear-regression-from-scratch

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer le pipeline complet
python src/main.py
```

---

## 🧮 Concepts mathématiques

| Concept         | Description                                            |
| --------------- | ------------------------------------------------------ |
| Normal Equation | θ = (XᵀX)⁻¹Xᵀy — solution directe, sans itération      |
| Terme biais     | Colonne de 1 ajoutée à X pour modéliser la constante b |
| MSE             | Erreur quadratique moyenne                             |
| R²              | Proportion de variance expliquée par le modèle         |

---

## 📊 Résultats

Après entraînement sur le dataset housing (surface m² → prix MAD) :

- **Équation** : y ≈ 8500·x + 150000
- **R² (test)** : ~0.97

---

## 🛠️ Outils utilisés

- **Python 3.10+**
- **NumPy** — algèbre linéaire, opérations matricielles
- **Pandas** — chargement et nettoyage des données
- **Matplotlib** — visualisations

---

## 💡 Points clés du projet

1. **Aucun framework ML** (pas de scikit-learn) → compréhension des fondamentaux
2. **Normal Equation** vs Gradient Descent : solution exacte en une étape
3. **Nettoyage robuste** : gestion NaN, doublons, outliers (méthode IQR)
4. **Validation** : split train/test, métriques multiples, analyse des résidus
