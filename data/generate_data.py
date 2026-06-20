"""
Script pour générer un dataset synthétique réaliste
Surface (m²) → Prix (MAD)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

surface = np.random.uniform(40, 250, n)
# Relation réelle : Prix ≈ 8500 * surface + 150_000 + bruit
noise = np.random.normal(0, 80_000, n)
prix = 8500 * surface + 150_000 + noise

df = pd.DataFrame({
    "surface_m2": np.round(surface, 1),
    "prix_mad": np.round(prix, 0).astype(int)
})

df.to_csv("data/raw/housing.csv", index=False)
print(f"Dataset généré : {len(df)} lignes")
print(df.describe())
