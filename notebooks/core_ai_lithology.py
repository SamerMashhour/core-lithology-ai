# 📁 File: notebooks/2_pca_geochem_analysis.ipynb
# Description: This notebook performs PCA on the synthetic geochemical dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load geochemical data
geochem_path = '../data/geochemistry.csv'
df = pd.read_csv(geochem_path)

# Optional: Encode Lithology numerically
lithology_mapping = {"Mafic": 0, "Ultramafic": 1, "Felsic": 2}
df["Lithology_Code"] = df["Lithology"].map(lithology_mapping)

# Select geochemical features
features = ["Ni_ppm", "Cu_ppm", "Fe_pct", "S_pct"]
X = df[features].values
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df["PC1"] = components[:, 0]
df["PC2"] = components[:, 1]

# Plot PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="PC1", y="PC2",
    hue="Lithology",
    palette="Set2",
    data=df,
    s=80,
    edgecolor='black'
)
plt.title("PCA of Geochemical Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
