# 📁 File: notebooks/5_hybrid_model.ipynb
# Description: Combines geochemistry + image features for hybrid lithology classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load geochemistry and image feature data
df_geo = pd.read_csv('../data/geochemistry.csv')
df_img = pd.read_csv('../data/image_features.csv')

# Merge datasets on SampleID
df_merged = pd.merge(df_geo, df_img, on='SampleID')

# Encode target labels
target_map = {"Mafic": 0, "Ultramafic": 1, "Felsic": 2}
df_merged["target"] = df_merged["Lithology"].map(target_map)

# Define input features
features = ["Ni_ppm", "Cu_ppm", "Fe_pct", "S_pct"] + [col for col in df_img.columns if col.startswith("hist_") or col in ["contrast", "homogeneity", "entropy"]]
X = df_merged[features]
y = df_merged["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_map.keys()))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=target_map.keys(), cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
importances.sort_values(ascending=False).head(15).plot(kind='barh')
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances (Geochem + Image)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()
