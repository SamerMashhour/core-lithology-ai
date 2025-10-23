
# CoreLithoClassifier

A demonstration project integrating geology, machine learning, and image processing to classify core samples based on synthetic geochemical data and core imagery.

##  Project Objective

This project simulates how core imagery and associated geochemical data can be used together to build lithology classifiers. It mimics a simplified version of what core scanning AI systems aim to achieve.

##  Features

- Synthetic core imagery generator for Mafic, Ultramafic, and Felsic lithologies
- Geochemical data simulation (Ni, Cu, Fe, S)
- Image preprocessing using OpenCV and PIL
- Feature extraction (color histograms, edge detection, GLCM texture)
- Geochemistry-based PCA and clustering
- CNN model to classify lithology directly from images

## 🛠 Tech Stack

- Python
- NumPy, Pandas, Matplotlib
- Scikit-learn
- OpenCV, Pillow
- TensorFlow/Keras (for CNNs)

##  File Structure

```
CoreLithoClassifier/
├── data/
│   ├── sample_core_images/       # Synthetic core images
│   └── geochemistry.csv          # Synthetic geochemical dataset
├── notebooks/                    # Jupyter notebooks
├── output/
│   └── figures/                  # Output plots
├── src/                          # Python scripts for modeling & utilities
├── README.md
└── requirements.txt
```

##  Next Steps

- Explore geochemical data (PCA, clustering)
- Train CNN on core images
- Combine features for hybrid lithology prediction

---

**Disclaimer:** All data used in this project is synthetic and intended for demonstration purposes only.
