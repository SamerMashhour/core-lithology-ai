# 📁 File: src/image_features.py
# Description: Extracts color histogram and GLCM texture features from core images

import os
import numpy as np
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.io import imread


def extract_features_from_image(img_path):
    img = imread(img_path)
    gray = rgb2gray(img)
    gray_uint8 = (gray * 255).astype('uint8')

    # GLCM texture features
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    # Color histogram features
    hist_r = np.histogram(img[:, :, 0], bins=16, range=(0, 255))[0]
    hist_g = np.histogram(img[:, :, 1], bins=16, range=(0, 255))[0]
    hist_b = np.histogram(img[:, :, 2], bins=16, range=(0, 255))[0]

    hist_features = np.concatenate([hist_r, hist_g, hist_b])
    hist_features = hist_features / np.sum(hist_features)  # Normalize

    return {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'entropy': entropy,
        **{f'hist_{i}': val for i, val in enumerate(hist_features)}
    }


def process_all_images(image_dir):
    feature_list = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            sample_id = filename.replace(".png", "")
            features = extract_features_from_image(os.path.join(image_dir, filename))
            features['SampleID'] = sample_id
            feature_list.append(features)
    return pd.DataFrame(feature_list)


if __name__ == "__main__":
    image_dir = "../data/sample_core_images"
    output_csv = "../data/image_features.csv"
    df = process_all_images(image_dir)
    df.to_csv(output_csv, index=False)
    print(f"Extracted features saved to {output_csv}")
