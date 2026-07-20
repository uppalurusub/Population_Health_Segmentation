"""
Population Health Patient Segmentation
HDBSCAN Clustering
"""

import pandas as pd
import numpy as np

import hdbscan

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(
    "data/patient_population_health.csv"
)

print("Dataset Shape:", df.shape)

# ==========================================
# Preprocessing
# ==========================================

df_model = df.copy()

encoder = LabelEncoder()

df_model["gender"] = encoder.fit_transform(
    df_model["gender"]
)

df_model.drop(
    columns=["patient_id"],
    inplace=True
)

# ==========================================
# Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df_model
)

# ==========================================
# HDBSCAN Model
# ==========================================

model = hdbscan.HDBSCAN(

    min_cluster_size=100,

    min_samples=20,

    metric="euclidean",

    cluster_selection_method="eom",

    prediction_data=True
)

labels = model.fit_predict(
    X_scaled
)

df["cluster"] = labels

# ==========================================
# Summary
# ==========================================

n_clusters = len(
    set(labels)
) - (1 if -1 in labels else 0)

n_noise = np.sum(
    labels == -1
)

print(
    f"Clusters Found: {n_clusters}"
)

print(
    f"Noise Patients: {n_noise}"
)

#Cluster Evaluation. Exclude noise points (-1).
mask = labels != -1

X_eval = X_scaled[mask]

labels_eval = labels[mask]

if len(set(labels_eval)) > 1:

    silhouette = silhouette_score(
        X_eval,
        labels_eval
    )

    davies = davies_bouldin_score(
        X_eval,
        labels_eval
    )

    calinski = calinski_harabasz_score(
        X_eval,
        labels_eval
    )

    print()

    print(
        f"Silhouette Score: "
        f"{silhouette:.4f}"
    )

    print(
        f"Davies-Bouldin Score: "
        f"{davies:.4f}"
    )

    print(
        f"Calinski-Harabasz Score: "
        f"{calinski:.4f}"
    )

#Cluster Distribution
cluster_distribution = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
)

print(cluster_distribution)


#Cluster Profiling
cluster_profile = df.groupby(
    "cluster"
).agg({

    "age":"mean",
    "bmi":"mean",

    "diabetes":"mean",
    "hypertension":"mean",
    "copd":"mean",

    "mental_health":"mean",

    "visits_per_year":"mean",
    "er_visits":"mean",
    "admissions":"mean",

    "total_cost":"mean"

})

print(cluster_profile)


#Cluster Confidence Scores
probabilities = model.probabilities_

df["cluster_confidence"] = (
    probabilities
)

print(
    df[
        ["cluster",
         "cluster_confidence"]
    ].head()
)

#Identify Outlier Patients
outliers = df[
    df["cluster"] == -1
]

print(
    "Outlier Patients:",
    len(outliers)
)

#PCA Visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure(figsize=(12,8))

scatter = plt.scatter(

    X_pca[:,0],
    X_pca[:,1],

    c=labels,

    cmap="tab20",

    alpha=0.8
)

plt.title(
    "HDBSCAN Patient Segmentation"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)

plt.show()



# =====================================================
# Save Model
# =====================================================
import joblib
joblib.dump(
    model,
    "saved_models/hdbscan.pkl"
)

print("\nModel Saved Successfully")

#Predicting New Patients
from hdbscan import (
    approximate_predict
)

new_patient = scaler.transform(
    patient_df
)

cluster, strength = (
    approximate_predict(
        model,
        new_patient
    )
)

print(
    "Cluster:",
    cluster[0]
)

print(
    "Confidence:",
    strength[0]
)