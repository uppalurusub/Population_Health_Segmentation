"""
Population Health Patient Segmentation
Fuzzy C-Means Clustering
"""

import pandas as pd
import numpy as np
import joblib
import skfuzzy as fuzz

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv(
    "data/patient_population_health.csv"
)

print(
    f"Dataset Shape: {df.shape}"
)

# ======================================
# Preprocessing
# ======================================

df_model = df.copy()

encoder = LabelEncoder()

df_model["gender"] = encoder.fit_transform(
    df_model["gender"]
)

df_model.drop(
    columns=["patient_id"],
    inplace=True
)

# ======================================
# Scaling
# ======================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df_model
)

# ======================================
# Fuzzy C-Means expects:
# features x samples
# ======================================

X_fuzzy = X_scaled.T

# ======================================
# Train FCM
# ======================================

n_clusters = 8

cntr, membership_matrix, _, _, _, _, fpc = (
    fuzz.cluster.cmeans(

        X_fuzzy,

        c=n_clusters,

        m=2.0,

        error=0.005,

        maxiter=1000,

        init=None,

        seed=42
    )
)

# ======================================
# Hard Cluster Assignment
# ======================================

labels = np.argmax(
    membership_matrix,
    axis=0
)

df["cluster"] = labels

print(
    f"\nFuzzy Partition Coefficient: {fpc:.4f}"
)

#Evaluation
silhouette = silhouette_score(
    X_scaled,
    labels
)

davies = davies_bouldin_score(
    X_scaled,
    labels
)

calinski = calinski_harabasz_score(
    X_scaled,
    labels
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

membership_df = pd.DataFrame(

    membership_matrix.T,

    columns=[
        f"Cluster_{i}"
        for i in range(n_clusters)
    ]
)

print(
    membership_df.head()
)

#Add Confidence Score
df["confidence"] = np.max(
    membership_matrix,
    axis=0
)

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

#Find Optimal Number of Clusters
fpcs = []

cluster_range = range(2,15)

for k in cluster_range:

    _, _, _, _, _, _, fpc = (

        fuzz.cluster.cmeans(

            X_fuzzy,

            c=k,

            m=2,

            error=0.005,

            maxiter=1000
        )
    )

    fpcs.append(fpc)

    print(
        f"K={k} FPC={fpc:.4f}"
    )

#Plot FPC Scores
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.plot(

    cluster_range,

    fpcs,

    marker="o"
)

plt.title(
    "Fuzzy Partition Coefficient"
)

plt.xlabel(
    "Number of Clusters"
)

plt.ylabel(
    "FPC"
)

plt.grid(True)

plt.show()

#Predict New Patient
patient = {

    "age":72,

    "gender":"F",

    "bmi":33,

    "diabetes":1,

    "hypertension":1,

    "copd":0,

    "mental_health":0,

    "visits_per_year":15,

    "er_visits":3,

    "admissions":2,

    "total_cost":35000
}

patient_df = pd.DataFrame(
    [patient]
)

patient_df["gender"] = (
    encoder.transform(
        patient_df["gender"]
    )
)

patient_scaled = scaler.transform(
    patient_df
)

u, _, _, _, _, _ = (
    fuzz.cluster.cmeans_predict(

        patient_scaled.T,

        cntr,

        m=2,

        error=0.005,

        maxiter=1000
    )
)

cluster = np.argmax(u)

confidence = np.max(u)

print(
    "Cluster:",
    cluster
)

print(
    "Confidence:",
    confidence
)


#Save Model Artifacts
joblib.dump(
    cntr,
    "saved_models/fcm_centers.pkl"
)