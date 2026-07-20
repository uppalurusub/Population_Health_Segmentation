"""
Population Health Patient Segmentation
Gaussian Mixture Models (GMM)
"""

import pandas as pd
import joblib

from sklearn.mixture import GaussianMixture

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
# Load Data
# ==========================================

df = pd.read_csv(
    "data/patient_population_health.csv"
)

print(
    f"Dataset Shape: {df.shape}"
)

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
# Train GMM
# ==========================================

gmm = GaussianMixture(

    n_components=8,

    covariance_type="full",

    random_state=42,

    n_init=10
)

gmm.fit(
    X_scaled
)

# ==========================================
# Cluster Assignment
# ==========================================

labels = gmm.predict(
    X_scaled
)

df["cluster"] = labels

# ==========================================
# Evaluation
# ==========================================

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

print("\nModel Evaluation")

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

#Soft Clustering Probabilities
probabilities = gmm.predict_proba(
    X_scaled
)

print(
    probabilities[:5]
)

#Assign Confidence Score
import numpy as np

confidence = np.max(
    probabilities,
    axis=1
)

df["cluster_confidence"] = (
    confidence
)

#Find Optimal Number of Clusters Using BIC
import matplotlib.pyplot as plt

bic_scores = []

cluster_range = range(2,15)

for k in cluster_range:

    model = GaussianMixture(

        n_components=k,

        random_state=42
    )

    model.fit(X_scaled)

    bic_scores.append(
        model.bic(X_scaled)
    )

plt.figure(figsize=(10,6))

plt.plot(

    cluster_range,

    bic_scores,

    marker="o"
)

plt.title(
    "BIC Score vs Number of Clusters"
)

plt.xlabel(
    "Number of Clusters"
)

plt.ylabel(
    "BIC Score"
)

plt.grid(True)

plt.show()

#Alternative: AIC
aic_scores = []

for k in range(2,15):

    model = GaussianMixture(
        n_components=k,
        random_state=42
    )

    model.fit(X_scaled)

    aic_scores.append(
        model.aic(X_scaled)
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

plt.figure(figsize=(10,6))

plt.scatter(

    X_pca[:,0],

    X_pca[:,1],

    c=labels,

    cmap="tab10"

)

plt.title(
    "GMM Patient Segmentation"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar()

plt.show()

# ==========================================
# Save Model
# ==========================================

joblib.dump(
    gmm,
    "saved_models/gmm.pkl"
)



print(
    "\nModel Saved Successfully"
)