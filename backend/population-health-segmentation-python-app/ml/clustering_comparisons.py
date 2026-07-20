#Best Model Selection Strategy
"""
Best_Score = (
    silhouette_score
    +
    normalized_calinski_score
    -
    normalized_davies_score
)

Priority:

1. Highest Silhouette Score
2. Lowest Davies Bouldin Score
3. Highest Calinski Harabasz Score
4. Lowest Noise Percentage

"""
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    OPTICS,
    SpectralClustering
)
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.CRITICAL)

from sklearn.mixture import GaussianMixture

import hdbscan
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

import os
os.makedirs("saved_models",exist_ok=True)

# ====================================================
# MLFLOW
# ====================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Population_Health_Clustering"
)

# ====================================================
# LOAD DATA
# ====================================================

df = pd.read_csv(
    "../data/patient_population_health.csv"
)

df_model = df.copy()

encoder = LabelEncoder()

df_model["gender"] = encoder.fit_transform(
    df_model["gender"]
)

df_model.drop(
    columns=["patient_id"],
    inplace=True
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df_model
)

# ====================================================
# MODELS
# ====================================================

models = {

    "KMeans":
        KMeans(
            n_clusters=8,
            random_state=42,
            n_init=20
        ),

    "Agglomerative":
        AgglomerativeClustering(
            n_clusters=8
        ),

    "Birch":
        Birch(
            n_clusters=8
        ),

    "DBSCAN":
        DBSCAN(
            eps=1.2,
            min_samples=25
        ),

    "OPTICS":
        OPTICS(
            min_samples=50
        ),

    "Spectral":
        SpectralClustering(
            n_clusters=8,
            random_state=42
        ),

    "GMM":
        GaussianMixture(
            n_components=8,
            random_state=42
        ),

    "HDBSCAN":
        hdbscan.HDBSCAN(
            min_cluster_size=100,
            prediction_data=True
        )
}

# ====================================================
# RESULTS
# ====================================================

results = []

best_model = None
best_score = -999999

# ====================================================
# TRAIN
# ====================================================

for model_name, model in models.items():

    print(
        f"\nTraining {model_name}"
    )

    with mlflow.start_run(
        run_name=model_name
    ):

        try:

            # ==================================
            # FIT
            # ==================================

            if model_name == "GMM":

                model.fit(X_scaled)

                labels = model.predict(
                    X_scaled
                )

            else:

                labels = model.fit_predict(
                    X_scaled
                )

            # ==================================
            # REMOVE NOISE
            # ==================================

            mask = labels != -1

            X_eval = X_scaled[mask]
            y_eval = labels[mask]

            if len(
                np.unique(y_eval)
            ) < 2:

                print(
                    "Skipped"
                )

                continue

            # ==================================
            # METRICS
            # ==================================

            silhouette = silhouette_score(
                X_eval,
                y_eval
            )

            davies = davies_bouldin_score(
                X_eval,
                y_eval
            )

            calinski = calinski_harabasz_score(
                X_eval,
                y_eval
            )

            final_score = (
                silhouette
                -
                davies
                +
                np.log1p(calinski)
            )

            mlflow.log_metric(
                "silhouette_score",
                silhouette
            )

            mlflow.log_metric(
                "davies_bouldin_score",
                davies
            )

            mlflow.log_metric(
                "calinski_harabasz_score",
                calinski
            )

            mlflow.log_metric(
                "final_score",
                final_score
            )

            # ==================================
            # SAVE MODEL
            # ==================================
            
            model_path = (
                f"saved_models/"
                f"{model_name}.pkl"
            )

            joblib.dump(
                model,
                model_path
            )

            mlflow.log_artifact(
                model_path
            )

            if final_score > best_score:

                best_score = final_score
                best_model = model_name
                best_model_name = model_name
                
                print(f"\nBest Model: {best_model_name}")
                print(f"Best Score: {best_score:.4f}")

                
            results.append({

                "Model":
                    model_name,

                "Silhouette":
                    silhouette,

                "Davies":
                    davies,

                "Calinski":
                    calinski,

                "FinalScore":
                    final_score
            })

        except Exception as e:

            print(
                model_name,
                str(e)
            )

# ====================================================
# RESULTS
# ====================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "FinalScore",
    ascending=False
)

print(
    "\nMODEL COMPARISON"
)

print(
    results_df
)

print(
    "\nBEST MODEL:",
    best_model
)

print(
    "BEST SCORE:",
    best_score
)

results_df.to_csv(
    "saved_models/clustering_comparison.csv",
    index=False
)


#Best Model Registration
with mlflow.start_run(
    run_name="Best_Clustering_Model"
) as run:

    mlflow.log_param(
        "best_model_name",
        best_model_name
    )

    mlflow.log_metric(
        "best_score",
        best_score
    )

    mlflow.sklearn.log_model(

        sk_model=best_model,

        artifact_path="model",

        registered_model_name=
        "PopulationHealthBestClusteringModel"
    )

    model_uri = (
        f"runs:/{run.info.run_id}/model"
    )

    print(
        "Best Model URI:",
        model_uri
    )

#print("Saving Best Model: ")
#joblib.dump(
#    best_model,
#    f"saved_models/{best_model}.pkl"
#)

# ====================================================
# SAVE PREPROCESSOR
# ====================================================

joblib.dump(
    scaler,
    "saved_models/scaler.pkl"
)

joblib.dump(
    encoder,
    "saved_models/encoder.pkl"
)