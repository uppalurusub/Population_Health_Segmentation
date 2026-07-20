# app/ml/train_model.py

import joblib
import skops.io as sio
import mlflow
import numpy as np
import mlflow.sklearn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="mlflow",
    message=".*Saving scikit-learn models in the pickle or cloudpickle format.*"
    )
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.CRITICAL)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


from sklearn.preprocessing import StandardScaler
import hdbscan

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import os
os.makedirs("best_saved_models",exist_ok=True)
# =====================================================
# MLFlow Tracking
# =====================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "Population_Health_Clustering"

MODEL_OUTPUT_PATH = "best_saved_models/phs_model.pkl"

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

# =========================================================
# Set Experiment
# =========================================================

experiment = mlflow.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    experiment_id = mlflow.create_experiment(
        EXPERIMENT_NAME
    )
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(
    EXPERIMENT_NAME
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


# =====================================================
# Model
# =====================================================
"""
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)"""

n_samples = len(X_scaled)

min_cluster_size = min(
    50,
    max(5, n_samples // 10)
)

min_samples = min(
    10,
    max(2, n_samples // 20)
)

print(n_samples)
print(min_cluster_size)
print(min_samples)

model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True
        )




with mlflow.start_run() as run:

    model.fit(X_scaled)

    labels = model.fit_predict(X_scaled)

    # ==================================
    # REMOVE NOISE
    # ==================================

    mask = labels != -1

    X_eval = X_scaled[mask]
    y_eval = labels[mask]

    if len(np.unique(y_eval)) < 2:

        print("Skipped")

        #continue

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
    
    
    mlflow.sklearn.log_model(
        sk_model=model,
        name="phs_model",
        serialization_format="cloudpickle"
        #registered_model_name="PopulationHealthSegmentationModel"
    )

    run_id = run.info.run_id
    MODEL_NAME = "PopulationHealthSegmentationModel"
    model_uri = f"runs:/{run_id}/phs_model"

    print("Model URI:", model_uri)


    
    #Register Best Model
    mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    
# =====================================================
# Save Model
# =====================================================
joblib.dump(
    model,
    "best_saved_models/phs_model.pkl"
)

joblib.dump(
    scaler,
    "best_saved_models/scaler.pkl"
)

joblib.dump(
    encoder,
    "best_saved_models/encoder.pkl"
)


print("Model Saved Successfully")