# Population Health Segmentation

## Overview

Population Health Segmentation is an end-to-end Machine Learning application that groups patients into clinically meaningful population segments using unsupervised learning (clustering). The solution helps healthcare organizations identify high-risk patient populations, optimize resource allocation, improve preventive care, reduce healthcare costs, and support value-based healthcare initiatives.

The project is built using a modular architecture with FastAPI for REST APIs, Streamlit for the user interface, MLflow for experiment tracking, and multiple clustering algorithms for patient segmentation.

---

# Features

- Population Health Patient Segmentation
- Multiple Clustering Algorithms
- Automatic Model Comparison
- Cluster Profiling
- Patient Risk Stratification
- Healthcare Cost Analysis
- Population Health Analytics
- MLflow Experiment Tracking
- REST API using FastAPI
- Interactive Streamlit Dashboard
- Modular Enterprise Architecture
- Production Ready Design

---

# Business Problem

Healthcare organizations manage thousands of patients with different clinical conditions, healthcare utilization patterns, and treatment costs.

Traditional rule-based segmentation often fails to identify hidden patient populations.

Machine Learning clustering enables:

- Early identification of high-risk patients
- Personalized healthcare interventions
- Population health management
- Resource optimization
- Cost reduction
- Preventive care planning

---

# Project Architecture

```
                    Population Health Dataset
                               │
                               ▼
                     Data Preprocessing
                               │
                               ▼
                  Feature Engineering
                               │
                               ▼
              Multiple Clustering Algorithms
                               │
      ┌──────────────┬───────────────┬───────────────┐
      │              │               │               │
   KMeans     Agglomerative      DBSCAN        Spectral
      │              │               │               │
      └──────────────┴───────────────┴───────────────┘
                               │
                               ▼
                     Model Comparison
                               │
                               ▼
                      Best Model Selection
                               │
                               ▼
                     MLflow Registration
                               │
                               ▼
                     FastAPI Prediction API
                               │
                               ▼
                      Streamlit Dashboard
```

---

# Technology Stack

| Component | Technology |
|----------|------------|
| Language | Python 3.x |
| API | FastAPI |
| UI | Streamlit |
| Machine Learning | Scikit-Learn |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib |
| Validation | Pydantic |
| Logging | Loguru |
| Data Processing | Pandas, NumPy |





# Dataset

The dataset contains patient demographic, clinical, utilization, and financial information.

## Features

| Feature | Description |
|----------|-------------|
| patient_id | Patient Identifier |
| age | Patient Age |
| gender | Gender |
| bmi | Body Mass Index |
| diabetes | Diabetes Flag |
| hypertension | Hypertension Flag |
| copd | COPD Flag |
| mental_health | Mental Health Flag |
| visits_per_year | Annual Visits |
| er_visits | Emergency Visits |
| admissions | Hospital Admissions |
| total_cost | Total Healthcare Cost |

---

# Machine Learning Algorithms

The project supports multiple clustering algorithms.

- K-Means
- Agglomerative Clustering
- DBSCAN
- Spectral Clustering
- Birch
- Gaussian Mixture Model (GMM)
- HDBSCAN
- OPTICS
- Divisive Clustering
- Fuzzy C-Means

---

# Cluster Evaluation Metrics

The clustering models can be evaluated using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Number of Clusters
- Cluster Size Distribution

---

# Cluster Profiles

Example patient segments include:

### Cluster 0
Healthy Population

Characteristics

- Young patients
- Low BMI
- Low healthcare cost
- Minimal admissions
- Preventive care focus

---

### Cluster 1

Chronic Disease Patients

Characteristics

- Diabetes
- Hypertension
- Moderate utilization
- Regular follow-up

---

### Cluster 2

High Risk Patients

Characteristics

- Elderly
- Multiple chronic diseases
- Frequent ER visits
- High admissions
- High healthcare cost

---

### Cluster 3

Behavioral Health Population

Characteristics

- Mental health conditions
- Frequent outpatient visits
- Care coordination needs

---

# Training and Comparing Various Clustering Models.
# Run below commands for comparing
python -m ml.clustering_comparison

# Run below command for Best Model
python -m ml.clustering_comparison

# Best Model is used for Prediction

# FastAPI Endpoints

## Train Model

```
POST /api/v1/patient-segmentation/train
```

---

## Predict Patient Segment

```
POST /api/v1/patient-segmentation/predict
```

Example Request

```json
{
    "age":72,
    "gender":"F",
    "bmi":34.5,
    "diabetes":1,
    "hypertension":1,
    "copd":0,
    "mental_health":0,
    "visits_per_year":15,
    "er_visits":4,
    "admissions":2,
    "total_cost":45000
}
```

---

## Model Evaluation

```
GET /api/v1/patient-segmentation/evaluation
```

---

## Cluster Distribution

```
GET /api/v1/patient-segmentation/cluster-distribution
```

---

## Cluster Profile

```
GET /api/v1/patient-segmentation/cluster-profile
```

---

## Segment Summary

```
GET /api/v1/patient-segmentation/segment-summary
```

---

## Health Check

```
GET /health
```

---

# Running the Project

## Clone Repository

```bash
git clone https://github.com/your-repository/Population_Health_Segmentation.git
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Start MLflow

```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--host 0.0.0.0 \
--port 5000
```

---

## Start FastAPI

```bash
cd app

uvicorn main:app --reload
```

Swagger

```
http://localhost:8000/docs
```

---

## Start Streamlit

```bash
streamlit run ui/streamlit_app.py
```

---

# Expected Workflow

```
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train Multiple Clustering Models
      │
      ▼
Evaluate Models
      │
      ▼
Select Best Model
      │
      ▼
Register in MLflow
      │
      ▼
Prediction API
      │
      ▼
Population Health Dashboard
```

---

# Applications

- Population Health Management
- Preventive Care Programs
- Disease Management
- Healthcare Analytics
- Insurance Risk Segmentation
- Patient Risk Stratification
- Hospital Resource Planning
- Care Management
- Value-Based Healthcare
- Public Health Analytics

---

# Future Enhancements

- Automated Hyperparameter Optimization
- SHAP Explainability
- Real-Time Patient Segmentation
- Streaming Predictions
- Azure Deployment
- Docker Support
- Kubernetes Deployment
- CI/CD Integration
- Role-Based Authentication
- Monitoring Dashboard
- Population Trend Analysis

---

# Requirements

```
fastapi
uvicorn
pandas
numpy
streamlit
loguru
scikit-learn
xgboost
joblib
pydantic
mlflow
shap
skops
scikit-learn-extra
hdbscan
scikit-fuzzy
python-multipart
```

---

# Author

**Population Health Segmentation**

Enterprise Machine Learning solution for healthcare organizations that combines unsupervised learning, population health analytics, and modern MLOps practices to support intelligent patient segmentation and data-driven clinical decision making.