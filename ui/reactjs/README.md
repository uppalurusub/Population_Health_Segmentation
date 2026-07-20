# Population Health Segmentation - React TypeScript UI

## Overview

The **Population Health Segmentation React Application** is a modern web application built with **React**, **TypeScript**, **Material UI (MUI)**, and **Axios**. It provides an intuitive user interface for predicting patient population segments by communicating with a FastAPI backend.

The application allows healthcare professionals, analysts, and administrators to enter patient demographic and clinical information and receive an AI-powered patient segmentation prediction.

---

# Features

- Modern React 19 + TypeScript application
- Material UI responsive interface
- FastAPI REST API integration
- Patient Segmentation Prediction
- Input validation through typed models
- Loading indicators during prediction
- Clean JSON/API response handling
- Responsive layout
- Easy integration with ML backend

---

# Technology Stack

| Technology | Version |
|------------|----------|
| React | 19.x |
| TypeScript | Latest |
| Vite | Latest |
| Material UI | Latest |
| Axios | Latest |
| CSS3 | Yes |

---

# Project Structure

```
src/
│
├── components/
│   └── PatientSegmentationForm.tsx
│
├── services/
│   └── patientService.ts
│
├── types/
│   └── patient.ts
│
├── App.tsx
├── main.tsx
└── index.css
```

---

# Application Architecture

```
                User

                  │

                  ▼

     PatientSegmentationForm

                  │

                  ▼

         patientService.ts

                  │

                  ▼

      Axios HTTP POST Request

                  │

                  ▼

 FastAPI Backend API

                  │

                  ▼

 Machine Learning Model

                  │

                  ▼

 Prediction Response

                  │

                  ▼

 React UI Display
```

---

# Input Parameters

The application accepts the following patient information.

| Field | Type | Description |
|---------|------|-------------|
| admissions | Number | Total admissions |
| age | Number | Patient age |
| bmi | Number | Body Mass Index |
| copd | Number | COPD Indicator |
| diabetes | Number | Diabetes Indicator |
| er_visits | Number | Emergency Visits |
| gender | String | M/F |
| hypertension | Number | Hypertension Indicator |
| mental_health | Number | Mental Health Indicator |
| total_cost | Number | Healthcare Cost |
| visits_per_year | Number | Annual Visits |

---

# Sample Request

```json
{
  "admissions": 2,
  "age": 72,
  "bmi": 34.5,
  "copd": 0,
  "diabetes": 1,
  "er_visits": 4,
  "gender": "F",
  "hypertension": 1,
  "mental_health": 0,
  "total_cost": 45000,
  "visits_per_year": 15
}
```

---

# Sample Response

```json
{
  "status": "success",
  "data": {
    "cluster": 3,
    "segment_name": "High Risk Chronic Care",
    "confidence": 0.94
  }
}
```

---

# API Endpoint

```
POST

http://127.0.0.1:8000/api/v1/patient-segmentation/predict
```

---

# Installation

Clone the repository.

```bash
git clone <repository-url>
```

Move into the project.

```bash
cd population-health-segmentation-react-app
```

Install dependencies.

```bash
npm install
```

---

# Running the Application

Start the development server.

```bash
npm run dev
```

The application will be available at

```
http://localhost:5173
```

---

# Backend Requirement

Ensure the FastAPI backend is running.

Example:

```
http://127.0.0.1:8000
```

Current API configuration

```typescript
const api = axios.create({
    baseURL: "http://127.0.0.1:8000/api/v1"
});
```

---

# Components

## App.tsx

Root application component responsible for rendering the patient segmentation form.

---

## PatientSegmentationForm.tsx

Responsible for:

- Rendering patient input form
- Managing React state
- Sending prediction request
- Displaying loading indicator
- Showing prediction response

---

## patientService.ts

Encapsulates all REST API communication.

```typescript
POST /patient-segmentation/predict
```

---

## patient.ts

Contains TypeScript interfaces.

### Request Model

```typescript
PatientRequest
```

### Response Model

```typescript
PatientResponse
```

---

# Styling

Global styling is defined in

```
src/index.css
```

Features include

- Responsive layout
- Typography
- Utility classes
- Scrollbar customization
- Response cards
- Form container
- JSON viewer
- Loader styles

---

# Current Workflow

```
User enters patient data

        │

        ▼

React validates input

        │

        ▼

Axios sends POST request

        │

        ▼

FastAPI endpoint

        │

        ▼

Machine Learning Model

        │

        ▼

Patient Segment Prediction

        │

        ▼

Prediction displayed
```

---

# Future Enhancements

- Dashboard Home
- Sidebar Navigation
- Header Component
- Dark Mode
- Authentication
- JWT Security
- API Configuration via Environment Variables
- Patient History
- Prediction History
- Export to CSV
- Export to Excel
- Export PDF Report
- Charts using Recharts
- Cluster Visualization
- Confidence Gauge
- Risk Indicators
- Error Boundary
- Toast Notifications
- Unit Testing
- Integration Testing
- Docker Support
- Kubernetes Deployment
- Azure App Service Deployment
- CI/CD using GitHub Actions

---

# Best Practices

- Strong TypeScript typing
- Component-based architecture
- Service layer abstraction
- Separation of concerns
- Responsive Material UI design
- Reusable interfaces
- Clean code organization

---

# Dependencies

- React
- React DOM
- TypeScript
- Axios
- Material UI
- Emotion
- Vite

---

# Author

Population Health Segmentation Project

React + TypeScript Frontend

Healthcare Analytics and Machine Learning UI

---

# License

This project is intended for educational, research, and healthcare analytics purposes.
