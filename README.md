# Breast Cancer MLOps Project

This repository implements an end-to-end MLOps pipeline for Breast Cancer classification. The project focuses on predicting whether a tumor is Malignant or Benign using clinical features, while incorporating industry-standard MLOps practices such as experiment tracking, data versioning, drift detection, and automated CI/CD.

## 🚀 Key Features

- **End-to-End Pipeline:** Automated data ingestion, preprocessing, and model training.
- **Experiment Tracking:** Integrated with **MLflow** to track parameters, metrics, and models.
- **Data & Model Versioning:** Powered by **DVC** (Data Version Control) with remote storage support.
- **Drift Detection:** Real-time data drift detection using **Alibi-detect (KSDrift)**.
- **Feature Store:** **Redis** is utilized as a feature store to maintain reference data for drift analysis.
- **Containerization:** Fully Dockerized environment for consistent deployment.
- **CI/CD:** **Jenkins** pipeline for automated testing and deployment.
- **Monitoring:** **Prometheus** integration for tracking prediction counts and drift occurrences.
- **Web Interface:** A **Flask**-based web application for real-time predictions.

## 🛠 Tech Stack

- **Languages:** Python
- **Machine Learning:** Scikit-learn, XGBoost, Pandas, NumPy
- **MLOps:** MLflow, DVC, Alibi-detect
- **Database/Store:** Redis (Feature Store)
- **DevOps:** Docker, Jenkins
- **Monitoring:** Prometheus
- **Web:** Flask, HTML/CSS

## 📁 Project Structure

```text
├── artifacts/          # DVC-tracked data and models
├── config/             # Configuration files (YAML and Python)
├── pipeline/           # Training pipeline orchestration
├── src/                # Core source code
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_store.py
│   ├── model_training.py
│   ├── logger.py
│   └── custom_exception.py
├── templates/          # Flask web templates
├── static/             # CSS and static assets
├── app.py              # Flask Application & Inference API
├── Dockerfile          # Application containerization
├── docker-compose.yaml # Multi-container orchestration (App + Redis + Prometheus)
├── Jenkinsfile         # CI/CD pipeline definition
└── requirements.txt    # Project dependencies
```

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd breast_cancer_mlops
```

### 2. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data & Model Versioning (DVC)
Initialize DVC and pull data (if remote is configured):
```bash
dvc pull
```

### 4. Running with Docker
The easiest way to run the entire stack (App, Redis, Prometheus) is using Docker Compose:
```bash
docker-compose up --build
```

## 📈 Usage

### Model Training
To trigger the training pipeline:
```bash
python pipeline/training_pipeline.py
```
This will process the data, train the model, log results to MLflow, and update the feature store in Redis.

### Web Application
Once the containers are running, access the web interface at:
- **Flask App:** `http://localhost:5000`
- **Prometheus Metrics:** `http://localhost:5000/metrics`

### Drift Detection
The application automatically compares incoming inference data against the reference data stored in Redis. If the statistical distribution shifts significantly, a "Data Drift Detected" warning will appear on the UI and be logged.

## 🛡 Monitoring
Prometheus scrapes the `/metrics` endpoint to monitor:
- `prediction_count`: Total requests processed.
- `drift_count`: Number of times drift was detected.
- `malignant_count` / `benign_count`: Prediction distribution.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
