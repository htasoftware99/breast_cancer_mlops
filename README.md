# 🎗️ Breast Cancer MLOps Project

An end-to-end MLOps pipeline for Breast Cancer classification using the Wisconsin Diagnostic Dataset. The project predicts whether a tumor is **Malignant** or **Benign** using 23 clinical features, while incorporating industry-standard MLOps practices including experiment tracking, data versioning, feature stores, drift detection, monitoring, and automated CI/CD.

---

## 🚀 Key Features

| Feature | Tool |
|---|---|
| Data ingestion from cloud | Google Cloud Storage (GCP) |
| Experiment tracking | MLflow + DagsHub |
| Data & model versioning | DVC + GCS remote |
| Feature store | Redis (Docker) |
| Drift detection | Alibi-detect (KSDrift) |
| Web interface | Flask |
| Containerization | Docker |
| CI/CD | Jenkins (DinD) |
| Monitoring | Prometheus + Grafana |

---

## 🛠 Tech Stack

- **Language:** Python 3.10
- **ML:** Scikit-learn, Logistic Regression, GridSearchCV
- **MLOps:** MLflow, DVC, Alibi-detect
- **Feature Store:** Redis
- **DevOps:** Docker, Jenkins
- **Monitoring:** Prometheus, Grafana
- **Cloud:** Google Cloud Storage, Google Cloud Run, GCR
- **Web:** Flask, HTML/CSS

---

## 📁 Project Structure

```text
breast_cancer_mlops/
├── artifacts/
│   ├── raw/                        # Raw data (train.csv, test.csv)
│   ├── processed/                  # Scaled data + scaler.pkl
│   └── models/                     # Trained model (.pkl)
├── config/
│   ├── config.yaml                 # Pipeline configuration
│   ├── model_params.py             # GridSearchCV hyperparameters
│   └── paths_config.py             # Centralized path definitions
├── custom_jenkins/
│   └── Dockerfile                  # Custom Jenkins image with Docker-in-Docker
├── pipeline/
│   └── training_pipeline.py        # End-to-end pipeline orchestrator
├── src/
│   ├── data_ingestion.py           # GCP bucket → raw CSV
│   ├── data_preprocessing.py       # Cleaning, correlation filter, scaling, Redis write
│   ├── feature_store.py            # Redis feature store client
│   ├── model_training.py           # GridSearchCV + MLflow logging
│   ├── logger.py                   # Centralized logging
│   └── custom_exception.py         # Structured exception handling
├── templates/
│   └── index.html                  # Flask prediction UI
├── utils/
│   └── common_functions.py         # YAML reader, data loader
├── app.py                          # Flask app + KSDrift + Prometheus
├── Dockerfile                      # Application image (python:3.10-slim)
├── Jenkinsfile                     # CI/CD pipeline definition
└── dvc.yaml                        # DVC pipeline stages
```

---

## ⚙️ Architecture & Pipeline

```
GCP Bucket (breast_cancer.csv)
        ↓
  data_ingestion.py          → artifacts/raw/
        ↓
  data_preprocessing.py      → artifacts/processed/  +  Redis Feature Store
        ↓
  model_training.py          → artifacts/models/     +  MLflow (DagsHub)
        ↓
  app.py (Flask)             → Predictions + KSDrift + Prometheus metrics
        ↓
  Jenkins CI/CD              → Docker image → GCR → Cloud Run
```

---

## 🔧 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/htasoftware99/breast_cancer_mlops.git
cd breast_cancer_mlops
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 3. GCP Credentials

Set your Google Cloud service account credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

---

## 🗄️ Redis Feature Store Setup

Redis is used as a feature store to hold unscaled training data, which is later used as the reference distribution for KSDrift-based drift detection.

### Start Redis via Docker

```bash
docker pull redis
docker run -d --name redis-container -p 6379:6379 redis
```

### Verify Connection

```bash
docker exec -it redis-container redis-cli ping
# Expected output: PONG
```

### How It Works

During preprocessing, unscaled feature data is written to Redis before the StandardScaler is applied. This ensures the drift detector in `app.py` can fit its own scaler on raw reference data — matching the reference distribution to incoming raw inference inputs.

```python
# data_preprocessing.py — Redis write happens BEFORE scaling
if self.feature_store:
    self.store_features_in_redis(train_df, test_df)  # raw values

train_df, test_df = self.scale_data(train_df, test_df)  # scale after
```

---

## 📊 MLflow Experiment Tracking

MLflow is used to log hyperparameters, metrics, and the trained model for every training run.

### Local Tracking (UI only)

```bash
python pipeline/training_pipeline.py
mlflow ui
# Open: http://127.0.0.1:5000
```

### DagsHub Remote Tracking

The project is integrated with [DagsHub](https://dagshub.com) for remote experiment tracking. Add the following to your `.env` file:

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/breast_cancer_mlops.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub_username>
MLFLOW_TRACKING_PASSWORD=<dagshub_token>
```

Then set the tracking URI in `model_training.py`:

```python
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
```

Each run logs:
- **Parameters:** `C`, `solver`, `max_iter`, `cv_folds`, `scoring`
- **Metrics:** `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- **Artifacts:** `logistic_regression_model.pkl`, sklearn model flavor
- **Registry:** Model registered as `BreastCancerLR`

---

## 🔄 DVC Data Versioning

DVC tracks data and model artifacts with remote storage on Google Cloud Storage.

```bash
# Initialize DVC (already done)
dvc init

# Add GCS remote
dvc remote add -d myremote gs://<your-bucket-name>/dvc-store

# Push artifacts to remote
dvc push

# Pull artifacts from remote
dvc pull
```

Pipeline stages are defined in `dvc.yaml`:

```yaml
stages:
  data_ingestion:   → artifacts/raw/
  data_preprocessing: → artifacts/processed/
  model_training:   → artifacts/models/
```

---

## 🌊 Drift Detection

Data drift is detected using **KSDrift** from the [Alibi-detect](https://github.com/SeldonIO/alibi-detect) library.

### How It Works

1. On application startup, `app.py` pulls all training records from Redis
2. A dedicated `drift_scaler` (StandardScaler) is fit on this raw reference data
3. The same scaler transforms both the reference data and each incoming inference input
4. KSDrift runs a Kolmogorov-Smirnov test comparing the two distributions
5. If `p_val < 0.05`, drift is flagged and a warning is shown in the UI

```python
# Two separate scalers are used:
scaler        → model prediction (fit during preprocessing)
drift_scaler  → drift detection only (fit on raw Redis data at startup)
```

If Redis is unavailable, drift detection is silently disabled and the app continues normally.

---

## 📈 Monitoring with Prometheus & Grafana

### Prometheus

The Flask app exposes a `/metrics` endpoint that Prometheus scrapes.

Tracked counters:

| Metric | Description |
|---|---|
| `prediction_count` | Total number of predictions made |
| `drift_count` | Number of drift detection events |
| `malignant_count` | Total malignant predictions |
| `benign_count` | Total benign predictions |

Access metrics at: `http://localhost:5000/metrics`

### Grafana

Connect Grafana to Prometheus as a data source and create dashboards to visualize prediction volume, drift frequency, and class distribution over time.

---

## 🖥️ Jenkins CI/CD Pipeline

### Jenkins Setup (Docker-in-Docker)

A custom Jenkins image was built to support Docker commands inside the pipeline:

```bash
cd custom_jenkins
docker build -t jenkins-dind .
docker run -d --name jenkins-dind \
  --privileged \
  -p 8080:8080 -p 50000:50000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v jenkins_home:/var/jenkins_home \
  jenkins-dind
```

After starting, retrieve the initial admin password:

```bash
docker logs jenkins-dind
```

### Manual Tools Installed Inside Jenkins Container

```bash
docker exec -u root -it jenkins-dind bash

# Python
apt update && apt install -y python3 python3-pip python3-venv
ln -s /usr/bin/python3 /usr/bin/python

# Google Cloud SDK
apt-get install -y google-cloud-sdk

# Fix Docker socket permissions
chmod 666 /var/run/docker.sock
```

### Pipeline Stages

```
1. Clone GitHub repo
        ↓
2. Docker build → tag as gcr.io/<project>/ml-project:latest
        ↓
3. Push to Google Container Registry (GCR)
        ↓
4. Deploy to Google Cloud Run
```

The `pip install -e .` step runs **inside Docker** (python:3.10-slim), not on Jenkins itself — this avoids Python version conflicts since Jenkins runs Python 3.13 which is incompatible with `alibi-detect` (requires < 3.13).

### Required Jenkins Credentials

| Credential ID | Type | Description |
|---|---|---|
| `github-token-bc` | Username/Password | GitHub access token |
| `bc-gcp-key` | Secret File | GCP service account JSON key |

---

## 🏃 Running the Project

### Full Training Pipeline

```bash
# 1. Start Redis
docker start redis-container

# 2. Run pipeline (ingestion → preprocessing → training)
python pipeline/training_pipeline.py

# 3. Start Flask app
python app.py
# Open: http://localhost:5000
```

### MLflow UI (local)

```bash
mlflow ui
# Open: http://127.0.0.1:5000
```

### Docker Build (local test)

```bash
docker build -t breast-cancer-app .
docker run -p 5000:5000 breast-cancer-app
```

---

## 🧪 Testing Drift Detection

**Normal input (no drift expected):** Enter realistic clinical values within the training data range.

**Anomalous input (drift expected):** Enter `9999` for all features. The UI will display a yellow "Data Drift Detected" warning banner.

---

## 📋 Configuration

All key configuration is centralized:

| File | Purpose |
|---|---|
| `config/config.yaml` | Bucket name, train ratio, correlation threshold |
| `config/model_params.py` | Logistic Regression hyperparameter grid |
| `config/paths_config.py` | All artifact and config file paths |

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.