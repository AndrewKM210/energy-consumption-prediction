# UE Energy Consumption Prediction: End-to-End Machine Learning Deployment Showcase

This project demonstrates the **full lifecycle of a machine learning system**, from data processing to public deployment.
The goal is **not** to build a production-ready predictor or a polished app, but to show **practical integration skills** across the modern ML engineering stack. The demo of the prediction is publically available [here](https://andrewkm-ecp.streamlit.app/). Visualization of the dataset and predictions are also shown in a Tableau story [here](https://public.tableau.com/app/profile/andrew.keon.mackay.parrott/viz/EnergyConsumptionUE/Story1).

---

## Overview

This repository contains all the components of an end-to-end machine learning project:

1. **Data Engineering** → preprocessing and feature extraction in **Databricks**.
2. **Model Training** → experiments using **Databricks Notebooks with MLflow**, **PyTorch neural networks** and **XGBoost**.
3. **Model Serving** → lightweight **FastAPI** application exposing `/predict` endpoints.
4. **Containerization** → reproducible build with **Docker**.
5. **Cloud Deployment** → scalable **Google Cloud Run** service.
6. **Frontend Integration** → interactive **Streamlit dashboard** consuming the public API.
7. **Visualization** → Tableau story

This pipeline demonstrates a full MLOps-style workflow: **data → model → API → cloud → user**.

---

## Architecture

```
Databricks (ETL + Training)
        ↓
Trained model artifact (.pkl)
        ↓
FastAPI service (Dockerized)
        ↓
Google Cloud Run deployment
        ↓
Streamlit dashboard (frontend)
```

All communication between components is handled through clean API interfaces and version-controlled artifacts.

---

## Technologies

| Layer               | Tools / Services                         |
| ------------------- | ---------------------------------------- |
| **Data & Training** | Databricks, PySpark, Pandas              |
| **ML**              | PyTorch, scikit-learn, XGBoost, MLflow   |
| **API Backend**     | FastAPI, Uvicorn                         |
| **Packaging**       | Docker                                   |
| **Deployment**      | Google Cloud Run                         |
| **Frontend**        | Streamlit Community Cloud                |
| **Authentication**  | Custom header secret (for demo security) |
| **Visualization**   | Tableau                                  |

---

## Data and Preprocessing

The raw data was obtained from [Kaggle](https://www.kaggle.com/datasets/elahesarlakian/europe-energy-consumption-by-users/data) and stored in ```estat_sdg_07_11_en.csv```. The source of the data is from Eurostat, and contains yearly metrics for each EU country related to energy consumption. The structure of the data stored in the file is the following:

| DATAFLOW             | LAST UPDATE       | freq | unit | geo | TIME_PERIOD | OBS_VALUE|
|----------------------|-------------------|------|------|-----|-------------|----------|
| ESTAT:SDG_07_11(1.0) | 18/04/24 23:00:00 | A    | I05  | AL  | 2000        | 80.6     |


* **DATAFLOW**: where the data comes from
* **LAST UPDATE**: date and time when the row was last updated
* **freq**: how often the data is updated
* **unit**: metric of energy consumption
    * MTOE: million tons of oil equivalent
    * TOE_HAB: tons of oil equivalent per capita
* **geo**: country (ISO2 2 letter codes)
* **TIME_PERIOD**: year which corresponds to the data
* **OBS_VALUE**: value of the measured metric

This data was loaded into a SQL Wharehouse table in Databricks. The [data_preprocessing README](data_preprocessing/README.md) contains detailed explanations as to how Databricks was used during this process. The [energy_preprocessing notebook](data_preprocessing/energy_preprocessing.html) shows the preprocessing stage of the table using PySpark and Pandas, which contains the following:
* Loading the table
* Visualizing the data
* Dropping columns that are not needed for ML
    * Several columns had only 1 value (DATAFLOW, freq)
    * LAST UPDATE is not valuable information
* Transform the table for ML use
    * Instead of unit and obs_value, create a column for each desired unit (TOE_HAB, MTOE)
* Standarizing column names
* Dealing with NULL values

The result is a ML ready table with the following structure:

| country | year | MTOE | TOE_HAB |
|---------|------|------|---------|
| SK      | 2019 | 11.2 | 2.05    |

---

## Machine Learning

Several ML algorithms where put to test with the data, with different feature engineering techniques applied to each individual case. The models in general get a country and year input, and predict the TOE_HAB. The train/test splits are temporal, with the test splits always being future years. This was done in order to measure the accuracy in forecasting several future years with no data. The tuning was done using cross-validation. MLflow was used to monitor the learning metrics during training. The [ML training notebook](ml_training/energy_ml_training.html) shows how all of this was done. An overview of the algorithms, feature engineering techniques and performance is the following:
* **Random Forests regressor** 
    * Algorithm of the scikit-learn package, two versions were tested
        * Use one-hot encoding for the countries, given that they are categorical values
        * Use numerical encoding of the countries as an integer input, surprisingly worked better
* **XGBoost regressor**
    * Algorithm of the xgboost library with numerical encoding for countries
    * Improved accuracy over Random Forests
* **Neural network**
    * Custom neural network designed with PyTorch
    * Use an embedding layer for learning country encodings
    * Best accuracy overall

The XGBoost and neural network models were then re-trained on all the data for deployment. The model artifacts where saved with MLflow and registered using Databricks. This allowed to access the models locally with MLflow using Databricks URIs and download them for the API and docker image.

---

## API

The ML models are integrated into a REST API created with FastAPI. Two different APIs where designed, one for the XGBoost model and another for the neural network model. All endpoints use the HTTP `GET` methods:

| Path            | Method | Description                            |
| --------------- | ------ | -------------------------------------- |
| `/health`       | GET    | Check service health                   |
| `/predict`      | GET    | Predict from country and year          |
| `/predict-year` | GET    | Predict all countries for a given year |

layer of security was added using an `Authorization` header containing a shared secret. The `ENERGY_API_SECRET` environment variable must be set to the desired secret. If the request was correct, the API will return a `200` code. If the shared secret is not included or incorrect, a `401`code will be returned. To run and test the XGBoost API locally:
```bash
pip install -r requirements_api.txt
ENERGY_API_SECRET=$(openssl rand -hex 32)
uvicorn api.main_xgb:app --reload --host 0.0.0.0 --port 8000

curl -X GET "http://127.0.0.1:8000/predict-year?year=2025" \
-H "Authorization: Bearer $ENERGY_API_SECRET"
```

---

## Dockerization

The FastAPI service is fully containerized. The [Dockerfile](Dockerfile) contains the definition for the XGBoost model. The secret variable is not included in the image for security reasons, but can be passed when running the image. The XGBoost model was chosen to reduce the docker image size (798MB vs 1.94GB, mainly due to the torch library, even if only CPU version). This makes it more cost efficient when deployed. To build and run locally:

```bash
docker build -t energy-api . # build image without secret
docker images # list images, check size
docker run -p 8080:8080 -e ENERGY_API_SECRET=$ENERGY_API_SECRET energy-api:latest # run image locally with secret
curl -X GET "http://127.0.0.1:8080/predict-year?year=2025" -H "Authorization: Bearer $ENERGY_API_SECRET"
```

---

## Deployment

The API is deployed using **Google Cloud Run**. First, the docker image has to be pushed to the **Artifact Registry**, the commands are shown below. Here `PROJECT` is the name of the Google Cloud project, `REPOSITORY` is the name of the Artifact Registry repository and `LOCATION` is the Google Cloud location (for example, us-west1).

```bash
gcloud auth login
gcloud config set project PROJECT
gcloud auth configure-docker LOCATION-docker.pkg.dev
docker tag energy-api:latest LOCATION-docker.pkg.dev/PROJECT/REPOSITORY/energy-api:latest
docker push LOCATION-docker.pkg.dev/PROJECT/REPOSITORY/energy-api:latest
```

The Google Cloud Run service was created by deploying the container from the Artifact Registry with the following relevant settings:
* **Authentication**
    * Requiring authentication with a **Service Account** was tested. 
    * However, it was impossible to configure publically with the Streamlit dashboard. 
    * Instead, the API is public but the `ENERGY_API_SECRET` controls the access. 
* **Secrets**: the secret is saved in the **Secret Manager** and exposed to the service
* **Billing**: request-based billing to only use the CPU when processing requests
* **Scaling**: auto-scaling with minimum 0 and maximum 1 instances
* **Requests**: maximum concurrent requests to 1 for costs saving

---

## Streamlit Dashboard

A minimal dashboard is hosted on **Streamlit Community Cloud**. It shows an interactive map with the predicted energy consumption for a given year of the different UE countries, with the predictions obtained from the Google Cloud API. The API URL and the authentication secret are stored locally in `.streamlit/secrets.toml`:

```toml
api_url = "..."
api_secret = "..."
```

These secret values can be configured in the public Streamlit demo without having to reveal them. The dashboard can run locally with:
```
pip install -r requirements.txt
streamlit run dashboard/main.py 
```

---

## Repository Structure

```
├── api/                        # FastAPI apps
│   ├── main_nn.py
│   ├── main_xgb.py
│
├── dashboard/                  # Streamlit dashboard and necessary initial data
│   ├── main_nn.py
│   ├── prediction_2025.csv
│   ├── prediction_ES.csv
│
├── data_preprocessing/         # Notebooks with how the data was processed
│   ├── energy_preprocessing.html
│   ├── energy_preprocessing.ipynb
|
├── Dockerfile                  # For building the neural network API container
|
├── estat_sdg_07_11_en.csv      # Raw dataset
|
├── ml_training/
│   ├── download_model.py       # Downloads all models and scalers/encoders from Databricks
│   ├── enegy_ml_training.html  # Notebook with how the ML algorithms where tested
│   ├── enegy_ml_training.ipynb # Notebook with how the ML algorithms where tested
│   ├── nn_model.py             # Definition of neural network PyTorch model
│   ├── requirements.txt        # Library dependancies for training the models
│   ├── train_nn.html           # Notebook with how the neural network was trained
│   ├── train_nn.ipynb          # Notebook with how the neural network was trained
|
├── requirements_api.txt        # Library dependancies for the API
|
├── requirements.txt            # Library dependancies for the dashboard
|
├── tableau/
│   ├── clean_data.ipynb        # Cleans the data of the estat_sdg_07_11_en.csv dataset
│   ├── toe_hab_pred.csv.csv    # Contains the predicted energy consumption for the next 25 years
│   ├── toe_hab.csv             # Contains the cleaned estat_sdg_07_11_en.csv dataset
│   ├── clean_data.ipynb        # Predicts the energy consumption over the next 25 years 
|
├── trained_models/             # Will contain trained models
│   ├── countries.pkl           # List of supported countries
│
└── README.md
```

---

## Future Improvements

* Improve models joining new data to current data (i.e GDP per capita)
* Add CI/CD pipeline (GitHub Actions + Cloud Run)
* Integrate monitoring (Prometheus / GCP logs)
* Try to deploy on AWS
* Extend dashboard with live dataset visualizations
* Tableau demo

---
