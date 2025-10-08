import mlflow
import os
import joblib
from ml_training.nn_model import NNModel
import torch
import pandas as pd
import json


def download_artifact(run_id, name, dst_path):
    """
        Downloads artifact if it is not downloaded already.
        run_id: ID of Databricks run
        name: name of artifact file
        dir: destination directory
    """
    if not os.path.exists(dst_path + name):
        print("Downloading artifact", name)
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=name,
            dst_path=dst_path
        )    

def unpickle_artifact(run_id, name, dir):
    """
        Downloads artifact if it is not downloaded already and unpickles it.
        run_id: ID of Databricks run
        name: name of artifact file
        dir: destination directory
    """
    download_artifact(run_id, name, dir)
    return joblib.load(dir + name)

# Setup Databricks environment
with open('ml_training/databricks_data.json') as f:
    databricks_data = json.load(f)
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(f"/Users/{databricks_data['user']}/{databricks_data['experiment']}")
nn_run_id = databricks_data["nn_run_id"]
xgb_run_id = databricks_data["xgb_run_id"]

# Load prediction models and encoders/scalers (download from Databricks if needed)
print("Loading models and encoders/scalers")
model_dir = "trained_models/"
xgb_model = mlflow.xgboost.load_model(f"runs:/{xgb_run_id}/xgboost_regressor")
joblib.dump(xgb_model, model_dir + "xgb_model.pkl")
download_artifact(nn_run_id, "neural_network.pt", model_dir)
download_artifact(nn_run_id, "country_encoder.pkl", model_dir)
params = unpickle_artifact(nn_run_id, "parameters.pkl", model_dir)
model = NNModel(hidden_sizes=params["hidden_sizes"], lr=params["lr"], dropout=params["dropout"], n_countries=params["n_countries"], embed_dim=params["embed_dim"])
model.net.load_state_dict(torch.load(model_dir + "neural_network.pt"))
output_scaler = unpickle_artifact(nn_run_id, "output_scaler.pkl", model_dir)
year_scaler = unpickle_artifact(nn_run_id, "year_scaler.pkl", model_dir)
model.year_scaler = year_scaler
model.output_scaler = output_scaler

# Save reconstructed NN model
dst_path = model_dir + "neural_network.pkl"
print("Saving reconstructed neural network to", dst_path)
joblib.dump(model, dst_path)

# Load and save list of countries
dst_path = model_dir + "countries.pkl"
print("Loading and saving list of countries to", dst_path)
df = pd.read_csv("estat_sdg_07_11_en.csv")
df = df[df["geo"] != "EU27_2020"]
joblib.dump(list(df["geo"].unique()), dst_path)