FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# dependancies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /api

# copy requirements_api file and pip install
COPY requirements_api.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# copy app, model and model definition
COPY api/main_nn.py ./api/main_nn.py
COPY trained_models/neural_network.pkl ./trained_models/neural_network.pkl
COPY trained_models/countries.pkl ./trained_models/countries.pkl
COPY trained_models/country_encoder.pkl ./trained_models/country_encoder.pkl
COPY ml_training/nn_model.py ./ml_training/nn_model.py
COPY ml_training/__init__.py ./ml_training/__init__.py

EXPOSE 8080
CMD ["uvicorn", "api.main_nn:app", "--host", "0.0.0.0", "--port", "8080"]
