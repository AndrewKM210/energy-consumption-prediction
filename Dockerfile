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
COPY api/main_xgb.py ./api/main_xgb.py
COPY trained_models/xgb_model.pkl ./trained_models/xgb_model.pkl
COPY trained_models/countries.pkl ./trained_models/countries.pkl
COPY trained_models/country_encoder.pkl ./trained_models/country_encoder.pkl

EXPOSE 8080
CMD ["uvicorn", "api.main_xgb:app", "--host", "0.0.0.0", "--port", "8080"]
