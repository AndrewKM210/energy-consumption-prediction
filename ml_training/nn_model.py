import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import mlflow.pyfunc
import numpy as np
import pandas as pd


class Net(nn.Module):
    def __init__(
        self,
        hidden_sizes=(32, 32),
        dropout=0.2,
        country_idx=0,
        year_idx=1,
        n_countries=None,
        embed_dim=4,
    ):
        super(Net, self).__init__()

        # Columns where year and country are
        self.year_idx, self.country_idx = year_idx, country_idx

        # Embeddings for the countries
        self.country_embed = nn.Embedding(n_countries, embed_dim)

        # Layer setup
        hidden_sizes = [embed_dim + 1] + list(hidden_sizes)
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat(
            [
                x[:, self.year_idx].reshape(-1, 1),
                self.country_embed(x[:, self.country_idx].long()),
            ],
            dim=1,
        )
        return self.fc_layers(x)


class NNModel(mlflow.pyfunc.PythonModel):
    def __init__(
        self, hidden_sizes=(32, 32), lr=0.01, dropout=0.2, n_countries=None, embed_dim=4
    ):

        # Custom neural network
        self.net = Net(
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            n_countries=n_countries,
            embed_dim=embed_dim,
        )

        # Scalers for standarizing the input and output values
        self.year_scaler, self.output_scaler = StandardScaler(), StandardScaler()
        self.country_idx, self.year_idx = 0, 1

        # Optimizer and loss function for training
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def _prepare_training_data(self, X, y):
        """Prepares the training data for the neural network."""
        # Prepare data
        assert (
            type(X) is pd.DataFrame and type(y) is pd.DataFrame
        ), "Train data must be DataFrames"
        assert (
            "country_encoded" in X.columns
        ), "X DataFrame must contain country_encoded column"
        assert "year" in X.columns, "X DataFrame must contain year column"
        self.year_scaler.fit(X["year"].values.reshape(-1, 1))
        self.output_scaler.fit(y.values)
        X["year"] = self.year_scaler.transform(X["year"].values.reshape(-1, 1))
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(self.output_scaler.transform(y.values), dtype=torch.float32)
        return X, y

    def fit(self, X, y, epochs=100, test_frac=0, mlflow_run=False, verbose=False):
        """
        Trains the neural network.
        X: [country, year], y=[TOE_HAB/MTOE]
        """
        X, y = self._prepare_training_data(X.copy(), y.copy())

        if test_frac > 0:
            n_test = int(test_frac * len(X))
            X_train, X_test = X[:-n_test], X[-n_test:]
            y_train, y_test = y[:-n_test], y[-n_test:]
        else:
            X_train, y_train = X, y

        # Train loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_hat = self.net(X_train)
            loss = self.mse(y_hat, y_train)
            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()

            if test_frac > 0:
                y_hat = self.net(X_test)
                val_loss = self.mse(y_hat, y_test).item()
                if mlflow_run:
                    mlflow.log_metrics(
                        {"train_mse": train_loss, "test_mse": val_loss}, step=epoch + 1
                    )
                if (epoch + 1) % 10 == 0 and verbose:
                    print(
                        f"Epoch {epoch+1}, Train loss: {train_loss:.4f}, Test loss: {val_loss:.4f}"
                    )
            else:
                if mlflow_run:
                    mlflow.log_metric("train_mse", train_loss, step=epoch + 1)
                if (epoch + 1) % 10 == 0 and verbose:
                    print(f"Epoch {epoch+1}, Train loss: {train_loss:.4f}")

        self.net.eval()

    def predict(self, model_input: pd.DataFrame) -> pd.Series:
        model_input = model_input.copy()
        model_input["year"] = self.year_scaler.transform(
            model_input["year"].values.reshape(-1, 1)
        )
        model_input = torch.tensor(model_input.values, dtype=torch.float32)
        with torch.no_grad():
            y_hat = self.net(model_input)
        y_hat = self.output_scaler.inverse_transform(y_hat)
        return pd.Series(y_hat.reshape(-1))

    def evaluate_rmse(self, X, y):
        assert (
            type(X) is pd.DataFrame and type(y) is pd.DataFrame
        ), "Data must be DataFrames"
        assert (
            "country_encoded" in X.columns
        ), "X DataFrame must contain country_encoded column"
        assert "year" in X.columns, "X DataFrame must contain year column"
        with torch.no_grad():
            y_pred = self.predict(X.copy())
        return np.sqrt(np.mean((y_pred.values - y.values.reshape(-1)) ** 2))