import logging
import os
import pathlib
import pickle
import warnings
from pathlib import Path

import catboost as cb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import dagshub
from sklearn.compose import *
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.tree import *

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("https://dagshub.com/BexTuychiev/pet_pawpularity.mlflow")

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)

# Create the global dagshub logger
dh_logger = dagshub.dagshub_logger(metrics_path=Path("metrics/metrics.csv"),
                                   hparams_path=Path("metrics/params.yml"))

SEED = 1121218


def get_metadata(random_state=SEED):
    train_df = pd.read_csv("data/raw/train.csv").drop(["Id"], axis=1)
    train, test = train_test_split(train_df, random_state=random_state, test_size=0.1)

    x_train, y_train = train.drop("Pawpularity", axis=1), train[["Pawpularity"]]
    x_test, y_test = test.drop("Pawpularity", axis=1), test[["Pawpularity"]]

    return (x_train, y_train), (x_test, y_test)


def log_to_mlflow(param_dict, metrics_dict):
    """
    A simple function to log experiment results to MLFlow.
    """
    client = mlflow.tracking.MlflowClient()
    client.create_experiment('baseline')

    with mlflow.start_run():
        mlflow.log_params(param_dict)
        mlflow.log_metrics(metrics_dict)


def log_to_git(logger, param_dict, metrics_dict):
    """
    A simple function to log experiment results to Git.
    """

    with logger as logger:
        logger.log_hyperparams(param_dict)
        logger.log_metrics(metrics_dict)


def baseline_model():
    """
    A baseline model that predicts the mean of the target for metadata.
    """
    (x_train, y_train), (x_test, y_test) = get_metadata()

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(x_train, y_train)

    y_pred = baseline.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse = round(rmse, 3)

    log_to_git(dh_logger, baseline.get_params(), {"rmse": rmse, "model_name": "baseline"})
    logging.log(logging.INFO, f"Baseline model RMSE: {rmse}")

    return rmse


def train_xgb_model(random_state=SEED):
    """
    A function to train an XGBoost model.
    """
    (x_train, y_train), (x_test, y_test) = get_metadata(random_state=random_state)
    model = xgb.XGBRegressor(n_estimators=10000, random_state=random_state)

    cv_results = cross_validate(model, x_train, y_train, cv=5,
                                scoring="neg_mean_squared_error", return_estimator=True,
                                return_train_score=True, n_jobs=-1,
                                verbose=2)
    rmse_train = np.sqrt(-cv_results["train_score"].mean())
    rmse_val = np.sqrt(np.abs(cv_results["test_score"])).mean()
    best_model = cv_results["estimator"][np.argmin(rmse_val)]
    rmse_test = np.sqrt(mean_squared_error(y_test, best_model.predict(x_test)))

    logging.log(logging.INFO, f"XGBoost model RMSE train: {rmse_train}")
    logging.log(logging.INFO, f"XGBoost model RMSE validation: {rmse_val}")
    logging.log(logging.INFO, f"XGBoost model RMSE test: {rmse_test}")

    log_to_git(dh_logger, best_model.get_params(),
               {"rmse": rmse_test, "model_name": "XGBoost"})


if __name__ == "__main__":
    train_xgb_model()