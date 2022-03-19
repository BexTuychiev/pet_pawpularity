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
from dagshub import DAGsHubLogger
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

SEED = 1121218


def get_metadata(random_state=SEED):
    train_df = pd.read_csv("../data/raw/train.csv").drop(["Id"], axis=1)
    train, test = train_test_split(train_df, random_state=random_state, test_size=0.1)

    x_train, y_train = train.drop("Pawpularity", axis=1), train[["Pawpularity"]]
    x_test, y_test = test.drop("Pawpularity", axis=1), test[["Pawpularity"]]

    return (x_train, y_train), (x_test, y_test)


def log_to_mlflow(param_dict, metrics_dict):
    """
    A simple function to log experiment results to MLFlow.
    """
    with mlflow.start_run():
        mlflow.log_params(param_dict)
        mlflow.log_metrics(metrics_dict)


def log_to_git(param_dict, metrics_dict, path="."):
    """
    A simple function to log experiment results to Git.
    """
    logger = DAGsHubLogger(metrics_path=f"{path}/metrics.csv",
                           hparams_path=f"{path}/params.yml")

    logger.log_hyperparameters(param_dict)
    logger.log_metrics(metrics_dict)
