import logging
import os
import pathlib
import pickle
import warnings

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

logger = DAGsHubLogger(
    metrics_path="../metrics/metrics.csv", hparams_path="../metrics/params.yml"
)

