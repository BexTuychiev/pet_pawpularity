import logging
import os
import warnings
import joblib

import bentoml
import catboost as cb
import lightgbm as lgb
import mlflow
import dagshub
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from preprocess import load_tf_datasets
import xgboost as xgb
import consts

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import *
from sklearn.model_selection import *

mlflow.keras.autolog()
mlflow.xgboost.autolog()
mlflow.lightgbm.autolog()

from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL


mlflow.set_tracking_uri("https://dagshub.com/BexTuychiev/pet_pawpularity.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = consts.MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = consts.MLFLOW_TRACKING_PASSWORD

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)
dh_logger = dagshub.dagshub_logger(metrics_path="metrics/metrics.csv",
                                   hparams_path="metrics/params.yml")
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

    with mlflow.start_run():
        mlflow.log_params(param_dict)
        mlflow.log_metrics(metrics_dict)


def log_to_git(logger, params, metrics):
    with logger as logger:
        logger.log_hyperparams(params)
        logger.log_metrics(metrics)


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


def get_xgb_model(random_state=SEED):
    """
    A function to create an XGB model.
    """
    model = xgb.XGBRegressor(n_estimators=10000,
                             max_depth=5,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             random_state=random_state, tree_method='gpu_hist')

    return model


def cv(model):
    """
    A function to perform cross-validation on a model.
    """
    (x_train, y_train), _ = get_metadata()

    cv_score = cross_validate(model, x_train, y_train, cv=5,
                              scoring="neg_mean_squared_error", return_estimator=True,
                              return_train_score=True, n_jobs=-1)

    return cv_score


def get_lgb_model(random_state=SEED):
    """
    A function to create an LGB model.
    """
    model = lgb.LGBMRegressor(n_estimators=10000, random_state=random_state, device="gpu",
                              subsample=0.8, colsample_bytree=0.8, max_depth=5)

    return model


def get_cb_model(random_state=SEED):
    """
    A function to create an CatBoost model.
    """
    model = cb.CatBoostRegressor(iterations=10000, random_state=random_state,
                                 task_type="GPU", verbose=False,
                                 subsample=0.8, colsample_bylevel=0.8, max_depth=5)

    return model


def train_simple(random_state=SEED):
    """
    A function to train simple models on the metadata.
    """
    (x_train, y_train), (x_test, y_test) = get_metadata(random_state=random_state)

    model = RandomForestRegressor(n_estimators=1500, random_state=random_state,
                                  max_depth=5, n_jobs=-1, min_samples_split=3,
                                  max_features="sqrt")

    with mlflow.start_run():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rmse_test = mean_squared_error(y_test, y_pred, squared=False)

        # Log the results to terminal
        logging.log(logging.INFO,
                    f"{model.__class__.__name__} model RMSE test: {rmse_test}")

    mlflow.end_run()


def get_keras_conv2d():
    """A function to build an instance of a Keras conv2d model."""
    inputs = keras.Input(shape=(224, 224, 3))

    X = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(
        inputs)
    X = MaxPool2D(2)(X)

    X = Conv2D(filters=34, kernel_size=3, padding='same', activation='relu')(X)
    X = MaxPool2D(3)(X)
    X = Dropout(0.25)(X)
    X = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(X)
    X = MaxPool2D(3)(X)
    X = Dropout(0.25)(X)
    X = Flatten()(X)

    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)

    outputs = Dense(1)(X)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_simple_keras():
    """
    A function to train simple Keras models on the metadata.
    """
    # load metadata
    (x_train, y_train), (x_test, y_test) = get_metadata()

    with mlflow.start_run():
        model = keras.Sequential()

        model.add(Dense(64, activation='relu', input_shape=x_train.shape))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')

        model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

        y_pred = model.predict(x_test)
        rmse_test = mean_squared_error(y_test, y_pred, squared=False)

        # Log the results to terminal
        logging.log(logging.INFO,
                    f"{model.__class__.__name__} model RMSE test: {rmse_test}")

    mlflow.end_run()


def fit_keras_conv2d():
    """
    A function to train a Keras conv2d model.
    """
    train_generator, validation_generator = load_tf_datasets()
    logging.log(logging.INFO, "Loaded the data generators")

    model = get_keras_conv2d()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                  patience=5)]

    logging.log(logging.INFO, "Started training...")
    model.fit(train_generator, validation_data=validation_generator, epochs=30,
              callbacks=callbacks, verbose=2)

    return model


def save(model, model_name, path):
    """
    A function to save a given model to BentoML local store and with joblib.
    """
    bentoml.keras.save(model_name, model, store_as_json_and_weights=True)

    joblib.dump(model, path)


def main():
    model = fit_keras_conv2d()

    logging.log(logging.INFO, "Saving...")

    save(model,
         "keras_conv2d_smaller",
         "models/keras_conv2d_smaller.joblib")

    logging.log(logging.INFO, "Done!")


if __name__ == "__main__":
    main()
