import os
import warnings
import sys

import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from urllib.parse import urlparse

import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings(action="ignore")
    np.random.seed(666)
    try:
        df = pd.read_csv("data/winequality-red.csv", sep=";")
    except Exception as e:
        logger.exception("Reading CSV file caused error %s", e)
    X = df.drop(columns=['quality'], axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    with mlflow.start_run():
        lr = LogisticRegression()
        # 模型训练
        lr.fit(X_train, y_train)
        # 模型预测
        predict_results = lr.predict(X_test)
        # 模型评估
        (rmse, mae, r2) = eval_metrics(y_test, predict_results)
        # 日志记录
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print("tracking_url_type_store : ", tracking_url_type_store)

        if tracking_url_type_store != "file":
            # register the model
            mlflow.sklearn.log_model(lr, "model", registered_model_name="LogisticRegression")
        else:
            mlflow.sklearn.log_model(lr, "model")
