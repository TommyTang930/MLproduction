import pandas as pd
import io
import h2o

from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from util.preprocessing import separate_id_col, match_col_types

app = FastAPI()

h2o.init()
client = MlflowClient()

# 基于logloss，从所有实验结果中加载最好的模型
all_experiments = [exp.experiment_id for exp in client.search_experiments()]
runs = mlflow.search_runs(experiment_ids=all_experiments, run_view_type=ViewType.ALL)
print("runs: \n", runs)
run_id = runs.loc[runs['metrics.log_loss'].idxmin()]['run_id']
exp_id = runs.loc[runs['metrics.log_loss'].idxmin()]['experiment_id']
print(f"最优模型的 run_id:{run_id}, experiment_id:{exp_id}")

# 最优模型
best_model = mlflow.h2o.load_model(f"mlruns/{exp_id}/{run_id}/artifacts/model/")
#print("best_model : \n", best_model)

@app.post("/predict")
async def predict(file: bytes = File(...)):
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    test_h2o = h2o.H2OFrame(test_df)

    # 分类ID列
    id_name, X_id, X_h2o = separate_id_col(test_h2o)

    # 保持test数据集与train数据集列名一致
    X_h2o = match_col_types(X_h2o)

    # 模型预测
    preds = best_model.predict(X_h2o)

    if id_name is not None:
        # 如果ID列存在
        preds_list = preds.as_data_frame()['predict'].tolist()
        id_list = X_id.as_data_frame()[id_name].tolist()
        preds_final = dict(zip(id_list, preds_list))
    else:
        # 不存在ID列
        preds_final = preds.as_data_frame()['predict'].tolist()

    # 转换为JSON格式
    preds_json = jsonable_encoder(preds_final)

    # 返回
    return JSONResponse(content=preds_json)


@app.get("/")
async def main():
    content = """
        <body>
            <h2>端到端AutoML部署案例</h2>
            <p>H2O模型和FastAPI实例创建成功</p>
            <p>浏览FastAPI UI : localhost:8000</p>
            <p>初始化Streamlit UI（frontend/app.py)提交预测请求</p>
        </body>
    """
    return HTMLResponse(content=content)