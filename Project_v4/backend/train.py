import argparse
import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient

import json

def parse_args():
    parser = argparse.ArgumentParser(description="H2O AutoML training and MLflow tracking")
    # 参数1
    parser.add_argument('--name',
                        '--n',
                        required=True,
                        default='automl-insurance',
                        help='案例名称',
                        type=str)
    # 参数2
    parser.add_argument('--target',
                        '--t',
                        required=True,
                        help='目标列',
                        type=str)
    # 参数3
    parser.add_argument('--models',
                        '--m',
                        required=True,
                        help='AutoML模型的数量，默认是10',
                        default=10,
                        type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    # 案例名称
    case_name = args.name
    # 初始化h2o cluster
    h2o.init()
    # 初始化mlflow client
    client = MlflowClient()
    # 创建 mlflow experiment
    try:
        experiment_id = mlflow.create_experiment(case_name)
        experiment = client.get_experiment_by_name(case_name)
    except:
        experiment = client.get_experiment_by_name(case_name)
    mlflow.set_experiment(case_name)

    print("Experiment details: ")
    print(f"Name: {case_name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifestype Stage: {experiment.lifecycle_stage}")
    print(f"Tracking uri: {mlflow.get_tracking_uri()}")

    # h2o 导入数据集
    df = h2o.import_file('data/train.csv')

    target = args.target
    features = [n for n in df.col_names if n!=target]

    df[target] = df[target].asfactor()

    with mlflow.start_run():
        pipeline = H2OAutoML(
            max_models=args.models, # 多少个base models
            seed=666,
            balance_classes=True, # 如果类不平衡，设置为True
            sort_metric='logloss', # 多分类的评估指标
            verbosity='info',
            exclude_algos=['GLM',"DRF"] # 排除的算法
        )
        # 初始化 AutoML training
        pipeline.train(x=features,
                       y=target,
                       training_frame=df)
        # 日志信息
        mlflow.log_metric('log_loss', pipeline.leader.logloss())
        mlflow.log_metric('AUC', pipeline.leader.auc())

        # 记录、保存模型
        mlflow.h2o.log_model(pipeline.leader, artifact_path='model')

        model_uri = mlflow.get_artifact_uri('model')
        print(f"AutoML best model saved in {model_uri}")

        exp_id = experiment.experiment_id
        run_id = mlflow.active_run().info.run_id

        # 保存 leaderboard
        board = get_leaderboard(pipeline, extra_columns='ALL')
        board_path = f'mlruns/{exp_id}/{run_id}/artifacts/model/leaderboard.csv'
        board.as_data_frame().to_csv(board_path, index=False)
        print(f"AutoML 完成，leaderboard保存的路径是：{board_path}")

if __name__ == "__main__":
    main()
