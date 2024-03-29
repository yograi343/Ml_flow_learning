import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":

    experiment = get_mlflow_experiment(experiment_name="testingmlflow1")

    print(f"Experiment Name: {experiment.name}")

    with mlflow.start_run(
        run_name="loggin_metrics", experiment_id=experiment.experiment_id
    ) as run:

        metrics1 = {"mse": 0.01, "rmse": 0.01, "mae": 0.001, "r2": 0.01}
        mlflow.log_metrics(metrics1)
        mlflow.log_params({"learning_rate": 0.01})

        print(f"run_id: {run.info.run_id}")
        print(f"experiment_id: {run.info.experiment_id}")
        print(f"status: {run.info.status}")
        print(f"start_time: {run.info.start_time}")
        print(f"lifecycle_stage: {run.info.lifecycle_stage}")
        print(f"end_time: {run.info.end_time}")
