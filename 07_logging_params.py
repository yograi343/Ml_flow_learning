import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":

    experiment = get_mlflow_experiment(experiment_name="testingmlflow1")
    print(f"name: {experiment.name}")

    with mlflow.start_run(
        run_name="logging_params", experiment_id=experiment.experiment_id
    ) as run:
        parameters = {
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 100,
            "loss_function": "mse",
            "optimizer": "adam",
        }
        mlflow.log_params(parameters)

        print(f"run_id: {run.info.run_id}")
        print(f"experiment_id: {run.info.experiment_id}")
        print(f"status: {run.info.status}")
        print(f"start_time: {run.info.start_time}")
        print(f"lifecycle_stage: {run.info.lifecycle_stage}")
        print(f"end_time: {run.info.end_time}")
