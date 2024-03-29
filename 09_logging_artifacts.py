import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":
    experiment = get_mlflow_experiment(experiment_name="testingmlflow1")

    print(f"name: {experiment.name}")

    with mlflow.start_run(
        run_name="logging_artifacts", experiment_id=experiment.experiment_id
    ) as run:

        # your machine learning coode goes here
        metrics1 = {"mse": 0.01, "rmse": 0.01, "mae": 0.001, "r2": 0.01}
        mlflow.log_metrics(metrics1)
        mlflow.log_params({"learning_rate": 0.01})
        # create a txt file
        with open("hello_world.txt", "w") as f:
            f.write("Hello world")

        # log the text file as an artifact
        mlflow.log_artifact(
            local_path="hello_world.txt", artifact_path="hello_world.txt"
        )

        print(f"run_id: {run.info.run_id}")
        print(f"experiment_id: {run.info.experiment_id}")
        print(f"status: {run.info.status}")
        print(f"start_time: {run.info.start_time}")
        print(f"lifecycle_stage: {run.info.lifecycle_stage}")
        print(f"end_time: {run.info.end_time}")
