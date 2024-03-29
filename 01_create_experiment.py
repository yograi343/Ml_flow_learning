import mlflow
from mlflow_utils import create_ml_flow_experiment

if __name__ == "__main__":
    # create a new mlflow experiment
    # mlflow.create_experiment(
    #     name="testing_mlflow1",
    #     artifact_location="testing_mlflow1_artifacts",
    #     tags={"env": "dev", "version": "1.0.0"},
    # )
    experiment_id = create_ml_flow_experiment(
        experimnent_name="testing_mlflow2",
        artifact_location="testing_mlflow1_artifacts",
        tags={"env": "dev", "version": "1.0.0"},
    )
    print(f"Experiment id: {experiment_id}")
