import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__ == "__main__":
    experiment = get_mlflow_experiment("114264747114248819", "testing_mlflow2")
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Creation timestamp: {experiment.creation_time}")
