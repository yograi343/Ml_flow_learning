import mlflow
from typing import Any


def create_ml_flow_experiment(
    experimnent_name: str, artifact_location: str, tags: dict[str, Any]
) -> str:
    """
    Create a new mlflow experiment with the given name and artifact location.
    """

    try:
        experiment_id = mlflow.create_experiment(
            name=experimnent_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"Experiment {experimnent_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experimnent_name).experiment_id
    return experiment_id


def get_mlflow_experiment(
    experiment_id: str = None, experiment_name: str = None
) -> str:
    """
    Retrieve the mlflow experiment with the given id or name

    Parameters:
    ----------
    experiment_id:str
        the id of the experiment to retrieve.
    experiment_name:str
        the name of the experiment to retrieve
    Returns:
    ----------
    experiment:mlflow.entitities.Experiment
        The mlflow experiment with the given id or name
    """
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided")
    return experiment
