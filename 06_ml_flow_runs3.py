import mlflow
from mlflow_utils import create_ml_flow_experiment

if __name__ == "__main__":
    # lets create an experiment
    experiment_id = create_ml_flow_experiment(
        experimnent_name="testingmlflow1",
        artifact_location="testing_mlflow1_artifacts",
        tags={"env": "dev", "version": "1.0.0"},
    )
    # mlflow.set_experiment(experiment_name="testingmlflow1")
    # lets start a run
    with mlflow.start_run(run_name="testing", experiment_id=experiment_id) as run:
        mlflow.log_param("learning_rate", 0.01)

        print(f"Run_id: {run.info.run_id}")
        print(f"experiment_id: {run.info.experiment_id}")
        print(f"status: {run.info.status}")
        print(f"start_time: {run.info.start_time}")
        print(f"end_time {run.info.end_time}")
        print(f"lifecycle_stage: {run.info.lifecycle_stage}")
