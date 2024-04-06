import mlflow
from mlflow_utils import create_ml_flow_experiment

experiment_id = create_ml_flow_experiment(
    experimnent_name="Nested Runs",
    artifact_location="nested_run_artifacts",
    tags={"purpose": "learning"},
)

with mlflow.start_run(
    run_name="parent", experiment_id=experiment_id, nested=True
) as parent:
    print(f"Run ID parent: {parent.info.run_id}")

    mlflow.log_param("parent_param", "parent_value")

    with mlflow.start_run(run_name="child1", nested=True) as child1:
        print(f"Child1 Run ID: {child1.info.run_id}")

        mlflow.log_param("child1_param", "child1_value")

        with mlflow.start_run(run_name="child_l1", nested=True) as child_l1:
            print(f"Child_l1 Run ID: {child_l1.info.run_id}")
            mlflow.log_param("childl1_param", "childl1_value")
        with mlflow.start_run(run_name="child_r1", nested=True) as child_r1:
            print(f"Child_l1 Run ID: {child_r1.info.run_id}")
            mlflow.log_param("childl1_param", "childl1_value")
    with mlflow.start_run(run_name="child2", nested=True) as child2:
        print(f"Child1 Run ID: {child2.info.run_id}")
        mlflow.log_param("child2_param", "child2_value")
