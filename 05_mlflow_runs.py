import mlflow

if __name__ == "__main__":
    with mlflow.start_run(run_name="ml_flow_runs") as run:

        mlflow.log_param("Learning rate", 0.01)

        print("Run id")
        print(run.info.run_id)
        print(run.info)
