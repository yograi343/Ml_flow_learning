import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

if __name__ == "__main__":

    run_id = "440544e89c9e46efae9be5f48330f8c2"

    x, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42
    )
    x = pd.DataFrame(x, columns=[f"feature {i}" for i in range(10)])
    y = pd.DataFrame(y, columns=["target"])
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.2, train_size=0.8, random_state=42
    )

    # load model
    model_uri = f"runs:/{run_id}/random_forest_classifier"
    rfc = mlflow.sklearn.load_model(model_uri=model_uri)

    y_pred = rfc.predict(x_test)
    y_pred = pd.DataFrame(y_pred, columns=["prediction"])

    print(y_pred.head())
