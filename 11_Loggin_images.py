import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

if __name__ == "__main__":

    experiment = get_mlflow_experiment(experiment_name="testingmlflow1")

    print(f"Name: {experiment.name}")

    with mlflow.start_run(
        run_name="logging images", experiment_id=experiment.experiment_id
    ) as run:

        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=5,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, train_size=0.8, random_state=42
        )

        rfc = RandomForestClassifier(random_state=42)

        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)

        # log the precision-recall curve
        fig_pr = plt.figure()
        pr_display = PrecisionRecallDisplay.from_predictions(
            y_test, y_pred, ax=plt.gca()
        )
        plt.title("Precision-Recall Curve")
        plt.legend()

        mlflow.log_figure(fig_pr, "metrics/precision_recall_curve.png")

        fig_roc = plt.figure()
        roc_display = RocCurveDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
        plt.title("Roc Curve")
        plt.legend()

        mlflow.log_figure(fig_roc, "metrics/Roc_Curve.png")

        fig_cm = plt.figure()
        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=plt.gca()
        )
        plt.title("Confusion matrix")

        mlflow.log_figure(fig_cm, "metrics/confusion_matrix.png")
