# ridge_regression.py

import numpy as np
from typing import List
from custom_types import ResultEntry


from models import Ridge_Regression
from custom_types import (
    XTrain, YTrain,
    XValidation, YValidation,
    XTest, YTest
)


def run_ridge_regression(
    X_train_raw: XTrain, y_train_raw: YTrain,
    X_validation_raw: XValidation, y_validation_raw: YValidation,
    X_test_raw: XTest, y_test_raw: YTest,
) -> None:
    """
    Run Part 3 – Ridge Regression end-to-end.
    All reshaping and accuracy computations are performed inline.
    This function only prints results and does not return anything.
    """

    print("\nRunning Ridge Regression...\n")

    lambda_values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]



    # ---------------------------------------------------------
    # Inline reshape into ridge-regression format:
    #   X_raw: (N, d) → X: (d, N)
    #   y_raw: (N,)   → y: (1, N)
    # ---------------------------------------------------------
    X_train = X_train_raw.T
    # y_train = y_train_raw.reshape(1, -1)
    y_train = y_train_raw
    X_validation = X_validation_raw.T
    y_validation = y_validation_raw

    X_test = X_test_raw.T
    y_test = y_test_raw
    results: List[ResultEntry] = []
    # -------------------------------------------------------------
    # Loop over each λ value
    # -------------------------------------------------------------
    for lambda_value in lambda_values:
        print(f"Training Ridge Regression with λ = {lambda_value}...")
        # ---------------------------------------------------------
        # Train model with the closed-form ridge-regression solution
        # ---------------------------------------------------------
        model = Ridge_Regression(lambda_value)
        model.fit(X_train, y_train)

        # ---------------------------------------------------------
        # Predictions
        # ---------------------------------------------------------
        yhat_train = model.predict(X_train)
        yhat_validation = model.predict(X_validation)
        yhat_test = model.predict(X_test)

        # ---------------------------------------------------------
        # Inline accuracy computation
        # ---------------------------------------------------------
        training_accuracy = float(np.mean(yhat_train == y_train))
        validation_accuracy = float(np.mean(yhat_validation == y_validation))
        test_accuracy = float(np.mean(yhat_test == y_test))



        result_entry: ResultEntry = {
            "lambda": lambda_value,
            "train_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy
        }

        results.append(result_entry)

    make_table(results)
    print_best_and_worst_lambdas(results)
    plot_best_and_worst_decision_boundaries(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )


def make_table(results: List[ResultEntry]) -> None:
    """
    Prints a formatted table of ridge regression accuracies.

    Parameters:
    results: A list of dictionaries. Each dictionary contains:
        "lambda": float,
        "train_accuracy": float,
        "validation_accuracy": float,
        "test_accuracy": float
    """

    header = (
        f"{'lambda':>10} | "
        f"{'train_accuracy':>16} | "
        f"{'validation_accuracy':>20} | "
        f"{'test_accuracy':>14}"
    )

    separator = "-" * len(header)

    print(header)
    print(separator)

    for entry in results:
        print(
            f"{entry['lambda']:>10.4f} | "
            f"{entry['train_accuracy']:>16.6f} | "
            f"{entry['validation_accuracy']:>20.6f} | "
            f"{entry['test_accuracy']:>14.6f}"
        )


from typing import List, Dict

ResultEntry = Dict[str, float]


def print_best_and_worst_lambdas(results: List[ResultEntry]) -> None:
    """
    Prints the best and worst regularization parameter values according to the
    validation accuracy.

    Parameters:
        results: A list of dictionaries. Each dictionary contains:
            "lambda": float,
            "train_accuracy": float,
            "validation_accuracy": float,
            "test_accuracy": float
    """

    # Find the entry with the highest validation accuracy
    best_entry: ResultEntry = max(
        results,
        key=lambda entry: entry["validation_accuracy"]
    )

    # Find the entry with the lowest validation accuracy
    worst_entry: ResultEntry = min(
        results,
        key=lambda entry: entry["validation_accuracy"]
    )

    # Print the results
    print("\nBest regularization parameter according to the validation set:")
    print(f"  lambda = {best_entry['lambda']}")
    print(f"  validation_accuracy = {best_entry['validation_accuracy']:.6f}")

    print("\nWorst regularization parameter according to the validation set:")
    print(f"  lambda = {worst_entry['lambda']}")
    print(f"  validation_accuracy = {worst_entry['validation_accuracy']:.6f}")

from typing import List
import numpy as np
from models import Ridge_Regression
from helpers import plot_decision_boundaries


def plot_best_and_worst_decision_boundaries(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """
    Trains ridge regression models for the best and worst regularization parameter
    values (hardcoded as 2.0 and 10.0), and plots their decision boundaries using
    the provided helper function.
    """

    # --------------------------------------------
    # Best regularization parameter (according to validation accuracy)
    # --------------------------------------------
    best_lambda: float = 2.0
    best_model = Ridge_Regression(lambd=best_lambda)
    best_model.fit(X_train, y_train)

    print(f"Plotting decision boundary for best lambda = {best_lambda}")
    print("best model W:", best_model.W)
    plot_decision_boundaries(
        model=best_model,
        X=X_test,  # transpose here
        y=y_test,
        title="Decision Boundary (Best lambda = 2.0)"
    )
    # --------------------------------------------
    # Worst regularization parameter
    # --------------------------------------------
    worst_lambda: float = 10.0
    worst_model = Ridge_Regression(lambd=worst_lambda)
    worst_model.fit(X_train, y_train)
    print(f"Plotting decision boundary for best lambda = {worst_lambda}")
    print("worst model W:", worst_model.W)
    plot_decision_boundaries(
        model=worst_model,
        X=X_test,  # transpose here
        y=y_test,
        title="Decision Boundary (Worst lambda = 10.0)"
    )








