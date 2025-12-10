from typing import Optional, Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from helpers import plot_decision_boundaries
from logistic_regression_utilities import run_logistic_regression_experiment



def run_binary_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
    )-> None:

    results = run_logistic_regression_experiment(
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test,
        output_dim=2,
        learning_rates=[0.1, 0.01, 0.001],
        number_of_epochs=10,
        batch_size=32,
        scheduler_factory=None
    )

    best_result = max(results, key=lambda r: max(r["validation_accuracies"]))
    # ---- print summary ----
    print("\nBest Model Summary:")
    print(f"  Learning rate: {best_result['learning_rate']}")
    print(f"  Best validation accuracy: {max(best_result['validation_accuracies']):.4f}")
    print(f"  Corresponding test accuracy: {max(best_result['test_accuracies']):.4f}")
    print("-" * 60)

    visualize_best_model_predictions(best_result, X_test, y_test)
    plot_full_training_curves(best_result)




from torch.optim.lr_scheduler import StepLR

def run_multiclass_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:

    # Determine number of classes automatically
    num_classes = len(np.unique(y_train))

    # Scheduler factory
    def scheduler_factory(optimizer):
        return StepLR(optimizer, step_size=5, gamma=0.3)

    # Train models for all learning rates
    results = run_logistic_regression_experiment(
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test,
        output_dim=num_classes,
        learning_rates=[0.01, 0.001, 0.0003],
        number_of_epochs=30,
        batch_size=32,
        scheduler_factory=scheduler_factory
    )

    # Pick best based on validation accuracy
    best_result = max(results, key=lambda r: max(r["validation_accuracies"]))

    print("\nBest Multiclass Model Summary:")
    print(f"  Learning rate: {best_result['learning_rate']}")
    print(f"  Best validation accuracy: {max(best_result['validation_accuracies']):.4f}")
    print(f"  Corresponding test accuracy: {max(best_result['test_accuracies']):.4f}")
    print("-" * 60)
    plot_accuracy_vs_learning_rate(results)
    plot_full_training_curves(best_result)

def run_ridge_logistic_regression(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> None:

    def scheduler_factory(optimizer):
        return StepLR(optimizer, step_size=5, gamma=0.3)

    results = []

    lambda_values = [0.0, 0.01, 0.1, 1.0, 10.0]
    for lambda_regularization in lambda_values:
        print(f"\n=========== Training with λ = {lambda_regularization} ===========")

        experiment_results = run_logistic_regression_experiment(
            X_train=training_features,
            y_train=training_labels,
            X_validation=validation_features,
            y_validation=validation_labels,
            X_test=test_features,
            y_test=test_labels,
            output_dim=len(np.unique(training_labels)),
            learning_rates=[0.01],        # assignment says fixed lr = 0.01
            number_of_epochs=30,
            scheduler_factory=scheduler_factory,
            lambda_regularization=lambda_regularization
        )

        # experiment_results is a list (one per learning rate)
        result = experiment_results[0]
        result["lambda_regularization"] = lambda_regularization
        results.append(result)

    # Choose best λ
    best_result = max(results, key=lambda r: max(r["validation_accuracies"]))

    print("\n========== Best Ridge Model ==========")
    print(f"Best λ: {best_result['lambda_regularization']}")
    print(f"Validation accuracy: {max(best_result['validation_accuracies']):.4f}")
    print(f"Test accuracy: {max(best_result['test_accuracies']):.4f}")

    plot_decision_boundaries(
        best_result["model"],
        test_features,
        test_labels,
        title=f"Ridge Logistic Regression (λ={best_result['lambda_regularization']})"
    )

    plot_full_training_curves(best_result)





def plot_accuracy_vs_learning_rate(results: list[dict]) -> None:
    learning_rates = [r["learning_rate"] for r in results]

    # Take the BEST accuracy from each model
    validation_accs = [max(r["validation_accuracies"]) for r in results]
    test_accs = [max(r["test_accuracies"]) for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(learning_rates, validation_accs, marker='o', label="Validation Accuracy")
    plt.plot(learning_rates, test_accs, marker='o', label="Test Accuracy")

    plt.xscale("log")  # optional but cleaner since LR values differ by factor 10

    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Learning Rate (Multiclass Logistic Regression)")
    plt.grid(True)
    plt.legend()
    plt.show()








def plot_full_training_curves(result: dict) -> None:
    epochs = range(1, len(result["training_losses"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, result["training_losses"], label="Training Loss")
    plt.plot(epochs, result["validation_losses"], label="Validation Loss")
    plt.plot(epochs, result["test_losses"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, result["training_accuracies"], label="Training Accuracy")
    plt.plot(epochs, result["validation_accuracies"], label="Validation Accuracy")
    plt.plot(epochs, result["test_accuracies"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_best_model_predictions(best_result, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Visualizes the decision boundaries and test predictions
    for the best logistic regression model.
    """

    best_model = best_result["model"]

    # Convert test points back to CPU numpy if needed
    # (plotting only works with CPU, numpy arrays)
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()

    print("Visualizing decision boundaries for the best model...")
    plot_decision_boundaries(best_model, X_test, y_test, title="Best Logistic Regression Model – Test Predictions")









