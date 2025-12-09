from typing import Optional, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler

from helpers import plot_decision_boundaries
from models import Logistic_Regression

def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader]:

    training_features = torch.tensor(X_train, dtype=torch.float32)
    training_labels = torch.tensor(y_train, dtype=torch.long)

    validation_features = torch.tensor(X_validation, dtype=torch.float32)
    validation_labels = torch.tensor(y_validation, dtype=torch.long)

    test_features = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)

    training_dataset = TensorDataset(training_features, training_labels)
    validation_dataset = TensorDataset(validation_features, validation_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return training_loader, validation_loader, test_loader

def run_logistic_regression_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dim: int,
    learning_rates: list[float],
    number_of_epochs: int,
    batch_size: int = 32,
    scheduler_factory: Optional[Callable[[torch.optim.Optimizer], LRScheduler]] = None
) -> list[dict]:

    # Build dataloaders
    training_loader, validation_loader, test_loader = build_dataloaders(
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test,
        batch_size
    )

    results = []

    for lr in learning_rates:
        print("=" * 60)
        print(f"Training logistic regression with learning rate = {lr}")
        print("=" * 60)

        model = Logistic_Regression(input_dim=X_train.shape[1], output_dim=output_dim)

        if scheduler_factory is not None:
            # Temporarily build an optimizer just to pass into scheduler_factory
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            scheduler = scheduler_factory(optimizer)
        else:
            scheduler = None

        result = train_single_model(
            model,
            training_loader,
            validation_loader,
            test_loader,
            learning_rate=lr,
            number_of_epochs=number_of_epochs,
            scheduler=scheduler
        )

        results.append(result)
    return results




def run_binary_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
    ) -> dict:

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

    # ---- restore plotting behavior ----
    visualize_best_model_predictions(best_result, X_test, y_test)
    plot_full_training_curves(best_result)

    return best_result




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














def compute_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    """
    Computes accuracy of the model over the given DataLoader.
    Returns the accuracy as a float in [0, 1].
    """

    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)                      # shape: (batch_size, num_classes)
            predicted_labels = torch.argmax(outputs, dim=1)

            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy


def compute_loss(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module
) -> float:
    """
    Computes the average loss of the model over the given DataLoader.
    Returns the average loss as a float.
    """

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            loss = loss_function(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    average_loss = total_loss / total_samples
    return average_loss



def train_single_model(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    learning_rate: float,
    number_of_epochs: int,
    scheduler: Optional[LRScheduler] = None
) -> dict:
    """
    Trains a logistic regression model using SGD,
    storing accuracy and loss curves for all datasets.
    """

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    training_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    training_losses = []
    validation_losses = []
    test_losses = []

    for epoch in range(number_of_epochs):

        # Train one epoch
        train_one_epoch(
            model=model,
            training_loader=training_loader,
            optimizer=optimizer,
            loss_function=loss_function
        )
        if scheduler is not None:
            scheduler.step()

        # Evaluate all data splits
        metrics = evaluate_all_datasets(
            model=model,
            training_loader=training_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            loss_function=loss_function
        )

        # Print via helper function
        print_epoch_metrics(
            epoch=epoch,
            number_of_epochs=number_of_epochs,
            metrics=metrics
        )

        # Store metrics
        training_accuracies.append(metrics["training_accuracy"])
        validation_accuracies.append(metrics["validation_accuracy"])
        test_accuracies.append(metrics["test_accuracy"])

        training_losses.append(metrics["training_loss"])
        validation_losses.append(metrics["validation_loss"])
        test_losses.append(metrics["test_loss"])

    return {
        "training_accuracies": training_accuracies,
        "validation_accuracies": validation_accuracies,
        "test_accuracies": test_accuracies,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "test_losses": test_losses,
        "model": model,
        "learning_rate": learning_rate,
    }



def evaluate_all_datasets(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    loss_function: nn.Module
) -> dict:
    """
    Computes accuracy and loss on training, validation, and test sets
    in one unified place.
    """

    training_accuracy = compute_accuracy(model, training_loader)
    training_loss = compute_loss(model, training_loader, loss_function)

    validation_accuracy = compute_accuracy(model, validation_loader)
    validation_loss = compute_loss(model, validation_loader, loss_function)

    test_accuracy = compute_accuracy(model, test_loader)
    test_loss = compute_loss(model, test_loader, loss_function)

    return {
        "training_accuracy": training_accuracy,
        "training_loss": training_loss,
        "validation_accuracy": validation_accuracy,
        "validation_loss": validation_loss,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
    }


def train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module
) -> None:
    """
    Runs one epoch of SGD over the training set.
    Returns the average loss over this epoch.
    """

    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_features, batch_labels in training_loader:


        # Forward pass
        predictions = model(batch_features)
        loss = loss_function(predictions, batch_labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size



def print_epoch_metrics(
    epoch: int,
    number_of_epochs: int,
    metrics: dict
) -> None:
    """
    Prints the training, validation, and test loss/accuracy for a given epoch.
    """

    print(f"Epoch {epoch}/{number_of_epochs}")
    print(f"  Training    - Loss: {metrics['training_loss']:.4f}, "
          f"Accuracy: {metrics['training_accuracy']:.4f}")
    print(f"  Validation  - Loss: {metrics['validation_loss']:.4f}, "
          f"Accuracy: {metrics['validation_accuracy']:.4f}")
    print(f"  Test        - Loss: {metrics['test_loss']:.4f}, "
          f"Accuracy: {metrics['test_accuracy']:.4f}")
    print("-" * 60)

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


import matplotlib.pyplot as plt


def plot_loss_curves(best_result: dict) -> None:
    """
    Plots training, validation, and test losses over epochs
    for the chosen logistic regression model.
    """

    training_losses = best_result["training_losses"]
    validation_losses = best_result["validation_losses"]
    test_losses = best_result["test_losses"]

    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.plot(epochs, test_losses, label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves for Best Logistic Regression Model")
    plt.legend()
    plt.grid(True)
    plt.show()




