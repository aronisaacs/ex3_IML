import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from models import Logistic_Regression

def get_device() -> torch.device:
    """
    Returns the best available device: CUDA → MPS → CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def run_binary_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:


    # -------------------------------------------------------------
    # Select device (CUDA → MPS → CPU)
    # -------------------------------------------------------------
    device = get_device()
    print(f"Using device: {device}")

    # -------------------------------------------------------------
    # Convert NumPy arrays to PyTorch tensors
    # -------------------------------------------------------------
    training_features = torch.tensor(X_train, dtype=torch.float32).to(device)
    training_labels = torch.tensor(y_train, dtype=torch.long).to(device)

    validation_features = torch.tensor(X_validation, dtype=torch.float32).to(device)
    validation_labels = torch.tensor(y_validation, dtype=torch.long).to(device)

    test_features = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_labels = torch.tensor(y_test, dtype=torch.long).to(device)

    # -------------------------------------------------------------
    # Create TensorDatasets
    # -------------------------------------------------------------
    training_dataset: TensorDataset = TensorDataset(training_features, training_labels)
    validation_dataset: TensorDataset = TensorDataset(validation_features, validation_labels)
    test_dataset: TensorDataset = TensorDataset(test_features, test_labels)

    # -------------------------------------------------------------
    # Create DataLoaders
    # -------------------------------------------------------------
    batch_size: int = 32

    training_loader: DataLoader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader: DataLoader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------------------------------------
    # Train three models with different learning rates
    # -------------------------------------------------------------
    learning_rates = [0.1, 0.01, 0.001]
    number_of_epochs = 10
    results = []

    for lr in learning_rates:
        print("=" * 60)
        print(f"Training logistic regression with learning rate = {lr}")
        print("=" * 60)

        model = Logistic_Regression(input_dim=2, output_dim=2).to(device)


        result = train_single_model(
            model=model,
            training_loader=training_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            learning_rate=lr,
            number_of_epochs=number_of_epochs
        )

        results.append(result)

    # -------------------------------------------------------------
    # Select best model based on validation accuracy
    # -------------------------------------------------------------
    best_result = max(
        results,
        key=lambda r: max(r["validation_accuracies"])
    )

    best_lr = best_result["learning_rate"]
    best_val = max(best_result["validation_accuracies"])
    best_test = max(best_result["test_accuracies"])

    print("\nBest Model Summary:")
    print(f"  Learning rate: {best_lr}")
    print(f"  Best validation accuracy: {best_val:.4f}")
    print(f"  Corresponding test accuracy: {best_test:.4f}")
    print("-" * 60)

    # TODO: visualization


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
    number_of_epochs: int
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
) -> float:
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

    return total_loss / total_samples

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





