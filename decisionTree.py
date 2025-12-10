import numpy as np
from sklearn.tree import DecisionTreeClassifier

from helpers import plot_decision_boundaries


def run_decision_tree(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    maximum_depth: int
) -> dict:

    decision_tree = DecisionTreeClassifier(max_depth=maximum_depth)
    decision_tree.fit(training_features, training_labels)

    training_accuracy = decision_tree.score(training_features, training_labels)
    validation_accuracy = decision_tree.score(validation_features, validation_labels)
    test_accuracy = decision_tree.score(test_features, test_labels)

    return {
        "model": decision_tree,
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy
    }


def decision_trees_stuff(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray
) -> tuple[dict, dict]:

    print("\n==============================")
    print(" Decision Tree: Maximum Depth = 2 ")
    print("==============================")

    result_depth_2 = run_decision_tree(
        training_features=training_features,
        training_labels=training_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        test_features=test_features,
        test_labels=test_labels,
        maximum_depth=2
    )

    print(f"Training accuracy:   {result_depth_2['training_accuracy']:.4f}")
    print(f"Validation accuracy: {result_depth_2['validation_accuracy']:.4f}")
    print(f"Test accuracy:       {result_depth_2['test_accuracy']:.4f}")

    plot_decision_boundaries(
        model=result_depth_2["model"],
        X=test_features,
        y=test_labels,
        title="Decision Tree (Maximum Depth = 2)"
    )


    print("\n==============================")
    print(" Decision Tree: Maximum Depth = 10 ")
    print("==============================")

    result_depth_10 = run_decision_tree(
        training_features=training_features,
        training_labels=training_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        test_features=test_features,
        test_labels=test_labels,
        maximum_depth=10
    )

    print(f"Training accuracy:   {result_depth_10['training_accuracy']:.4f}")
    print(f"Validation accuracy: {result_depth_10['validation_accuracy']:.4f}")
    print(f"Test accuracy:       {result_depth_10['test_accuracy']:.4f}")

    plot_decision_boundaries(
        model=result_depth_10["model"],
        X=test_features,
        y=test_labels,
        title="Decision Tree (Maximum Depth = 10)"
    )

    return result_depth_2, result_depth_10

