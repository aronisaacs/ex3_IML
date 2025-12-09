# main.py

import numpy as np
from helpers import read_data_demo
from ridge_regression import run_ridge_regression


if __name__ == "__main__":

    # -------------------------------------------------------------
    # Load binary classification data (original shape: X = (N, d), y = (N,))
    # -------------------------------------------------------------
    train_data, train_columns = read_data_demo("train.csv")
    validation_data, validation_columns = read_data_demo("validation.csv")
    test_data, test_columns = read_data_demo("test.csv")

    X_train: np.ndarray = train_data[:, :2]
    y_train: np.ndarray = train_data[:, 2]

    X_validation: np.ndarray = validation_data[:, :2]
    y_validation: np.ndarray = validation_data[:, 2]

    X_test: np.ndarray = test_data[:, :2]
    y_test: np.ndarray = test_data[:, 2]

    # -------------------------------------------------------------
    # Load multiclass classification data (also original shape)
    # -------------------------------------------------------------
    # -----------------------------
    # Load the multiclass datasets
    # -----------------------------
    train_multiclass, train_multiclass_columns = read_data_demo("train_multiclass.csv")
    validation_multiclass, validation_multiclass_columns = read_data_demo("validation_multiclass.csv")
    test_multiclass, test_multiclass_columns = read_data_demo("test_multiclass.csv")

    # -----------------------------
    # Extract features and labels
    # -----------------------------
    X_train_multiclass = train_multiclass[:, :2]
    y_train_multiclass = train_multiclass[:, 2]

    X_validation_multiclass = validation_multiclass[:, :2]
    y_validation_multiclass = validation_multiclass[:, 2]

    X_test_multiclass = test_multiclass[:, :2]
    y_test_multiclass = test_multiclass[:, 2]

    # -------------------------------------------------------------
    # Run Part 3 (Ridge Regression)
    #
    # NOTE:
    #     Ridge regression will reshape the data internally.
    #     main.py should NOT worry about shapes or transposes.
    # -------------------------------------------------------------
    run_ridge_regression(
        X_train, y_train,
        X_validation,   y_validation,
        X_test,  y_test
    )
