import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
#     """
#     Plots decision boundaries of a classifier and colors the space by the prediction of each point.
#
#     Parameters:
#     - model: The trained classifier (sklearn model).
#     - X: Numpy Feature matrix.
#     - y: Numpy array of Labels.
#     - title: Title for the plot.
#     """
#     # h = .02  # Step size in the mesh
#
#     # enumerate y
#     y_map = {v: i for i, v in enumerate(np.unique(y))}
#     enum_y = np.array([y_map[v] for v in y]).astype(int)
#
#     h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
#     h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200
#
#     # Plot the decision boundary.
#     added_margin_x = h_x * 20
#     added_margin_y = h_y * 20
#     x_min, x_max = X[:, 0].min() - added_margin_x, X[:, 0].max() + added_margin_x
#     y_min, y_max = X[:, 1].min() - added_margin_y, X[:, 1].max() + added_margin_y
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))
#
#     # Make predictions on the meshgrid points.
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     print(Z.shape)
#     Z = np.array([y_map[v] for v in Z])
#     Z = Z.reshape(xx.shape)
#     vmin = np.min([np.min(enum_y), np.min(Z)])
#     vmax = np.min([np.max(enum_y), np.max(Z)])
#
#     # Plot the decision boundary.
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)
#
#     # Scatter plot of the data points with matching colors.
#     plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)
#
#     plt.title("Decision Boundaries")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.title(title)
#     plt.show()

# def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
#     """
#     Works with models whose predict() expects shape (n_features, n_samples),
#     and with data X provided either as (n_samples, 2) or (2, n_samples).
#     """
#
#     # --- normalize orientation for plotting ---
#     # If X has shape (2, N), transpose it for visualization.
#     if X.shape[1] != 2 and X.shape[0] == 2:
#         X_plot = X.T
#     else:
#         X_plot = X
#
#     # enumerate y
#     y_map = {v: i for i, v in enumerate(np.unique(y))}
#     enum_y = np.array([y_map[v] for v in y]).astype(int)
#
#     h_x = (np.max(X_plot[:, 0]) - np.min(X_plot[:, 0])) / 200
#     h_y = (np.max(X_plot[:, 1]) - np.min(X_plot[:, 1])) / 200
#
#     added_margin_x = h_x * 20
#     added_margin_y = h_y * 20
#     x_min, x_max = X_plot[:, 0].min() - added_margin_x, X_plot[:, 0].max() + added_margin_x
#     y_min, y_max = X_plot[:, 1].min() - added_margin_y, X_plot[:, 1].max() + added_margin_y
#
#     xx, yy = np.meshgrid(
#         np.arange(x_min, x_max, h_x),
#         np.arange(y_min, y_max, h_y)
#     )
#
#     # Flatten grid for prediction
#     grid = np.c_[xx.ravel(), yy.ravel()]  # (M, 2)
#     grid_T = grid.T                       # (2, M)   model format
#
#     # Predict with model that uses (features, samples)
#     Z = model.predict(grid_T)
#     Z = np.array([y_map[v] for v in Z])
#     Z = Z.reshape(xx.shape)
#
#     vmin = min(enum_y.min(), Z.min())
#     vmax = max(enum_y.max(), Z.max())
#
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)
#     plt.scatter(
#         X_plot[:, 0], X_plot[:, 1],
#         c=enum_y, cmap=plt.cm.Paired,
#         edgecolors='k', s=40, alpha=0.7,
#         vmin=vmin, vmax=vmax
#     )
#
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.title(title)
#     plt.show()

def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
    """
    Works with ANY model that has predict():
    - Some models expect inputs shaped (n_samples, n_features)
    - Others expect inputs shaped (n_features, n_samples)

    This helper automatically detects and reshapes as needed.
    """

    # -------------------------------------------------------------
    # Normalize X to shape (N, 2) for plotting purposes
    # -------------------------------------------------------------
    if X.ndim == 2 and X.shape[1] != 2 and X.shape[0] == 2:
        X_plot = X.T  # (2, N) → (N, 2)
    else:
        X_plot = X.copy()  # already (N, 2)

    # -------------------------------------------------------------
    # Prepare labels (map to 0,1,...)
    # -------------------------------------------------------------
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    # -------------------------------------------------------------
    # Set grid resolution
    # -------------------------------------------------------------
    h_x = (np.max(X_plot[:, 0]) - np.min(X_plot[:, 0])) / 200
    h_y = (np.max(X_plot[:, 1]) - np.min(X_plot[:, 1])) / 200

    added_margin_x = h_x * 20
    added_margin_y = h_y * 20

    x_min = X_plot[:, 0].min() - added_margin_x
    x_max = X_plot[:, 0].max() + added_margin_x
    y_min = X_plot[:, 1].min() - added_margin_y
    y_max = X_plot[:, 1].max() + added_margin_y

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h_x),
        np.arange(y_min, y_max, h_y)
    )

    # -------------------------------------------------------------
    # Flatten grid into both shapes:
    #   grid    = (M, 2)
    #   grid_T  = (2, M)
    # -------------------------------------------------------------
    grid = np.c_[xx.ravel(), yy.ravel()]  # (M, 2)
    grid_T = grid.T  # (2, M)

    # -------------------------------------------------------------
    # Automatically choose correct shape for model.predict()
    # -------------------------------------------------------------
    try:
        # Try (M, 2) — typical PyTorch forward input
        Z = model.predict(grid)
    except Exception:
        # Fallback: try (2, M)
        Z = model.predict(grid_T)

    # Convert predicted labels to 0/1 mapping
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)

    vmin = min(enum_y.min(), Z.min())
    vmax = max(enum_y.max(), Z.max())

    # -------------------------------------------------------------
    # Plot decision boundary
    # -------------------------------------------------------------
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8,
                 vmin=vmin, vmax=vmax)

    plt.scatter(
        X_plot[:, 0], X_plot[:, 1],
        c=enum_y,
        cmap=plt.cm.Paired,
        edgecolors='k',
        s=40,
        alpha=0.7,
        vmin=vmin, vmax=vmax
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def read_data_demo(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    data_numpy = df.values

    return data_numpy, col_names

