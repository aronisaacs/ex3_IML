import numpy as np
import torch
from torch import nn



class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd
        self.W = None

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5) # transform the labels to -1 and 1, instead of 0 and 1.

        number_of_training_samples: int = X.shape[1]
        number_of_features: int = X.shape[0]

        identity_matrix: np.ndarray = np.eye(number_of_features)

        # Compute (X X^T / N + lambda I)
        left_matrix: np.ndarray = (X @ X.T) / number_of_training_samples
        left_matrix = left_matrix + self.lambd * identity_matrix

        # Compute (X Y^T / N)
        right_vector: np.ndarray = (X @ Y.T) / number_of_training_samples

        # Closed-form ridge solution
        self.W = np.linalg.inv(left_matrix) @ right_vector


    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. 
        :return: The predicted output. 
        """
        # Compute the raw linear scores: shape (1, number_of_samples)
        raw_scores: np.ndarray = self.W.T @ X

        # Convert raw scores into {-1, +1} decisions
        predictions: np.ndarray = np.sign(raw_scores)

        # Flatten to shape (number_of_samples,)
        predictions = predictions.flatten()

        # Convert from {-1, +1} into {0, 1}
        predictions = (predictions + 1) / 2

        return predictions



# class Logistic_Regression(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Logistic_Regression, self).__init__()
#
#         ########## YOUR CODE HERE ##########
#
#         # define a linear operation.
#
#         ####################################
#         pass
#
#     def forward(self, x):
#         """
#         Computes the output of the linear operator.
#         :param x: The input to the linear operator.
#         :return: The transformed input.
#         """
#         # compute the output of the linear operator
#
#         ########## YOUR CODE HERE ##########
#
#         # return the transformed input.
#         # first perform the linear operation
#         # should be a single line of code.
#
#         ####################################
#
#         pass
#
#     def predict(self, x):
#         """
#         THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
#         """
#         x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
#         x = self.forward(x)
#         x = nn.functional.softmax(x, dim=1)
#         x = x.detach().cpu().numpy()
#         x = np.argmax(x, axis=1)
#         return x

class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()

        ########## YOUR CODE HERE ##########
        # define a linear operation.
        self.linear = nn.Linear(input_dim, output_dim)
        ####################################


    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """

        ########## YOUR CODE HERE ##########
        # return the transformed input.
        # first perform the linear operation
        return self.linear(x)
        ####################################


    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x
