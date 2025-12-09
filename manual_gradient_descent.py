import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(num_iters: int = 1000, lr: float = 0.1) -> None:
    """
    Runs gradient descent on the function f(x,y) = (x - 3)^2 + (y - 5)^2.
    Prints the final point and plots the trajectory.
    Does NOT return anything.
    """

    xy: np.ndarray = np.array([0.0, 0.0])
    history: np.ndarray = np.zeros((num_iters, 2))

    for t in range(num_iters):
        grad: np.ndarray = np.array([2*(xy[0] - 3), 2*(xy[1] - 5)])
        xy = xy - lr * grad
        history[t] = xy

    # Print result
    print("Final point:", history[-1])

    # Plot result
    x_vals = history[:, 0]
    y_vals = history[:, 1]

    plt.scatter(x_vals, y_vals, c=np.arange(len(history)))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("GD trajectory")
    plt.colorbar(label="iteration")
    plt.show()


# ----------------------------------------------------------------------
# No code executes on import.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    gradient_descent()
