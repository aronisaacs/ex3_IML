# types.py

import numpy as np
from typing import Dict

# Raw datasets (as loaded directly from CSV files)
XTrain = np.ndarray
YTrain = np.ndarray

XValidation = np.ndarray
YValidation = np.ndarray

XTest = np.ndarray
YTest = np.ndarray



ResultEntry = Dict[str, float]

