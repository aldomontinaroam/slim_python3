import pandas as pd
import numpy as np
import os

def load_data(csv_path, outcome_col=0, zero_to_neg_one=True):
    """
    Load and prepare data from a CSV file for SLIM.

    Parameters
    ----------
    csv_path : str
        Full path to the CSV file containing data.
    outcome_col : int
        Index of the column containing the outcome variable.
    zero_to_neg_one : bool
        If True, convert 0 in the outcome column to -1.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    Y : np.ndarray, shape (n_samples,)
    X_names : list of str
        Names of the X columns. The intercept is *not* inserted here.
    Y_name : list of str
        Name of the outcome variable column.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"The CSV file {csv_path} is empty or unreadable.")

    # Convert to numpy
    data = df.values
    data_headers = list(df.columns.values)

    # Extract Y
    Y = data[:, outcome_col].copy()
    Y_name = [data_headers[outcome_col]]

    # Convert 0 in Y to -1 if requested
    if zero_to_neg_one:
        Y[Y == 0] = -1

    # Extract X by excluding outcome_col
    X_col_idx = [j for j in range(data.shape[1]) if j != outcome_col]
    X = data[:, X_col_idx]
    X_names = [data_headers[j] for j in X_col_idx]

    # insert a column of ones to X for the intercept
    X = np.insert(arr = X, obj = 0, values = np.ones(N), axis = 1)
    X_names.insert(0, '(Intercept)')

    return X, Y, X_names, Y_name

def check_data(X, Y, X_names=None):
    """
    Validate shapes, check no missing values, confirm Y in {-1, +1}, etc.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    Y : np.ndarray, shape (n_samples,) or (n_samples, 1)
    X_names : list of str, optional
        Names of each feature in X.

    Raises
    ------
    ValueError
        If data fails any validation checks.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Number of samples in X does not match number of labels in Y.")
    if not np.all(np.isin(np.unique(Y), [-1, +1])):
        raise ValueError("Outcome vector Y must only contain -1 or +1.")

    # Optionally check X for NaN, etc.
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Found NaN or Inf in X.")

    print("Data checks passed successfully.")
