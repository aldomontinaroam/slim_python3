import os
import sys
import time
import numpy as np
import cplex
import warnings
from prettytable import PrettyTable


def print_log(msg, print_flag=True):
    """Prints a formatted log message."""
    if print_flag:
        timestamp = time.strftime("%m/%d/%y @ %I:%M %p", time.localtime())
        print(f'{timestamp} | {msg!r}')
        sys.stdout.flush()


def get_rho_string(rho, vtypes='I'):
    """Converts rho values into a formatted string."""
    if len(vtypes) == 1:
        return ' '.join(map(str, map(int, rho))) if vtypes == 'I' else ' '.join(map(str, rho))
    
    return ' '.join(
        str(int(val)) if vtypes[i] == 'I' else f'{val:.6f}'
        for i, val in enumerate(rho)
    )


def easy_type(data_value):
    """Determines a simplified type representation of the given value."""
    type_name = type(data_value).__name__
    if isinstance(data_value, (list, set)):
        types = {easy_type(item) for item in data_value}
        return next(iter(types)) if len(types) == 1 else "float" if types <= {"int", "float"} else "multiple"
    
    return {"str": "str", "int": "int", "float": "float", "bool": "bool"}.get(type_name, "unknown")


def convert_str_to_bool(val):
    """Converts string representation of boolean values to actual booleans."""
    return {"true": True, "false": False}.get(val.lower().strip())


def get_or_set_default(settings, setting_name, default_value, type_check=False, print_flag=False):
    """Retrieves a setting or sets it to a default if not present."""
    if setting_name in settings:
        if type_check and not isinstance(settings[setting_name], type(default_value)):
            print_log(f"Type mismatch on {setting_name}: expected {type(default_value)}, got {type(settings[setting_name])}", print_flag)
            settings[setting_name] = default_value
    else:
        print_log(f"Setting {setting_name} to its default value: {default_value}", print_flag)
        settings[setting_name] = default_value

    return settings


def get_prediction(x, rho):
    """Computes prediction using the given model coefficients."""
    return np.sign(x.dot(rho))


def check_data(X, X_names, Y):
    """Performs data validation checks."""
    assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray), "X and Y should be numpy arrays"
    assert isinstance(X_names, list), "X_names should be a list"
    
    N, P = X.shape
    assert N > 0 and P > 0, "X must have at least 1 row and 1 column"
    assert len(Y) == N, "Y should have the same number of rows as X"
    assert len(set(X_names)) == len(X_names) and len(X_names) == P, "X_names should be unique and match X columns"
    
    if "(Intercept)" in X_names:
        assert np.all(X[:, X_names.index("(Intercept)")] == 1.0), "Intercept column should be all 1s"
    else:
        warnings.warn("No intercept column found in X_names")
    
    assert not np.isnan(X).any() and not np.isinf(X).any(), "X contains NaN or Inf values"
    assert np.all((Y == 1) | (Y == -1)), "Y values should be -1 or 1"
    if np.all(Y == 1) or np.all(Y == -1):
        warnings.warn("All Y values are the same")
