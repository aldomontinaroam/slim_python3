import os
import sys
import time
import numpy as np
import cplex
import warnings
from prettytable import PrettyTable

# PRINTING AND LOGGING
def print_log(msg, print_flag=True):
    if print_flag:
        if isinstance(msg, str):
            print('%s | %s' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg))
        else:
            print('%s | %r' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg))
        sys.stdout.flush()

def get_rho_string(rho, vtypes='I'):
    if len(vtypes) == 1:
        if vtypes == 'I':
            rho_string = ' '.join(map(lambda x: str(int(x)), rho))
        else:
            rho_string = ' '.join(map(str, rho))
    else:
        rho_string = ''
        for j in range(len(rho)):
            if vtypes[j] == 'I':
                rho_string += ' ' + str(int(rho[j]))
            else:
                rho_string += (' %1.6f' % rho[j])

    return rho_string

# LOADING SETTINGS FROM DISK
def easy_type(data_value):
    type_name = type(data_value).__name__
    if type_name in {"list", "set"}:
        types = {easy_type(item) for item in data_value}
        if len(types) == 1:
            return next(iter(types))
        elif types.issubset({"int", "float"}):
            return "float"
        else:
            return "multiple"
    elif type_name == "str":
        if data_value in {'True', 'TRUE'}:
            return "bool"
        elif data_value in {'False', 'FALSE'}:
            return "bool"
        else:
            return "str"
    elif type_name == "int":
        return "int"
    elif type_name == "float":
        return "float"
    elif type_name == "bool":
        return "bool"
    else:
        return "unknown"

def convert_str_to_bool(val):
    val = val.lower().strip()
    if val == 'true':
        return True
    elif val == 'false':
        return False
    else:
        return None

def get_or_set_default(settings, setting_name, default_value, type_check=False, print_flag=False):
    if setting_name in settings:
        if type_check:
            default_type = type(default_value)
            user_type = type(settings[setting_name])
            if user_type != default_type:
                print_log("Type mismatch on %s: user provided type %s but expected type %s" % (setting_name, user_type, default_type), print_flag)
                print_log("Setting %s to its default value: %r" % (setting_name, default_value), print_flag)
                settings[setting_name] = default_value
    else:
        print_log("Setting %s to its default value: %r" % (setting_name, default_value), print_flag)
        settings[setting_name] = default_value

    return settings

# PROCESSING
def get_prediction(x, rho):
    return np.sign(x.dot(rho))

def get_true_positives_from_pred(yhat, pos_ind):
    return np.sum(yhat[pos_ind] == 1)

def get_false_positives_from_pred(yhat, pos_ind):
    return np.sum(yhat[~pos_ind] == 1)

def get_true_negatives_from_pred(yhat, pos_ind):
    return np.sum(yhat[~pos_ind] != 1)

def get_false_negatives_from_pred(yhat, pos_ind):
    return np.sum(yhat[pos_ind] != 1)

def get_accuracy_stats(model, data, error_checking=True):
    accuracy_stats = {
        'train_true_positives': np.nan,
        'train_true_negatives': np.nan,
        'train_false_positives': np.nan,
        'train_false_negatives': np.nan,
        'valid_true_positives': np.nan,
        'valid_true_negatives': np.nan,
        'valid_false_positives': np.nan,
        'valid_false_negatives': np.nan,
        'test_true_positives': np.nan,
        'test_true_negatives': np.nan,
        'test_false_positives': np.nan,
        'test_false_negatives': np.nan,
    }

    model = np.array(model).reshape(data['X'].shape[1], 1)

    for data_prefix in ['train', 'valid', 'test']:
        X_field_name = 'X' if data_prefix == 'train' else f"X_{data_prefix}"
        Y_field_name = 'Y' if data_prefix == 'train' else f"Y_{data_prefix}"

        if X_field_name in data and Y_field_name in data:
            X, Y = data[X_field_name], data[Y_field_name]
            Yhat = get_prediction(X, model)
            pos_ind = Y == 1

            accuracy_stats[f"{data_prefix}_true_positives"] = get_true_positives_from_pred(Yhat, pos_ind)
            accuracy_stats[f"{data_prefix}_true_negatives"] = get_true_negatives_from_pred(Yhat, pos_ind)
            accuracy_stats[f"{data_prefix}_false_positives"] = get_false_positives_from_pred(Yhat, pos_ind)
            accuracy_stats[f"{data_prefix}_false_negatives"] = get_false_negatives_from_pred(Yhat, pos_ind)

            if error_checking:
                N_check = (accuracy_stats[f"{data_prefix}_true_positives"] +
                           accuracy_stats[f"{data_prefix}_true_negatives"] +
                           accuracy_stats[f"{data_prefix}_false_positives"] +
                           accuracy_stats[f"{data_prefix}_false_negatives"])
                assert X.shape[0] == N_check

    return accuracy_stats

# DATA CHECKING
def check_data(X, X_names, Y):
    assert isinstance(X, np.ndarray), "X must be a numpy.ndarray"
    assert isinstance(Y, np.ndarray), "Y must be a numpy.ndarray"
    assert isinstance(X_names, list), "X_names must be a list"

    N, P = X.shape
    assert N > 0, "X must have at least one row"
    assert P > 0, "X must have at least one column"
    assert len(Y) == N, "Length of Y must match the number of rows in X"
    assert len(set(X_names)) == len(X_names), "X_names must be unique"
    assert len(X_names) == P, "Length of X_names must match the number of columns in X"

    if '(Intercept)' in X_names:
        intercept_index = X_names.index('(Intercept)')
        assert np.all(X[:, intercept_index] == 1), "Intercept column must be all ones"
    else:
        warnings.warn("No '(Intercept)' column found in X_names")

    assert not np.any(np.isnan(X)), "X contains NaN values"
    assert not np.any(np.isinf(X)), "X contains Inf values"

    assert np.all(np.isin(Y, [-1, 1])), "Y must contain only -1 and 1"
    if np.all(Y == 1):
        warnings.warn("All values in Y are 1")
    if np.all(Y == -1):
        warnings.warn("All values in Y are -1")

# PRINTING MODEL
def print_slim_model(rho, X_names, Y_name, show_omitted_variables=False):
    rho_values = np.copy(rho)
    rho_names = list(X_names)

    if '(Intercept)' in rho_names:
        intercept_ind = rho_names.index('(Intercept)')
        intercept_val = int(rho[intercept_ind])
        rho_values = np.delete(rho_values, intercept_ind)
        rho_names.pop(intercept_ind)
    else:
        intercept_val = 0

    predict_string = f"PREDICT {Y_name.upper()} IF SCORE >= {intercept_val}" if Y_name else f"PREDICT Y = +1 IF SCORE >= {intercept_val}"

    if not show_omitted_variables:
        selected_ind = np.flatnonzero(rho_values)
        rho_values = rho_values[selected_ind]
        rho_names = [rho_names[i] for i in selected_ind]

        sort_ind = np.argsort(-rho_values)
        rho_values = rho_values[sort_ind]
        rho_names = [rho_names[i] for
        i in sort_ind]

    rho_values_string = [f"{int(value)} points" for value in rho_values]
    n_variable_rows = len(rho_values)
    total_string = f"ADD POINTS FROM ROWS 1 to {n_variable_rows}"

    max_name_col_length = max(len(predict_string), len(total_string), max([len(name) for name in rho_names], default=0)) + 2
    max_value_col_length = max(7, max([len(value) for value in rho_values_string], default=0)) + 2

    table = PrettyTable()
    table.field_names = ["Variable", "Points", "Tally"]
    table.add_row([predict_string, "", ""])
    table.add_row(["=" * max_name_col_length, "=" * max_value_col_length, "========="])

    for i in range(n_variable_rows):
        table.add_row([rho_names[i], rho_values_string[i], "+ ....."])

    table.add_row(["=" * max_name_col_length, "=" * max_value_col_length, "========="])
    table.add_row([total_string, "SCORE", "= ....."])
    table.header = False
    table.align["Variable"] = "l"
    table.align["Points"] = "r"
    table.align["Tally"] = "r"

    return table

# RHO SUMMARY
def get_rho_summary(rho, slim_info, X, Y):
    printed_model = print_slim_model(rho, X_names=slim_info['X_names'], Y_name=slim_info['Y_name'], show_omitted_variables=False)

    y = np.array(Y.flatten(), dtype=float)
    pos_ind = y == 1
    neg_ind = ~pos_ind
    N = len(Y)
    N_pos = np.sum(pos_ind)
    N_neg = N - N_pos

    yhat = X.dot(rho) > 0
    yhat = np.array(yhat, dtype=float)
    yhat[yhat == 0] = -1

    true_positives = np.sum(yhat[pos_ind] == 1)
    false_positives = np.sum(yhat[neg_ind] == 1)
    true_negatives = np.sum(yhat[neg_ind] == -1)
    false_negatives = np.sum(yhat[pos_ind] == -1)

    rho_summary = {
        'rho': rho,
        'pretty_model': printed_model,
        'string_model': printed_model.get_string(),
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'mistakes': np.sum(y != yhat),
        'error_rate': (false_positives + false_negatives) / N,
        'true_positive_rate': true_positives / N_pos,
        'false_positive_rate': false_positives / N_neg,
        'L0_norm': np.sum(rho[slim_info['L0_reg_ind']]),
    }

    return rho_summary

# SLIM SUMMARY
def get_slim_summary(slim_IP, slim_info, X, Y):
    slim_summary = {
        'solution_status_code': slim_IP.solution.get_status(),
        'solution_status': slim_IP.solution.get_status_string(slim_IP.solution.get_status()),
        'objective_value': slim_IP.solution.get_objective_value(),
        'optimality_gap': slim_IP.solution.MIP.get_mip_relative_gap(),
        'objval_lowerbound': slim_IP.solution.MIP.get_best_objective(),
        'simplex_iterations': slim_IP.solution.progress.get_num_iterations(),
        'nodes_processed': slim_IP.solution.progress.get_num_nodes_processed(),
        'nodes_remaining': slim_IP.solution.progress.get_num_nodes_remaining(),
        'rho': np.nan,
        'pretty_model': np.nan,
        'string_model': np.nan,
        'true_positives': np.nan,
        'true_negatives': np.nan,
        'false_positives': np.nan,
        'false_negatives': np.nan,
        'mistakes': np.nan,
        'error_rate': np.nan,
        'true_positive_rate': np.nan,
        'false_positive_rate': np.nan,
        'L0_norm': np.nan,
    }

    try:
        rho = np.array(slim_IP.solution.get_values(slim_info['rho_idx']))
        slim_summary.update(get_rho_summary(rho, slim_info, X, Y))
    except cplex.exceptions.CplexError as e:
        print_log(e)

    return slim_summary
