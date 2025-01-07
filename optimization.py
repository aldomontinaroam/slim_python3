import cplex
import numpy as np
from math import ceil, floor
from .utils import *
from .constraints import SLIMCoefficientConstraints

def create_slim_IP(input_data, print_flag=False):
    """
    Create SLIM Integer Programming model

    :param input_data: dict containing 'X', 'Y', and constraint details
    :param print_flag: whether to print logs
    :return: cplex model and related information
    """
    # Setup printing
    def print_handle(msg):
        if print_flag:
            print_log(msg)

    # Check preconditions
    assert 'X' in input_data, 'no field named X in input'
    assert 'X_names' in input_data, 'no field named X_names in input'
    assert 'Y' in input_data, 'no field named Y in input'
    assert input_data['X'].shape[0] == input_data['Y'].shape[0]
    assert input_data['X'].shape[1] == len(input_data['X_names'])
    assert all((input_data['Y'] == 1) | (input_data['Y'] == -1))

    X = input_data['X']
    Y = input_data['Y']
    XY = X * Y[:, np.newaxis]  # Align shapes for element-wise multiplication

    # Sizes
    N = X.shape[0]
    P = X.shape[1]
    pos_ind = np.flatnonzero(Y == 1)
    neg_ind = np.flatnonzero(Y == -1)
    N_pos = len(pos_ind)
    N_neg = len(neg_ind)

    # Outcome variable name
    if 'Y_name' in input_data and isinstance(input_data['Y_name'], list):
        input_data['Y_name'] = input_data['Y_name'][0]
    elif 'Y_name' not in input_data:
        input_data['Y_name'] = 'Outcome'

    # Set default parameters
    input_data = get_or_set_default(input_data, 'C_0', 0.01, print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'w_pos', 1.0, print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'w_neg', 2.0 - input_data['w_pos'], print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'L0_min', 0, print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'L0_max', P, print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'err_min', 0.00, print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'err_max', 1.00, print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'pos_err_min', 0.00, print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'pos_err_max', 1.00, print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'neg_err_min', 0.00, print_flag=print_flag)
    input_data = get_or_set_default(input_data, 'neg_err_max', 1.00, print_flag=print_flag)

    # Coefficient constraints
    if 'coef_constraints' in input_data:
        coef_constraints = input_data['coef_constraints']
    else:
        coef_constraints = SLIMCoefficientConstraints(variable_names=input_data['X_names'])

    assert len(coef_constraints) == P

    # Bounds and other parameters
    rho_lb = np.array(coef_constraints.lb)
    rho_ub = np.array(coef_constraints.ub)
    rho_max = np.maximum(np.abs(rho_lb), np.abs(rho_ub))
    beta_ub = rho_max
    beta_lb = np.zeros_like(rho_max)
    beta_lb[rho_lb > 0] = rho_lb[rho_lb > 0]
    beta_lb[rho_ub < 0] = rho_ub[rho_ub < 0]

    # L0 regularization penalty
    C_0j = np.copy(coef_constraints.C_0j)
    L0_reg_ind = np.isnan(C_0j)
    C_0j[L0_reg_ind] = input_data['C_0']
    C_0 = C_0j

    # Weights for misclassification
    w_pos = input_data['w_pos']
    w_neg = input_data['w_neg']
    w_total = w_pos + w_neg
    w_pos = 2.0 * (w_pos / w_total)
    w_neg = 2.0 * (w_neg / w_total)

    # Total error bounds
    err_min = max(ceil(N * input_data['err_min']), 0)
    err_max = min(floor(N * input_data['err_max']), N)

    # Initialize CPLEX model
    slim_IP = cplex.Cplex()
    slim_IP.objective.set_sense(slim_IP.objective.sense.minimize)

    # Define variables: rho, alpha, error
    rho_names = [f"rho_{j}" for j in range(P)]
    alpha_names = [f"alpha_{j}" for j in range(P)]
    error_names = [f"error_{i}" for i in range(N)]

    slim_IP.variables.add(
        obj=[0.0] * P + [C_0] * P + [w_pos] * N,
        lb=rho_lb.tolist() + [0] * P + [0] * N,
        ub=rho_ub.tolist() + [1] * P + [1] * N,
        types="I" * P + "B" * P + "B" * N,
        names=rho_names + alpha_names + error_names
    )

    # Add L0-norm constraints
    for j in range(P):
        slim_IP.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[rho_names[j], alpha_names[j]], val=[1.0, -rho_lb[j]])],
            senses="G",
            rhs=[0.0],
            names=[f"L0_norm_{j}"]
        )

    # Add misclassification constraints
    for i in range(N):
        slim_IP.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=rho_names + [error_names[i]], val=(XY[i, :].tolist() + [1.0]))],
            senses="G",
            rhs=[0.0],
            names=[f"error_{i}"]
        )

    # Additional info for debugging and validation
    slim_info = {
        "rho_names": rho_names,
        "alpha_names": alpha_names,
        "error_names": error_names,
        "X": X,
        "Y": Y,
        "N": N,
        "P": P,
        "C_0": C_0
    }

    return slim_IP, slim_info
