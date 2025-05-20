import gurobipy as gp
from gurobipy import GRB
import numpy as np
from math import ceil, floor
from .SLIMCoefficientConstraints import SLIMCoefficientConstraints
from .helpers_gurobi import slimGurobiHelpers as sgh

def create_slim_IP_gurobi(input, print_flag=False):
    if print_flag:
        def print_handle(msg): sgh.print_log(msg)
    else:
        def print_handle(msg): pass

    assert 'X' in input, 'no field named X  in input'
    assert 'X_names' in input, 'no field named X_names in input'
    assert 'Y' in input, 'no field named Y in input'
    assert input['X'].shape[0] == input['Y'].shape[0]
    assert input['X'].shape[1] == len(input['X_names'])
    assert all((input['Y'] == 1) | (input['Y'] == -1))

    XY = input['X'] * input['Y'][:, np.newaxis]

    # --- dimensioni e flag
    N  = input['X'].shape[0]
    P  = input['X'].shape[1]
    pos_ind, neg_ind = np.flatnonzero(input['Y'] == 1), np.flatnonzero(input['Y'] == -1)
    N_pos, N_neg = len(pos_ind), len(neg_ind)
    binary_data_flag = np.all((input['X'] == 0) | (input['X'] == 1))

    # --- nome output
    if ('Y_name' in input) and (type(input['Y_name']) is list):
        input['Y_name'] = input['Y_name'][0]
    elif ('Y_name' in input) and (type(input['Y_name']) is str):
        pass
    else:
        input['Y_name'] = 'Outcome'

    # --- default
    input = sgh.get_or_set_default(input, 'C_0', 0.01, print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'w_pos', 1.0, print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'w_neg', 2.0 - input['w_pos'], print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'L0_min', 0, print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'L0_max', P, print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'err_min', 0.00, print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'err_max', 1.00, print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'pos_err_min', 0.00, print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'pos_err_max', 1.00, print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'neg_err_min', 0.00, print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'neg_err_max', 1.00, print_flag=print_flag)

    # --- parametri interni
    input = sgh.get_or_set_default(input, 'C_1', float('nan'), print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'M',   float('nan'), print_flag=print_flag)
    input = sgh.get_or_set_default(input, 'epsilon', 0.001, print_flag=print_flag)

    # --- vincoli coeff
    coef_constraints = input.get('coef_constraints',
                                 SLIMCoefficientConstraints(variable_names=input['X_names']))
    assert len(coef_constraints) == P

    rho_lb, rho_ub = np.array(coef_constraints.lb), np.array(coef_constraints.ub)
    rho_max = np.maximum(np.abs(rho_lb), np.abs(rho_ub))
    beta_ub = rho_max
    beta_lb = np.zeros_like(rho_max)
    beta_lb[rho_lb > 0] = rho_lb[rho_lb > 0]
    beta_lb[rho_ub < 0] = rho_ub[rho_ub < 0]

    signs     = coef_constraints.sign
    sign_pos  = signs == 1
    sign_neg  = signs == -1
    types     = coef_constraints.get_field_as_list('vtype')
    rho_type  = ''.join(types)

    # --- pesi classe e regolarizzazione
    w_pos, w_neg = input['w_pos'], input['w_neg']
    w_total      = w_pos + w_neg
    w_pos, w_neg = 2.0*(w_pos/w_total), 2.0*(w_neg/w_total)
    assert abs(w_pos + w_neg - 2.0) < 1e-6

    C_0j = np.copy(coef_constraints.C_0j)
    L0_reg_ind = np.isnan(C_0j)
    C_0j[L0_reg_ind] = input['C_0']
    C_0 = C_0j
    assert all(C_0[L0_reg_ind] > 0.0)

    L1_reg_ind = L0_reg_ind
    if not np.isnan(input['C_1']):
        C_1 = input['C_1']
    else:
        C_1 = 0.5 * min(w_pos/N, w_neg/N, min(C_0[L1_reg_ind]/np.sum(rho_max)))
    C_1 = C_1 * np.ones(shape=(P,))
    C_1[~L1_reg_ind] = 0.0
    assert all(C_1[L1_reg_ind] > 0.0)

    # --- bounds cardinalità ed errori
    L0_min, L0_max = ceil(max(input['L0_min'], 0.0)), floor(min(input['L0_max'], np.sum(L0_reg_ind)))
    pos_err_min = max(ceil(N_pos*(0.0 if np.isnan(input['pos_err_min']) else input['pos_err_min'])), 0)
    pos_err_max = min(floor(N_pos*(1.0 if np.isnan(input['pos_err_max']) else input['pos_err_max'])), N_pos)
    neg_err_min = max(ceil(N_neg*(0.0 if np.isnan(input['neg_err_min']) else input['neg_err_min'])), 0)
    neg_err_max = min(floor(N_neg*(1.0 if np.isnan(input['neg_err_max']) else input['neg_err_max'])), N_neg)
    err_min     = max(ceil(N*(0.0 if np.isnan(input['err_min']) else input['err_min'])), 0)
    err_max     = min(floor(N*(1.0 if np.isnan(input['err_max']) else input['err_max'])), N)

    # --- M & epsilon
    epsilon = input['epsilon']
    if np.isnan(input['M']):
        max_points      = np.maximum(XY*rho_lb, XY*rho_ub)
        max_score_reg   = np.sum(-np.sort(-max_points[:, L0_reg_ind])[:, :int(L0_max)], axis=1)
        max_score_noReg = np.sum(max_points[:, ~L0_reg_ind], axis=1)
        max_score       = max_score_reg + max_score_noReg
        M = max_score + 1.05*epsilon
    else:
        M = input['M']
    M = M * np.ones(shape=(N,))

    # =============================================================================
    #               MODEL
    # =============================================================================
    model = gp.Model("SLIM_IP") 

    # --- variabili
    err_cost = np.ones(shape=(N,))
    err_cost[pos_ind], err_cost[neg_ind] = w_pos, w_neg
    C_0, C_1 = N*C_0, N*C_1
    obj      = np.r_[np.zeros(P), C_0, C_1, err_cost]

    # ρ (rho_j)
    rho = model.addVars(
        P, obj=obj[:P].tolist(),
        lb=rho_lb.tolist(), ub=rho_ub.tolist(),
        vtype=[GRB.CONTINUOUS if t == 'C' else GRB.INTEGER for t in rho_type],
        name="rho")   

    # α (alpha_j) – L0
    alpha = model.addVars(
        P, obj=C_0.tolist(),
        lb=[0]*P, ub=[1]*P,
        vtype=GRB.BINARY, name="alpha") 

    # β (beta_j) – L1
    beta = model.addVars(
        P, obj=C_1.tolist(),
        lb=beta_lb.tolist(), ub=beta_ub.tolist(),
        vtype=GRB.CONTINUOUS, name="beta") 

    # error_i
    error = model.addVars(
        N, obj=err_cost.tolist(),
        lb=[0]*N, ub=[1]*N,
        vtype=GRB.BINARY, name="error") 

    # variabili ausiliarie
    total_l0_norm  = model.addVar(lb=L0_min, ub=L0_max, vtype=GRB.INTEGER, name="total_l0_norm")
    total_error    = model.addVar(lb=err_min, ub=err_max, vtype=GRB.INTEGER, name="total_error")
    total_error_pos = model.addVar(lb=pos_err_min, ub=pos_err_max, vtype=GRB.INTEGER, name="total_error_pos")
    total_error_neg = model.addVar(lb=neg_err_min, ub=neg_err_max, vtype=GRB.INTEGER, name="total_error_neg")

    # --- vincoli di perdita
    for i in range(N):
        expr = gp.LinExpr(XY[i, :].tolist(), [rho[j] for j in range(P)])
        model.addConstr(expr + M[i]*error[i] >= epsilon,
                        name=f"error_{i}")

    # --- 0-norm (α ↔ ρ) lower & upper
    for j in range(P):
        model.addConstr(rho[j] - rho_lb[j]*alpha[j] >= 0, name=f"L0_norm_lb_{j}")
        model.addConstr(-rho[j] + rho_ub[j]*alpha[j] >= 0, name=f"L0_norm_ub_{j}")

    # --- 1-norm (β ≥ |ρ|)
    for j in range(P):
        model.addConstr(-rho[j] + beta[j] >= 0, name=f"L1_norm_pos_{j}")
        model.addConstr( rho[j] + beta[j] >= 0, name=f"L1_norm_neg_{j}")

    # --- variabili ausiliarie ↔ somme
    model.addConstr(gp.quicksum(alpha[j] for j in range(P)) == total_l0_norm,
                    name="total_L0_norm")
    model.addConstr(gp.quicksum(error[i] for i in pos_ind) == total_error_pos,
                    name="total_pos_error")
    model.addConstr(gp.quicksum(error[i] for i in neg_ind) == total_error_neg,
                    name="total_neg_error")
    model.addConstr(total_error == total_error_pos + total_error_neg,
                    name="total_error")

    # --- eliminazione di variabili/vincoli inutili
    no_L0_reg_ind = np.flatnonzero(~L0_reg_ind).tolist()
    no_L1_reg_ind = np.flatnonzero(~L1_reg_ind).tolist()
    fixed_value_ind = np.flatnonzero(rho_lb == rho_ub).tolist()

    # helper per nascondere variabili: in Gurobi si può fissare ub = lb = 0
    def deactivate(var_dict, idx_list):
        for j in idx_list:
            var_dict[j].ub = 0.0
            var_dict[j].lb = 0.0
    deactivate(alpha, no_L0_reg_ind + fixed_value_ind)
    deactivate(beta,  no_L0_reg_ind + no_L1_reg_ind + fixed_value_ind)

    model.update()   # ← prima sincronizzi

    expr  = gp.LinExpr()
    expr += gp.quicksum(C_0[j] * alpha[j] for j in range(P))
    expr += gp.quicksum(C_1[j] * beta[j]  for j in range(P))
    expr += gp.quicksum(err_cost[i] * error[i] for i in range(N))
    # (rho_j non ha costo → non serve aggiungerli)

    model.setObjective(expr, GRB.MINIMIZE)


    # =============================================================================
    #               ***   COSTRUZIONE slim_info ***
    # =============================================================================
    slim_info = {
        "C_0": C_0, "C_1": C_1,
        "w_pos": w_pos, "w_neg": w_neg,
        "err_min": err_min, "err_max": err_max,
        "pos_err_min": pos_err_min, "pos_err_max": pos_err_max,
        "neg_err_min": neg_err_min, "neg_err_max": neg_err_max,
        "L0_min": L0_min, "L0_max": L0_max,
        "N": N, "P": P, "N_pos": N_pos, "N_neg": N_neg,
        "rho_ub": rho_ub, "rho_lb": rho_lb,
        "M": M, "epsilon": epsilon,
        "binary_data_flag": binary_data_flag,
        "pos_ind": pos_ind, "neg_ind": neg_ind,
        "L0_reg_ind": L0_reg_ind, "L1_reg_ind": L1_reg_ind,
        "n_variables": model.NumVars, "n_constraints": model.NumConstrs,
        "names": [v.VarName for v in model.getVars()],
        #
        "rho_vars":   [rho[j]   for j in range(P)],
        "alpha_vars": [alpha[j] for j in range(P)],
        "beta_vars":  [beta[j]  for j in range(P)],
        "error_vars": [error[i] for i in range(N)],
        "total_l0_norm_var": total_l0_norm,
        "total_error_var": total_error,
        "total_error_pos_var": total_error_pos,
        "total_error_neg_var": total_error_neg,
        #
        "X_names": input['X_names'], "Y_name": input['Y_name'],
    }

    return model, slim_info