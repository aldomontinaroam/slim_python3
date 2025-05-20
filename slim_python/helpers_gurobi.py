import os
import sys
import time
import numpy as np
import warnings
import re
from prettytable import PrettyTable
import gurobipy as gp
from gurobipy import GRB, GurobiError


class slimGurobiHelpers:
    """Raccolta di utility indipendenti dallo stato: tutti i metodi sono statici."""

    # ------------------------------------------------------------------
    # PRINTING / LOGGING
    # ------------------------------------------------------------------
    @staticmethod
    def print_log(msg, print_flag=True):
        if print_flag:
            timestamp = time.strftime("%m/%d/%y @ %I:%M %p", time.localtime())
            printable = msg if isinstance(msg, str) else repr(msg)
            print(f"{timestamp} | {printable}")
            sys.stdout.flush()

    @staticmethod
    def get_rho_string(rho, vtypes='I'):
        if len(vtypes) == 1:
            return ' '.join(str(int(x)) if vtypes == 'I' else str(x) for x in rho)

        out = []
        for j, val in enumerate(rho):
            if vtypes[j] == 'I':
                out.append(str(int(val)))
            else:
                out.append(f"{val:1.6f}")
        return ' '.join(out)

    # ------------------------------------------------------------------
    # LOADING SETTINGS
    # ------------------------------------------------------------------
    @staticmethod
    def easy_type(data_value):
        type_name = type(data_value).__name__
        if type_name in {"list", "set"}:
            types = {slimGurobiHelpers.easy_type(item) for item in data_value}
            if len(types) == 1:
                return next(iter(types))
            if types.issubset({"int", "float"}):
                return "float"
            return "multiple"
        if type_name == "str":
            return "bool" if data_value in {'True', 'TRUE', 'False', 'FALSE'} else "str"
        if type_name in {"int", "float", "bool"}:
            return type_name
        return "unknown"

    @staticmethod
    def convert_str_to_bool(val):
        val = val.lower().strip()
        if val == 'true':
            return True
        if val == 'false':
            return False
        return None

    @staticmethod
    def get_or_set_default(settings, name, default_value,
                           type_check=False, print_flag=False):
        if name in settings and type_check:
            if not isinstance(settings[name], type(default_value)):
                slimGurobiHelpers.print_log(
                    f"type mismatch on {name}: "
                    f"user type {type(settings[name])}, expected {type(default_value)}",
                    print_flag
                )
                settings[name] = default_value
        else:
            if name not in settings:
                slimGurobiHelpers.print_log(f"setting {name} to default {default_value!r}", print_flag)
            settings[name] = settings.get(name, default_value)
        return settings

    # ------------------------------------------------------------------
    # PROCESSING
    # ------------------------------------------------------------------
    @staticmethod
    def get_prediction(x, rho):
        return np.sign(x.dot(rho))

    @staticmethod
    def get_true_positives_from_pred(yhat, pos_ind):
        return np.sum(yhat[pos_ind] == 1)

    @staticmethod
    def get_false_positives_from_pred(yhat, pos_ind):
        return np.sum(yhat[~pos_ind] == 1)

    @staticmethod
    def get_true_negatives_from_pred(yhat, pos_ind):
        return np.sum(yhat[~pos_ind] != 1)

    @staticmethod
    def get_false_negatives_from_pred(yhat, pos_ind):
        return np.sum(yhat[pos_ind] != 1)

    @staticmethod
    def get_accuracy_stats(model, data, error_checking=True):
        acc = {f'{split}_{metric}': np.nan
               for split in ['train', 'valid', 'test']
               for metric in ['true_positives', 'true_negatives',
                              'false_positives', 'false_negatives']}

        model_vec = np.asarray(model).reshape(data['X'].shape[1], 1)

        def _fill(prefix, Xmat, Yvec):
            yhat = slimGurobiHelpers.get_prediction(Xmat, model_vec)
            pos  = Yvec == 1
            acc[f'{prefix}_true_positives']  = slimGurobiHelpers.get_true_positives_from_pred(yhat, pos)
            acc[f'{prefix}_true_negatives']  = slimGurobiHelpers.get_true_negatives_from_pred(yhat, pos)
            acc[f'{prefix}_false_positives'] = slimGurobiHelpers.get_false_positives_from_pred(yhat, pos)
            acc[f'{prefix}_false_negatives'] = slimGurobiHelpers.get_false_negatives_from_pred(yhat, pos)

            if error_checking:
                Ncheck = (acc[f'{prefix}_true_positives'] +
                          acc[f'{prefix}_true_negatives'] +
                          acc[f'{prefix}_false_positives'] +
                          acc[f'{prefix}_false_negatives'])
                assert Xmat.shape[0] == Ncheck

        # train
        _fill('train', data['X'], data['Y'])

        # valid
        if 'X_valid' in data and 'Y_valid' in data and data['X_valid'].size:
            _fill('valid', data['X_valid'], data['Y_valid'])

        # test
        if 'X_test' in data and 'Y_test' in data and data['X_test'].size:
            _fill('test', data['X_test'], data['Y_test'])

        return acc

    # ------------------------------------------------------------------
    # DATA CHECKING
    # ------------------------------------------------------------------
    @staticmethod
    def check_data(X, X_names, Y):
        assert isinstance(X, np.ndarray),  "X must be ndarray"
        assert isinstance(Y, np.ndarray),  "Y must be ndarray"
        assert isinstance(X_names, list),  "X_names must be list"

        N, P = X.shape
        assert N > 0 and P > 0
        assert len(Y) == N
        assert len(set(X_names)) == len(X_names)
        assert len(X_names) == P

        if '(Intercept)' in X_names:
            assert np.all(X[:, X_names.index('(Intercept)')] == 1.0)
        else:
            warnings.warn("no '(Intercept)' column")

        assert np.all(np.isfinite(X))
        assert np.all(np.isin(Y, [-1, 1]))

    # ------------------------------------------------------------------
    # POST-SOLVER CHECKS
    # ------------------------------------------------------------------
    @staticmethod
    def check_slim_IP_output(model, slim_info, X, Y, coef_constraints):
        """Replica dei test di coerenza, adattata a Gurobi."""
        assert model.NumVars == slim_info['n_variables']

        rho   = np.array([v.X for v in slim_info['rho_vars']])
        alpha = np.array([v.X for v in slim_info['alpha_vars']])
        err   = np.array([v.X for v in slim_info['error_vars']])

        L0_reg_ind = slim_info['L0_reg_ind']
        L1_reg_ind = slim_info['L1_reg_ind']
        rho_L0_reg = rho[L0_reg_ind]
        rho_L1_reg = rho[L1_reg_ind]

        # ricostruisci beta full
        beta_full = np.zeros_like(rho)
        for v in slim_info['beta_vars']:
            m = re.search(r'\[(\d+)\]$', v.VarName)
            j = int(m.group(1)) if m else int(v.VarName.split('_')[-1])
            beta_full[j] = v.X
        beta = beta_full[L1_reg_ind]

        total_error      = slim_info['total_error_var'].X
        total_error_pos  = slim_info['total_error_pos_var'].X
        total_error_neg  = slim_info['total_error_neg_var'].X
        total_l0_norm    = slim_info['total_l0_norm_var'].X

        beta_ub_reg = np.maximum(np.abs(coef_constraints.ub[L1_reg_ind]),
                                 coef_constraints.lb[L1_reg_ind])
        beta_lb_reg = np.maximum(0, slim_info['rho_lb'][L1_reg_ind])
        beta_lb_reg = np.maximum(beta_lb_reg, -slim_info['rho_ub'][L1_reg_ind])

        # coefficient vector tests
        assert rho.size == len(coef_constraints)
        assert np.all(rho <= slim_info['rho_ub'])
        assert np.all(rho >= slim_info['rho_lb'])

        # L0 indicator
        a_reg = alpha[L0_reg_ind]
        assert np.all(np.isin(a_reg, [0, 1]))
        assert np.all(np.abs(rho_L0_reg[a_reg == 0]) == 0)
        assert np.all(np.abs(rho_L0_reg[a_reg == 1]) > 0)

        # L1 helper
        assert np.all(np.abs(rho_L1_reg) == beta)
        assert np.all(beta >= beta_lb_reg)
        assert np.all(beta <= beta_ub_reg)

        # L0 norm bounds
        l0_norm = np.count_nonzero(rho[L0_reg_ind])
        assert total_l0_norm == l0_norm
        assert slim_info['L0_min'] <= l0_norm <= slim_info['L0_max']

        # error vector tests
        scores = (Y.reshape(-1, 1) * X).dot(rho)
        expected_err = scores <= slim_info['epsilon']
        assert np.all(np.isin(err, [0, 1]))
        assert np.all(err == expected_err)
        assert total_error == err.sum()
        assert total_error == total_error_pos + total_error_neg
        assert total_error_pos == err[slim_info['pos_ind']].sum()
        assert total_error_neg == err[slim_info['neg_ind']].sum()

    # ------------------------------------------------------------------
    # PRINT / SUMMARY UTILS
    # ------------------------------------------------------------------
    @staticmethod
    def print_slim_model(rho, X_names, Y_name, show_omitted_variables=False):
        rho_vals = np.copy(rho)
        rho_names = list(X_names)

        intercept_val = 0
        if '(Intercept)' in rho_names:
            idx = rho_names.index('(Intercept)')
            intercept_val = int(rho_vals[idx])
            rho_vals = np.delete(rho_vals, idx)
            rho_names.pop(idx)

        head = (f"PREDICT {Y_name[0].upper()} IF SCORE >= {intercept_val}"
                if Y_name else f"PREDICT Y = +1 IF SCORE >= {intercept_val}")

        if not show_omitted_variables:
            sel = np.flatnonzero(rho_vals)
            rho_vals = rho_vals[sel]
            rho_names = [rho_names[i] for i in sel]
            order = np.argsort(-rho_vals)
            rho_vals = rho_vals[order]
            rho_names = [rho_names[i] for i in order]

        rho_strings = [f"{int(v)} points" for v in rho_vals]
        total_str = f"ADD POINTS FROM ROWS 1 to {len(rho_vals)}"

        table = PrettyTable()
        table.field_names = ["Variable", "Points", "Tally"]
        table.add_row([head, "", ""])
        table.add_row(['=' * len(max([head, total_str] + rho_names, key=len)),
                       "=" * len(max(rho_strings + ["points"], key=len)),
                       "========="])
        for n, v in zip(rho_names, rho_strings):
            table.add_row([n, v, "+ ....."])
        table.add_row(['=' * len(max([head, total_str] + rho_names, key=len)),
                       "=" * len(max(rho_strings + ["points"], key=len)),
                       "========="])
        table.add_row([total_str, "SCORE", "= ....."])
        table.header = False
        table.align["Variable"] = "l"
        table.align["Points"] = "r"
        table.align["Tally"] = "r"

        return table

    @staticmethod
    def get_rho_summary(rho, slim_info, X, Y):
        pretty = slimGurobiHelpers.print_slim_model(
            rho, slim_info['X_names'], slim_info['Y_name']
        )

        #transform Y
        y = np.array(Y.flatten(), dtype=float)  # or dtype=np.float64
        pos_ind = y == 1
        neg_ind = ~pos_ind
        N = len(Y)
        N_pos = np.sum(pos_ind)
        N_neg = N - N_pos

        #get predictions
        yhat = X.dot(rho) > 0
        yhat = np.array(yhat, dtype = float)
        yhat[yhat == 0] = -1

        true_positives = np.sum(yhat[pos_ind] == 1)
        false_positives = np.sum(yhat[neg_ind] == 1)
        true_negatives= np.sum(yhat[neg_ind] == -1)
        false_negatives = np.sum(yhat[pos_ind] == -1)

        rho_summary = {
            'rho': rho,
            'pretty_model': pretty,
            'string_model': pretty.get_string(),
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

        return(rho_summary)

    # ------------------------------------------------------------------
    # OVERALL SUMMARY
    # ------------------------------------------------------------------
    @staticmethod
    def get_slim_summary(model, slim_info, X, Y):
        GRB_STATUS_STR = {
            GRB.OPTIMAL: 'OPTIMAL', GRB.SUBOPTIMAL: 'SUBOPTIMAL',
            GRB.INFEASIBLE: 'INFEASIBLE', GRB.UNBOUNDED: 'UNBOUNDED',
            GRB.INF_OR_UNBD: 'INF_OR_UNBD', GRB.NUMERIC: 'NUMERIC',
            GRB.INTERRUPTED: 'INTERRUPTED', GRB.TIME_LIMIT: 'TIME_LIMIT',
            GRB.NODE_LIMIT: 'NODE_LIMIT', GRB.SOLUTION_LIMIT: 'SOLUTION_LIMIT'
        }

        summary = {
            'solution_status_code': model.Status,
            'solution_status': GRB_STATUS_STR.get(model.Status, f'CODE_{model.Status}'),
            'objective_value': model.ObjVal if model.SolCount else np.nan,
            'optimality_gap': model.MIPGap if model.IsMIP and model.SolCount else np.nan,
            'objval_lowerbound': model.ObjBound if model.IsMIP else np.nan,
            'simplex_iterations': model.IterCount,
            'nodes_processed': model.NodeCount,
            'nodes_remaining': model.OpenNodeCount,
        }

        try:
            rho = np.array([v.X for v in slim_info['rho_vars']])
            summary.update(slimGurobiHelpers.get_rho_summary(rho, slim_info, X, Y))
        except GurobiError as e:
            slimGurobiHelpers.print_log(e)

        return summary