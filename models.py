from .optimization import *
def fit_slim_model(X, Y, constraints):
    """
    Fit SLIM model to given data

    :param X: Feature matrix
    :param Y: Target labels
    :param constraints: SLIMCoefficientConstraints
    """
    # Prepare input for create_slim_IP
    input_data = {
        'X': X.to_numpy(),
        'Y': Y,
        'X_names': X.columns.tolist(),  # Include feature names
        'constraints': constraints
    }
    
    # Create SLIM Integer Programming model
    slim_ip, slim_info = create_slim_IP(input_data)
    
    # Solve the model
    slim_ip.solve()
    
    # Extract the coefficients from the solution
    coefficients = slim_ip.solution.get_values()
    
    return coefficients, slim_info
