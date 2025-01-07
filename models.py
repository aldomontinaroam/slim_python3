def fit_slim_model(X, Y, constraints):
    """
    Fit SLIM model to given data

    :param X: Feature matrix
    :param Y: Target labels
    :param constraints: SLIMCoefficientConstraints
    """
    slim_ip, slim_info = create_slim_IP({'X': X, 'Y': Y, 'constraints': constraints})
    slim_ip.solve()

    return slim_ip.solution.get_values()
