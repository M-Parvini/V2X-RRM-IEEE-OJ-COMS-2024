import numpy as np

def ode_platoon(t, y, AA, BB, input_val):
    """
    :param t: time
    :param y: differential equations variables consisting of the state variables of the vehicles and the state variables
              related to the acceleration errors. y = [X, e]
    :param input_val: input acceleration data that we wish the vehicles follow
    :return: the differential equation related to the platoon movement
    -------------------------------------------------------------------
    args : tuple, optional
        Additional arguments to pass to the user-defined functions.  If given,
        the additional arguments are passed to all user-defined functions.
        In this case 'ode_platoon' has the signature ``ode_platoon(t, y, AA, BB, input_val)``,
        then `jac` (if given) and any event functions must have the same
        signature, and `args` must be a tuple of length 3.
    """
    dydt = np.dot(AA, y).reshape(-1, 1) + BB * input_val
    return dydt.reshape(-1)