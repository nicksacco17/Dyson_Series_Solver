
from math import sqrt
from math import exp

T = 1
R = False           # Resonator Flag

T_CB = {'T' : T, 'R' : R}

# CALLBACK FOR EVALUATING TIME-DEPENDENT HAMILTONIAN USING MASTER EQUATION
# STATUS - WT
def start_H_time_coeff_cb(t, T_CB):
    return (1.0 - (t / T))

# CALLBACK FOR EVALUATING TIME-DEPENDENT HAMILTONIAN USING MASTER EQUATION
# STATUS - WT
def stop_H_time_coeff_cb(t, T_CB):
    return (t / T)

# CALLBACK FOR EVALUATING TIME-DEPENDENT HAMILTONIAN USING MASTER EQUATION
# STATUS - WT
def driver_H_time_coeff_cb(t, T_CB):
    
    if R:
        return resonator_H_time_coef_cb(t, T_CB)
    else:
        return ((t / T) * (1.0 - (t / T)))
    
def resonator_H_time_coef_cb(t, T_CB):
    
    alpha0 = 2
    alpha0_sq = alpha0 ** 2

    a = (t/T) ** 2 + 1e-8
    a_sq = a ** 2

    p = (sqrt(1 - exp(-alpha0_sq * a_sq)) / sqrt(1 + exp(-alpha0_sq * a_sq)))
    factor = ((p-1/p) ** 2) / ((p+1/p) ** 2)

    return (t/T) * factor
   
   