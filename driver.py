
import numpy as np
import numpy.matlib as mat

import time as time

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

import qutip as qutip

from matrix_trapezoidal import CN_Solver
from qutip_verification import ME_Solver
from circuit import Dyson_Series_Solver

def f(t, args):
    return (1.0 - t ** 2)

def H_TI_FUNC(t, args):
    H_TI_MAT = mat.asmatrix([[1, 2, 0], [2, 0, 2], [0, 2, -1]])
    H_TI = qutip.Qobj(H_TI_MAT)
    return H_TI

def me_solver():

    H_TI = qutip.Qobj([[1, 2, 0], [2, 0, 2], [0, 2, -1]])

    H_TI_FUNC = lambda t, args : H_TI
    H_TD_FUNC = lambda t, args : H_TI * (1.0 - t ** 2)

    psi0 = qutip.Qobj([[1], [0], [0]])

    m_solver = ME_Solver(simulation_time = 1.0, time_step = 1e-2, Hamiltonian = H_TI_FUNC, dimension = 3, init_state = psi0)
    
    #m_solver.param_print()
    #print(m_solver.t)
    #m_solver.evolve()
    #m_solver.plot()

def cn_driver():

    H_TI = qutip.Qobj([[1, 2, 0], [2, 0, 2], [0, 2, -1]])
    psi0 = qutip.Qobj([[1], [0], [0]])

    H_TI_FUNC = lambda t : H_TI

    m_solver = CN_Solver(simulation_time = 1, time_step = 1e-4, Hamiltonian = H_TI_FUNC, dimension = 3, order = 1, init_state = psi0)
    m_solver.run_simulation()

def ds_driver():

    H_TI = qutip.Qobj([[1, 2, 0], [2, 0, 2], [0, 2, -1]])
    psi0 = qutip.Qobj([[1], [0], [0]])

    H_TI_FUNC = lambda t : H_TI

    solver = Dyson_Series_Solver(order = 2, start_time = 0, simulation_time = 1, time_steps = 10, time_segments = 4, Hamiltonian = H_TI_FUNC, dimension = 3, initial_state = psi0)

    solver.param_print()

    solver.evolve()

    #solver.plot()

if __name__ == '__main__':

    print("TIME DEPENDENT SCHRODINGER EQUATION SOLVER MODULE")
    print("Approach I: N-th order matrix Trapezoidal method")
    print("Approach II: Dyson Series Approximation, developed for Quantum Computer")
    print("Approach III: QuTiP Verification")
    print("Approach IV: Parellization ")

    #cn_driver()
    #me_solver()
    ds_driver()