
import numpy as np
import numpy.matlib as mat

import time as time

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

import qutip as qutip

from matrix_trapezoidal import CN_Solver

def cn_driver():

    H_TI = mat.asmatrix([[1, 2, 0], [2, 0, 2], [0, 2, -1]])
    psi0 = mat.zeros((3, 1), dtype = np.cdouble)
    psi0[0, 0] = 1

    H_TI_FUNC = lambda t : H_TI

    m_solver = CN_Solver(simulation_time = 2.0, time_step = 1e-4, Hamiltonian = H_TI_FUNC, dimension = 3, order = 1, init_state = psi0)

    m_solver.run_simulation()


if __name__ == '__main__':

    print("TIME DEPENDENT SCHRODINGER EQUATION SOLVER MODULE")
    print("Approach I: N-th order matrix Trapezoidal method")
    print("Approach II: Dyson Series Approximation, developed for Quantum Computer")
    print("Approach III: QuTiP Verification")
    print("Approach IV: Parellization ")

    cn_driver()