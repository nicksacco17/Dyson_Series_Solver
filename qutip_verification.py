
import numpy as np
import numpy.matlib as mat

import time as time

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

import qutip as qutip

from solver import Solver

def f(t, args):
    return (1.0 - t ** 2)

class ME_Solver(Solver):

    def __init__(self, simulation_time, time_step, Hamiltonian, init_state):

        super().__init__(simulation_time, time_step, Hamiltonian, init_state)

    def evolve(self):

        t = np.linspace(0, self.simulation_time, self.num_iterations)
    result = qutip.mesolve(H_TD, psi0, t, c_ops = None, e_ops = None, options = QME_OPTIONS, args = None)
    
    print(result.states[0][0])

    print(result.states[0].shape[0])



def time_independent_master_equation(H_TI, psi_init, start_time, stop_time, number_time_points):

    print("Time Independent Schrodinger Equation Verification")



def time_dependent_master_equation(H_TD):

    print("Time Dependent Schrodinger Equation Verification")

    QME_NUM_STEPS = 1e9
    #T_CB = {'T' : T, 'R' : R}
    QME_OPTIONS = qutip.Options(nsteps = QME_NUM_STEPS)

    H_TI_MAT = mat.asmatrix([[1, 2, 0], [2, 0, 2], [0, 2, -1]])

    H_TI = qutip.Qobj(H_TI_MAT)

    print(H_TI)

    H_TD = [[H_TI, f]]

    psi0_MAT = mat.zeros((3, 1), dtype = np.cdouble)
    psi0_MAT[0, 0] = 1

    psi0 = qutip.Qobj(psi0_MAT)
    print(psi0)

    t = np.linspace(0, 1, 100)
    result = qutip.mesolve(H_TD, psi0, t, c_ops = None, e_ops = None, options = QME_OPTIONS, args = None)
    
    print(result.states[0][0])

    print(result.states[0].shape[0])

    norms = np.zeros((3, 100), dtype = float)

    for n in range(0, 100):

        current_state = result.states[n]

        for k in range(0, 3):

            norms[k, n] = np.abs(current_state[k]) ** 2

        #norms[:, n] = np.abs(current_state) ** 2

    for n in range(0, 3):

        plt.plot(t, norms[n, :], markersize = 2.0, linewidth = 2.0, label = '|%d>' % n)

    plt.legend()
    plt.title("Qubit Dynamics: State Probabilities vs. Time")
    plt.xlabel("Time (sec)")
    plt.ylabel("Probability")
    plt.show()