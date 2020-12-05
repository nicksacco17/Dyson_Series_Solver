
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

    def __init__(self, simulation_time, time_step, Hamiltonian, dimension, init_state):

        super().__init__(simulation_time, time_step, Hamiltonian, dimension, init_state)

    def evolve(self):

        QME_NUM_STEPS = 1e9
        #T_CB = {'T' : T, 'R' : R}
        QME_OPTIONS = qutip.Options(nsteps = QME_NUM_STEPS)

        result = qutip.mesolve(self.H, self.psi0, self.t, c_ops = None, e_ops = None, options = QME_OPTIONS, args = None)

        for t in range(0, len(result.states)):

            self.psi_t[t] = result.states[t]


def time_independent_master_equation(H_TI, psi_init, start_time, stop_time, number_time_points):

    print("Time Independent Schrodinger Equation Verification")

def time_dependent_master_equation(H_TD):

    print("Time Dependent Schrodinger Equation Verification")

    