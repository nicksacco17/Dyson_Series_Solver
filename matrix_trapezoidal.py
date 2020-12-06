
import numpy as np
import numpy.matlib as mat

import time as time

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

import qutip as qutip
from solver import Solver

class CN_Solver(Solver):

    def __init__(self, simulation_time, time_step, Hamiltonian, dimension, order, init_state):

       super().__init__(simulation_time, time_step, Hamiltonian, dimension, init_state)
       self.order = order
       self.psi_t[0] = self.psi0
       
    def evolve(self):
        
        H_pow_k = qutip.Qobj(shape = (self.dim, self.dim))

        for i in range(1, self.num_iterations):

            if i % 100 == 0:
                print(i)
            backwards_term = qutip.Qobj(shape = (self.dim, self.dim))
            forwards_term = qutip.Qobj(shape = (self.dim, self.dim))

            for k in range(0, self.order + 1):

                factor_kb = (1.0 / np.math.factorial(k)) * ((1j * self.time_step / 2) ** k)
                factor_kf = (1.0 / np.math.factorial(k)) * ((-1j * self.time_step / 2) ** k)
                
                H_pow_k = self.H(self.t[i], args = None) ** k
            
                backwards_term += (factor_kb * H_pow_k)
                forwards_term += (factor_kf * H_pow_k)

            self.psi_t[i] = (backwards_term.inv() * forwards_term) * self.psi_t[i - 1]
        
    def run_simulation(self):

        self.evolve()
        #self.plot()

class TISE_Solver(Solver):

    def __init__(self, simulation_time, time_step, Hamiltonian, init_state):

        super().__init__(simulation_time, time_step, Hamiltonian, init_state)

    def evolve(self):
        
        t = np.linspace(0, self.simulation_time, self.num_iterations)

        for i in range(0, self.num_iterations):

            self.psi_t[:, i] = np.matmul(la.expm(-1j * t[i] * self.H), self.psi0)

def H_TD_FUNC(t):
    H_TI = mat.asmatrix([[1, 2, 0], [2, 0, 2], [0, 2, -1]])
    return (1.0 - t ** 2) * (H_TI)

def time_independent_method():
    time_step = (1e-3)
    I_DIM = sp.sparse.identity(3).toarray()

    num_iterations = 1000

    factor = 1j * time_step / 2

    H_TI = mat.asmatrix([[1, 2, 0], [2, 0, 2], [0, 2, -1]])

    U_tilde = mat.zeros((3, 3), dtype = np.cdouble)
    U_tilde[0, 0] = 1
    U_tilde[1, 1] = 1
    U_tilde[2, 2] = 1

    for i in range(0, num_iterations):

        #print(i)
        U_tilde = np.matmul(np.matmul(la.inv(I_DIM + factor * H_TI), (I_DIM - factor * H_TI)), U_tilde)

    print("----- TIME INDEPENDENT -----")
    print(U_tilde)



def exact():

    #t = np.linspace(0, 1, 1000)

    H_TI = mat.asmatrix([[1, 2, 0], [2, 0, 2], [0, 2, -1]])
    psi0 = mat.zeros((3, 1), dtype = np.cdouble)
    psi0[0, 0] = 1

    m_solver = TISE_Solver(simulation_time = 1, time_step = 1e-3, Hamiltonian = H_TI, init_state = psi0)

    m_solver.param_print()

    m_solver.evolve()

    m_solver.normalization_check()

    m_solver.plot()

def crank_nicolson():

    H_TI = mat.asmatrix([[1, 2, 0], [2, 0, 2], [0, 2, -1]])
    psi0 = mat.zeros((3, 1), dtype = np.cdouble)
    psi0[0, 0] = 1

    TIME_STEP = 1e-3
    for i in range(0, 2):

        TIME_STEP /= (10 ** i)
        m_solver = CN_Solver(simulation_time = 1, time_step = TIME_STEP, Hamiltonian = H_TD_FUNC, init_state = psi0)
        
        m_solver.param_print()

        start_time = time.time_ns()
        m_solver.evolve()
        stop_time = time.time_ns()

        print("EVOLUTION TIME = %0.8lf ms" % ((stop_time - start_time) / 1e6))

        m_solver.normalization_check()

        m_solver.plot()