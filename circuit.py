
import numpy as np
import numpy.matlib as mat

import time as time

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

import qutip as qutip
from solver import Solver

def spectral_norm(H):

    H_dagger = H.H
    eigenvalues = la.eigvals(np.matmul(H_dagger, H))

    max_eigval = np.max(eigenvalues)
    return np.sqrt(max_eigval)

class Circuit_Solver:

    def __init__(self, num_qubits, H, error, simulation_time):

        self.num_qubits = num_qubits
        self.Hamiltonian = H
        self.max_error = error
        self.simulatin_time = simulation_time
        self.dim = self.Hamiltonian.shape[0]

class Dyson_Series_Solver(Solver):

    def __init__(self, order, start_time, simulation_time, time_steps, time_segments, Hamiltonian, dimension, initial_state):

        self.K = order
        self.T = simulation_time
        self.t0 = start_time
        self.M = time_steps
        self.r = time_segments
        
        super().__init__(simulation_time, (1.0/self.M), Hamiltonian, dimension, initial_state)

        # Override the default Solver settings
        self.num_iterations = self.r * self.M

        #self.t = np.linspace(0, simulation_time, self.num_iterations)
        
        self.t = np.linspace(self.t0, self.simulation_time, self.r)
        self.psi_t = np.ndarray(self.r, dtype = qutip.Qobj)
        #self.psi_t = np.ndarray(self.num_iterations, dtype = qutip.Qobj)
        self.norm_t = np.ndarray((self.dim, self.r), dtype = np.double)

    def evolve(self):

        time_scale = (self.T / (self.r * self.M))
        I_DIM = qutip.qeye(self.dim)
        for s in range(0, self.r):

            print("CURRENT TIME SEGMENT = %d" % s)

            segment_start_time = self.t0 + (s * self.T / self.r)
            segment_stop_time = self.t0 + (s + 1) * (self.T / self.r)

            time_segment = np.linspace(segment_start_time, segment_stop_time, num = self.M)

            print("INTERVAL: [%lf --> %lf)" % (segment_start_time, segment_stop_time))
            print(time_segment)
            
            U = mat.zeros((self.dim, self.dim), dtype = np.cdouble)
            U = qutip.Qobj(shape = (self.dim, self.dim))

            for k in range(1, self.K + 1):

                print("ORDER k = %d" % k)
                time_variables = np.zeros(k, dtype = np.int)
                prefactor = ((-1j * self.T / self.r) ** k) / (self.M ** k * np.math.factorial(k))

                total_iterations = self.M ** k

                it = 0

                U_tilde = qutip.Qobj(shape = (self.dim, self.dim))
                while it < total_iterations:

                    if it % (100) == 0:
                        print("ITERATION = %d" % it)
                    for j in range(0, k):

                        if j == 0:
                            #time_variables[j] = time_segment[it % self.M]
                            time_variables[j] = it % self.M
                        else:
                            #print("HERE")
                            print(it / (self.M ** j))
                            #time_variables[j] = time_segment[int(it / (self.M ** j))]
                            time_variables[j] = it / (self.M ** j)
                    print(time_variables)
                    #ordered_time_var = np.sort(time_variables)[::-1]
                    #print("TIME VAR = " + str(time_variables))
                    #print("TIME ORDERED TIME VAR = " + str(ordered_time_var))

                    #U_temp = I_DIM
                    #for tk in range(0, k):

                        #print("H(%d) = %s" % (tk, str(self.H(ordered_time_var[tk]))))
                        #U_temp = U_temp * self.H(ordered_time_var[tk])
                        #U_temp = np.matmul(U_temp, self.H(ordered_time_var[tk]))
                
                    #U_tilde += U_temp
                    #print(U_tilde)
                    it += 1

                #U += (prefactor * U_tilde)
                #print(U)
            
            #U += I_DIM
            #self.psi_t[s] = U * self.psi0
        '''
        #print("FINAL U")
        #print(U)

def circuit_main():

    H_TI = mat.asmatrix([[1, 2, 0], [2, 0, 2], [0, 2, -1]])

    gamma = 0.5
 
    K = 3
    M = 64

    H0 = mat.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    H1 = mat.asmatrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    H2 = mat.asmatrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    H3 = mat.asmatrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    
    H4 = mat.asmatrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    H5 = mat.asmatrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    H6 = mat.asmatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    H7 = mat.asmatrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    H8 = mat.asmatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    H9 = mat.asmatrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    H10 = mat.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    H11 = mat.asmatrix([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])

    H_ARRAY = [H0, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11]

    H_SUM = mat.zeros((3, 3), dtype = np.cdouble)

    L = len(H_ARRAY)

    print(L)

    for i in H_ARRAY:

        H_SUM += i

    H_SUM *= gamma

    print(H_SUM)

    U_tilde = mat.zeros((3, 3), dtype = np.cdouble)

    T = 1

    r = 1

    k = 1

    gamma_pow_k = (gamma * T / r) ** k
    M_pow_k = M ** k
    minus_i_pow_k = (-1j) ** k

    for j in range(0, M):

        print(j)
        beta_j = gamma_pow_k / M_pow_k
        
        for l in range(0, L):

            V_j = minus_i_pow_k * H_ARRAY[l]
            U_tilde += beta_j * V_j

    print(U_tilde)

    k = 2

    gamma_pow_k = (gamma * T / r) ** k
    M_pow_k = M ** k
    minus_i_pow_k = (-1j) ** k

    for j1 in range(0, M):

        for j2 in range(0, M):

            if j1 <= j2:

                if j1 == j2:
                    repetitions = 2
                else:
                    repetitions = 1

                beta_j = gamma_pow_k / (M_pow_k * repetitions)

                for l1 in range(0, L):

                    for l2 in range(0, L):

                        V_j = minus_i_pow_k * np.matmul(H_ARRAY[l2], H_ARRAY[l1])

                        U_tilde += beta_j * V_j

    print(U_tilde)
    '''