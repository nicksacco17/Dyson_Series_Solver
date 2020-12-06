
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
                time_variables = np.zeros(k, dtype = np.double)
                prefactor = ((-1j * self.T / self.r) ** k) / (self.M ** k * np.math.factorial(k))

                total_iterations = self.M ** k

                it = 0

                U_tilde = qutip.Qobj(shape = (self.dim, self.dim))
                while it < total_iterations:

                    if it % 10000 == 0:
                            print(it)
                    for j in range(0, k):

                        if j == 0:
                            time_variables[j] = time_segment[it % self.M]
                            #time_variables[j] = it % self.M
                        else:
                            #print("HERE")

                            #print(it / (self.M ** (j + 1)))
                            #print(int(np.floor(it / (self.M ** (j + 1)))))
                            #time_variables[j] = time_segment[int(it / (self.M ** j))]
                            time_variables[j] = time_segment[(int(it / (self.M ** j)) % self.M)]
                            #time_variables[j] = (int(it / (self.M ** j)) % self.M)
                            #time_variables[j] = 0
                    #print(time_variables)
                    #print(total_iterations)
                    ordered_time_var = np.sort(time_variables)[::-1]
                    #print("TIME VAR = " + str(time_variables))
                    #print("TIME ORDERED TIME VAR = " + str(ordered_time_var))

                    U_temp = I_DIM
                    for tk in range(0, k):

                        #print("H(%d) = %s" % (tk, str(self.H(ordered_time_var[tk]))))
                        U_temp = U_temp * self.H(ordered_time_var[tk], args = None)
                        #U_temp = np.matmul(U_temp, self.H(ordered_time_var[tk]))
                
                    U_tilde += U_temp
                    #print(U_tilde)
                    it += 1

                U += (prefactor * U_tilde)
            
            U += I_DIM
            print(U)
            if s == 0:
                self.psi_t[s] = U * self.psi0
            else:
                self.psi_t[s] = U * self.psi_t[s - 1]