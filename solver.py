import numpy as np
import numpy.matlib as mat

import time as time

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

import qutip as qutip

class Solver:

    def __init__(self, simulation_time, time_step, Hamiltonian, init_state):

        # Direct Parameters
        
        self.simulation_time = simulation_time
        self.time_step = time_step
        
        self.H = Hamiltonian
        self.dim = 3
        self.psi0 = init_state

        # Computed Parameters
        self.num_iterations = int(1.0 / self.time_step)
        self.t = np.linspace(0, simulation_time, self.num_iterations)
        self.psi_t = mat.zeros((self.dim, self.num_iterations), dtype = np.cdouble)

    def param_print(self):

        print("DIMENSION = %d" % self.dim)
        print("SIMULATION TIME = %d" % self.simulation_time)
        print("TIME STEP = %e" % self.time_step)
        print("NUMBER OF ITERATIONS = %d" % self.num_iterations)

    def normalization_check(self):

        psi_norm = np.zeros(self.num_iterations, dtype = float)

        for i in range(0, self.num_iterations):

            current_state = self.psi_t[:, i]

            for j in range(0, self.dim):

                psi_norm[i] += (current_state[j] * np.conj(current_state[j])).real

        error = 1.0 - psi_norm
        max_error = np.max(error)

        print("MAXIMUM NORMALIZATION ERROR = %e" % max_error)

        plt.plot(self.t, psi_norm, 'b-')
        plt.plot(self.t, error, 'g-')
        plt.ylim(0, 2)
        plt.show()

    def plot(self):

        for n in range(0, self.dim):

            current_state = np.zeros(self.num_iterations, dtype = np.double)
           
            for k in range(0, self.num_iterations):
                current_state[k] = (np.abs(self.psi_t[n, k]) ** 2)

            plt.plot(self.t, current_state, markersize = 2.0, linewidth = 2.0, label = '|%d>' % n)

        plt.legend()
        plt.title("Qubit Dynamics: State Probabilities vs. Time")
        plt.xlabel("Time (sec)")
        plt.ylabel("Probability")
        plt.show()