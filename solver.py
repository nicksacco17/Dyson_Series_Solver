import numpy as np
import numpy.matlib as mat

import time as time

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

import qutip as qutip

class Solver:

    def __init__(self, simulation_time, time_step, Hamiltonian, dimension, init_state):

        # Direct Parameters
        
        self.simulation_time = simulation_time
        self.time_step = time_step
        
        self.H = Hamiltonian
        self.dim = dimension
        self.psi0 = init_state

        # Computed Parameters
        self.num_iterations = int(self.simulation_time / self.time_step)
        self.t = np.linspace(0, simulation_time, self.num_iterations)
        self.psi_t = np.ndarray(self.num_iterations, dtype = qutip.Qobj)
        self.norm_t = np.ndarray((self.dim, self.num_iterations), dtype = np.double)

    def param_print(self):

        print("DIMENSION = %d" % self.dim)
        print("SIMULATION TIME = %d" % self.simulation_time)
        print("TIME STEP = %e" % self.time_step)
        print("NUMBER OF ITERATIONS = %d" % self.num_iterations)
        print("NUMBER OF STATES IN SIMULATION = %d" % len(self.psi_t) )

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

    def get_ground_state_probability(self):
        
        gnd_state_prob = np.ndarray(self.num_iterations, dtype = np.double)

        for t in range(0, len(self.t)):
            gnd_state_prob[t] = np.abs((np.conj(self.psi_t[t][0]) * self.psi_t[t][0])[0][0]) ** 2
        return gnd_state_prob

    def print_ground_state(self):

        for t in range(0, len(self.t)):
            print(self.psi_t[t][0])

    def print_ground_state_mag(self):
        for t in range(0, len(self.t)):
            print((np.conj(self.psi_t[t][0]) * self.psi_t[t][0])[0][0])

    def plot_ground_state(self):

        self.plot_basis_state(0)

    def plot_basis_state(self, n):

        plt.plot(self.t, self.norm_t[n, :], markersize = 2.0, linewidth = 2.0, label = '|%d>' % n)
        plt.legend()
        plt.title("Qubit Dynamics: State Probabilities vs. Time")
        plt.xlabel("Time (sec)")
        plt.ylabel("Probability")
        plt.show()

    def calc_prob(self):
        for n in range(0, self.dim):

            for t in range(0, len(self.psi_t)):

                #print(np.conj(self.psi_t[t][n]) * self.psi_t[t][n])
                self.norm_t[n][t] = np.abs((np.conj(self.psi_t[t][n]) * self.psi_t[t][n])[0][0]) ** 2

    def plot(self):

        for n in range(0, self.dim):

            plt.plot(self.t, self.norm_t[n, :], markersize = 2.0, linewidth = 2.0, label = '|%d>' % n)

        plt.legend()
        plt.title("Qubit Dynamics: State Probabilities vs. Time")
        plt.xlabel("Time (sec)")
        plt.ylabel("Probability")
        plt.show()