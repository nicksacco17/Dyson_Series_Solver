
import numpy as np
import numpy.matlib as mat

import time as time

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

import qutip as qutip

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

class Dyson_Series_Solver:

    def __init__(self, order, simulation_time, time_segments, time_steps, Hamiltonian):

        self.K = order
        self.T = simulation_time
        self.r = time_segments
        self.M = time_steps
        self.H = Hamiltonian

        #self.psi0 = init_state
        self.dim = 3

        self.num_iterations = self.r * self.M
        self.psi_t = mat.zeros((self.dim, self.r), dtype = np.cdouble)

    def evolve(self):

        U = mat.zeros((3, 3), dtype = np.cdouble)

        for k in range(1, self.K + 1):

            print("k = %d" % k)
            time_variables = np.zeros(k, dtype = np.int)
            prefactor = ((-1j * self.T / self.r) ** k) / (self.M ** k * np.math.factorial(k))

            total_iterations = self.M ** k

            print(prefactor)
            it = 0

            U_tilde = mat.zeros((3, 3), dtype = np.cdouble)
            while it < total_iterations:

                if it % (total_iterations / 100) == 0:
                    print("ITERATION = %d" % it)
                for tk in range(0, k):

                    if tk == 0:
                        time_variables[tk] = (it % self.M)
                    else:
                        time_variables[tk] = (it / (self.M ** tk))
                #print(time_variables)
                ordered_time_var = np.sort(time_variables)[::-1]
                #print("TIME VAR = " + str(time_variables))
                #print("TIME ORDERED TIME VAR = " + str(ordered_time_var))

                U_temp = sp.sparse.identity(3).toarray()
                for tk in range(0, k):

                    #print("H(%d) = %s" % (tk, str(self.H(ordered_time_var[tk]))))
                    U_temp = np.matmul(U_temp, self.H(ordered_time_var[tk]))
            
                U_tilde += U_temp
                #print(U_tilde)
                it += 1

            U += (prefactor * U_tilde)
            print(U)
        I_DIM = sp.sparse.identity(3).toarray()
        U += I_DIM

        print("FINAL U")
        print(U)

'''
            while not compare_arrays(time_variables[0 : k], [self.M] * k):

                for tk in range(0, k):

                    print(time_variables[0 : k])

            for tk in range(0, k):

                iterations = 0

                print(time_variables[0 : k])
                print([self.M] * k)

                print(compare_arrays(time_variables[0 : k], [self.M] * k))
                
                while time_variables[tk] != (self.M):
                    print(time_variables[0 : k])
                    time_variables[tk] += 1
                    iterations += 1
'''
            #    for jk in range(0, k):

            #        

            #        time_variables[jk] += 1

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

def ds_main():

    H_TI = mat.asmatrix([[1, 2, 0], [2, 0, 2], [0, 2, -1]])

    solver = Dyson_Series_Solver(order = 6, simulation_time = 1, time_segments = 4, time_steps = 10, Hamiltonian = H_TI_func)

    solver.evolve()

'''
    for k in range(0, K + 1):

        M_pow_k = M ** k
        minus_i_pow_k = (-1j) ** k
        gamma_pow_k = (gamma * T / r) ** k

        for l1 in range(0, L):

            for l2 in range(0, L):

                for l3 in range(0, L):

                    print("j = (k, (l1, l2, l3)) = (%d, (%d, %d, %d))" % (k, l1, l2, l3))
                    for j1 in range(0, M):

                        for j2 in range(j1, M):

                            for j3 in range(j2, M):

                                if time_ordering(j1, j2, j3):

                                    counter_array = np.asarray([j1, j2, j3])
                                    num_occurrences = np.bincount(counter_array)

                                    repetitions = np.extract(num_occurrences > 0, num_occurrences)

                                    repetition_factor = 1
                                    for i in repetitions:
                                        repetition_factor *= np.math.factorial(i)

                                    #print(repetition_factor)
                                    beta_j = (gamma_pow_k) / (M_pow_k * repetition_factor)

                                    temp = np.matmul(H_ARRAY[l2], H_ARRAY[l1])

                                    V_j = minus_i_pow_k * np.matmul(H_ARRAY[l3], temp)

                                    U_tilde += (beta_j * V_j)
                                    #print(U_tilde)
                                    #print("j = (k, (l1, l2, l3), (j1, j2, j3)) = (%d, (%d, %d, %d), (%d, %d, %d))" % (k, l1, l2, l3, j1, j2, j3))
            print(U_tilde)         