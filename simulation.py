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
from Ising import Ising
from PauliInteraction import PauliInteraction

import HardInstances as HI
import Callbacks as cb

N = 7

def create_hermitian_matrix(N):

    np.random.seed(1)
    mat_H = np.zeros((N, N), dtype = np.cdouble)

    for i in range(0, N):

        for j in range(i, N):

            rand_real = np.random.uniform(low = -1, high = 1)
            if i == j:
                mat_H[i, j] = rand_real
            else:
                rand_imag = np.random.uniform(low = -1, high = 1) * 1j
                mat_H[i, j] = (rand_real + rand_imag)
                mat_H[j, i] = (rand_real - rand_imag)
    
    hermitian_mat = qutip.Qobj(mat_H)

    return hermitian_mat

def ising_test():

    basic_network = PauliInteraction(N)

    H_START = qutip.Qobj()
    for i in range(N):
        H_START += basic_network.get_ztensor(i)

    H_ISING = Ising(N, HI.J0, HI.h0, "X")
    H_DRIVE = Ising(N, HI.J0, HI.h0, "Z")

    eigenvalues_H_start, eigenkets_H_start = H_START.eigenstates()
    eigenvalues_H_ising, eigenkets_H_ising = H_ISING.my_Hamiltonian.eigenstates()

    PSI_GND_i = eigenkets_H_start[0]
    E_GND_i = eigenvalues_H_start[0]

    PSI_GND_f = eigenkets_H_ising[0]
    E_GND_f = eigenvalues_H_ising[0]

    #print(PSI_GND_i)
    #print(PSI_GND_f)

    #H_ISING.printHamiltonian()
    #H_DRIVE.printHamiltonian()
    #print(H_START)

    #time_dependent_H_2 = [
    #        [H_START, cb.start_H_time_coeff_cb],
    #        [H_DRIVE.my_Hamiltonian, cb.driver_H_time_coeff_cb],
    #        [H_ISING.my_Hamiltonian, cb.stop_H_time_coeff_cb]
    #    ]

    #t = np.linspace(0, 100, 100)
    #plt.plot(t, cb.start_H_time_coeff_cb(t, 'T'))
    #plt.plot(t, 1.0 - t/100)
    #plt.show()

    H_DRIVE_FORMATTED = qutip.Qobj(H_DRIVE.my_Hamiltonian.data.toarray())
    H_ISING_FORMATTED = qutip.Qobj(H_ISING.my_Hamiltonian.data.toarray())
    H_START_FORMATTED = qutip.Qobj(H_START.data.toarray())

    #time_dependent_H = lambda t, args: (1.0 - t/100) * H_START_FORMATTED + (t/100 * (1.0 - t/100)) * H_DRIVE_FORMATTED + t * H_ISING_FORMATTED
    time_dependent_H = lambda t, args: (1.0 - t) * H_START_FORMATTED + (t * (1.0 - t)) * H_DRIVE_FORMATTED + t * H_ISING_FORMATTED

    return time_dependent_H, PSI_GND_i, PSI_GND_f, 
    '''
        def evolve(self):

        #self.current_ground_state_eigenvalues, self.current_ground_state_eigenkets = qp.parfor(self.par_evolve_cb, range(0, self.annealing_time + 1))

        #print(self.current_ground_state_eigenvalues)

        # Need to go [0, T] inclusive - continuous time evolution not discrete
        
        for t in range(self.annealing_time + 1):
          
            # Evolving correctly now
            self.current_H = (cb.start_H_time_coeff_cb(t, T_CB) * self.start_H) + \
                            (cb.driver_H_time_coeff_cb(t, T_CB) * self.driver_H) + \
                            (cb.stop_H_time_coeff_cb(t, T_CB) * self.stop_H)
            self.measure(t)
        
        
        #print(self.current_ground_state_eigenvalues)

            def evaluate(self, index):
  
        #self.overlap_aqc_qme, self.overlap_aqc_psi, self.overlap_qme_psi = qp.parfor(self.par_eval, range(0, self.m_routine_time))

        csv_file = CSV_FILE_PATH + "\\instance_driver" + str(0) + ".csv"
        for i in range(self.m_routine_time):
            self.overlap_aqc_qme[i] = (qp.fidelity(self.m_quantum_annealer_gnd_states[i], self.m_master_equation_gnd_states[i]) ** 2)
            self.overlap_aqc_psi[i] = (qp.fidelity(self.m_quantum_annealer_gnd_states[i], self.m_expected_state) ** 2)
            self.overlap_qme_psi[i] = (qp.fidelity(self.m_master_equation_gnd_states[i], self.m_expected_state) ** 2)
        #fmt = "Iteration, %d, %f", 
        #np.savetxt(csv_file, self.overlap_aqc_qme.T, header = ("TEST" + str(index)), delimiter = ",")
        
        wtr = csv.writer(open (csv_file, 'a'), delimiter=',', lineterminator=',')
        for x in self.overlap_aqc_qme : wtr.writerow ([x])
        wtr.writerow(['\n'])
        
        print("OVERLAP: %lf" % self.overlap_aqc_qme[self.m_routine_time - 1])
                    
    def measure(self, time_index):
        
        temp_eigval, temp_eigket = self.current_H.eigenstates()
        self.current_eigenvalues[time_index] = temp_eigval
        self.current_ground_state_eigenvalues[time_index] = temp_eigval[0]

        self.current_eigenkets[time_index] = temp_eigket
        self.current_ground_state_eigenkets[time_index] = temp_eigket[0]
    '''


def me_solver():

    ising_hamiltonian, initial_state, compare_state = ising_test()

    H_TI = qutip.Qobj([[1, 2, 0], [2, 0, 2], [0, 2, -1]])

    H_TI_FUNC = lambda t, args : H_TI
    H_TD_FUNC = lambda t, args : H_TI * (1.0 - t ** 2)

    #psi0 = qutip.Qobj([[1], [0], [0]])
    psi0 = qutip.Qobj(initial_state.data.toarray())

    #print(psi0)

    m_solver = ME_Solver(simulation_time = 1, time_step = 1e-2, Hamiltonian = ising_hamiltonian, dimension = 128, init_state = psi0)
    
    data_array = compare_state.data.toarray()

    new_compare_state = qutip.Qobj(data_array)
    #print(new_compare_state)
    m_solver.evolve()

    fidelity_array = []

    print(m_solver.psi_t[-1])

    for i in range(0, len(m_solver.psi_t)):

        overlap = qutip.fidelity(m_solver.psi_t[i], new_compare_state) ** 2
        
        fidelity_array.append(overlap)
    #(qp.fidelity(self.m_master_equation_gnd_states[i], self.m_expected_state) ** 2)

    #print(fidelity_array)
    plt.plot(m_solver.t, fidelity_array)
    plt.show()


    #print(compare_state)

    #print(m_solver.psi_t[0])

    #print(compare_state.conj() * m_solver.psi_t[0])

    #m_solver.param_print()
    #print(m_solver.t)
    #m_solver.evolve()
    #m_solver.print_ground_state()
    #m_solver.print_ground_state_mag()
    #m_solver.plot_ground_state()

    #fidelity_array = []
   

def cn_driver():

    H_TI = qutip.Qobj([[1, 2, 0], [2, 0, 2], [0, 2, -1]])
    psi0 = qutip.Qobj([[1], [0], [0]])

    H_TI_FUNC = lambda t : H_TI

    ising_hamiltonian, initial_state, compare_state = ising_test()
    psi0 = qutip.Qobj(initial_state.data.toarray())

    data_array = compare_state.data.toarray()

    new_compare_state = qutip.Qobj(data_array)

    m_solver = CN_Solver(simulation_time = 1, time_step = 1e-2, Hamiltonian = ising_hamiltonian, dimension = 128, order = 1, init_state = psi0)
    m_solver.run_simulation()

    fidelity_array = []

    for i in range(0, len(m_solver.psi_t)):

        overlap = qutip.fidelity(m_solver.psi_t[i], new_compare_state) ** 2
        
        fidelity_array.append(overlap)
    #(qp.fidelity(self.m_master_equation_gnd_states[i], self.m_expected_state) ** 2)

    #print(fidelity_array)
    plt.plot(m_solver.t, fidelity_array)
    plt.show()

def ds_driver():

    H_TI = qutip.Qobj([[1, 2, 0], [2, 0, 2], [0, 2, -1]])
    psi0 = qutip.Qobj([[1], [0], [0]])

    H_TI_FUNC = lambda t : H_TI

    ising_hamiltonian, initial_state, compare_state = ising_test()
    psi0 = qutip.Qobj(initial_state.data.toarray())

    data_array = compare_state.data.toarray()

    new_compare_state = qutip.Qobj(data_array)

    m_solver = Dyson_Series_Solver(order = 3, start_time = 0, simulation_time = 1, time_steps = 10, time_segments = 100, Hamiltonian = ising_hamiltonian, dimension = 128, initial_state = psi0)

    m_solver.evolve()

    fidelity_array = []

    #print(m_solver.psi_t[-1])

    for i in range(0, len(m_solver.psi_t)):

        overlap = qutip.fidelity(m_solver.psi_t[i], new_compare_state) ** 2
        
        fidelity_array.append(overlap)
    #(qp.fidelity(self.m_master_equation_gnd_states[i], self.m_expected_state) ** 2)

    #print(fidelity_array)
    plt.plot(m_solver.t, fidelity_array)
    plt.show()

    #solver.param_print()

    

    #solver.plot()

def me_step_test():

    simulation_time = []
    STEP_SIZE = np.logspace(-6, -1, 6)

    NUM_QUBITS = 5
    H_size = int(2 ** NUM_QUBITS)
    dense_hermitian_matrix = create_hermitian_matrix(H_size)

    H_TD_FUNC = lambda t, args : dense_hermitian_matrix * (1.0 - t ** 2)
    psi0 = qutip.basis(H_size)

    for h in STEP_SIZE:

        print("STEP SIZE = %e" % h)
        m_solver = ME_Solver(simulation_time = 1.0, time_step = h, Hamiltonian = H_TD_FUNC, dimension = H_size, init_state = psi0)
        
        start_time = time.time()
        m_solver.evolve()
        stop_time = time.time()

        total_time = stop_time - start_time
        
        print("SIMULATION TIME = %lf sec" % total_time)
        simulation_time.append(total_time)

    np.savetxt("me_step_size_test.csv", np.asarray(simulation_time), delimiter = ",")

def me_size_test():

    simulation_time = []
    NUM_QUBITS = 10
    
    for N in range(1, NUM_QUBITS + 1):

        H_size = int(2 ** N)
        
        dense_hermitian_matrix = create_hermitian_matrix(H_size)
        H_TD_FUNC = lambda t, args : dense_hermitian_matrix * (1.0 - t ** 2)
        psi0 = qutip.basis(H_size)

        m_solver = ME_Solver(simulation_time = 1.0, time_step = 1e-4, Hamiltonian = H_TD_FUNC, dimension = H_size, init_state = psi0)
        start_time = time.time()
        m_solver.evolve()
        stop_time = time.time()

        total_time = stop_time - start_time

        simulation_time.append(total_time)
    
    np.savetxt("me_size_test.csv", np.asarray(simulation_time), delimiter = ",")

def plot_from_data(FILE_NAME, TITLE, X_LABEL, Y_LABEL):

    simulation_time = np.genfromtxt(FILE_NAME, delimiter = ', ')
    N_array = np.linspace(1, len(simulation_time), len(simulation_time))
    plt.plot(N_array, simulation_time, 'r*-', linewidth = 1.5, markersize = 1.5)

    plt.title(TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.show()


if __name__ == '__main__':

    print("TIME DEPENDENT SCHRODINGER EQUATION SOLVER MODULE")
    print("Approach I: N-th order matrix Trapezoidal method")
    print("Approach II: Dyson Series Approximation, developed for Quantum Computer")
    print("Approach III: QuTiP Verification")
    print("Approach IV: Parellization ")

    #cn_driver()
    #me_solver()
    #ds_driver()
    #me_size_test()
    me_step_test()

    #plot_from_data(FILE_NAME = 'me_size_test.csv', TITLE = "Master Equation: Computation Time vs. Hamiltonian Size", X_LABEL = "Number of Qubits", Y_LABEL = "Computation Time (sec)")

  