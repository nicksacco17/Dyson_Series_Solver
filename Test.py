
from PauliInteraction import PauliInteraction
from Ising import Ising
from CoefficientGenerator import CoefficientGenerator
from Evaluator import Evaluator

#from DataVisualizer import DataVisualizer
#from DataLogger import DataLogger

import Driver as Driver_H
import HardInstances as HardInst

from MasterEquation import MasterEquation
from QuantumAnnealer import QuantumAnnealer


import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
import time as time

X_HAT = "X"
Y_HAT = "Y"
Z_HAT = "Z"

N = 7
T = 100
NUM_INTERACTIONS = int((N * (N - 1)) / 2)
DISPLAY_FLAG = True

def test(gen_rand_coef, iteration, driver_flag):

#---------------------------- STEP 1: COEFFICIENTS -----------------------------


    print("ITERATION = %d\n" % iteration)
    start_time = time.time()

    if gen_rand_coef:

        ising_coefficients = CoefficientGenerator(NUM_INTERACTIONS, N)
        ising_coefficients.generateCoefficients_gaussian(0.0, 1.0)
        J = ising_coefficients.get_coupling_coefs()
        h = ising_coefficients.get_field_coefs()
    
    else:
        J = HardInst.J_ARRAY[iteration]
        h = HardInst.H_ARRAY[iteration]
    

#-------------------------------------------------------------------------------

#---------------------------- STEP 2: HAMILTONIANS -----------------------------
    
    basic_network = PauliInteraction(N)

    H_START = qp.Qobj()
    for i in range(N):
        H_START += basic_network.get_ztensor(i)
    #print(H_START)
    if driver_flag:
        H_DRIVE = Driver_H.create_resonator_driver(N, J, h)
    elif not driver_flag:

        #J = np.zeros(NUM_INTERACTIONS, dtype = int)
        #h = np.zeros(N, dtype = int)
        H_DRIVE = Ising(N, J, h, Z_HAT)
        #H_DRIVE.printHamiltonian()
        #H_DRIVE = qp.qzero(2)
        #for i in range(N - 1):
        #    H_DRIVE = qp.tensor(H_DRIVE, qp.qzero(2))
        #print(H_DRIVE)
    H_PROB = Ising(N, J, h, X_HAT)
    #H_PROB.printHamiltonian()


#-------------------------------------------------------------------------------

#---------------------------- STEP 3: BASE METRICS -----------------------------

    # CREATE THE GROUND STATE TO EVOLVE
    eigenvalues_H_start, eigenkets_H_start = H_START.eigenstates()
    eigenvalues_H_prob, eigenkets_H_prob = H_PROB.my_Hamiltonian.eigenstates()

    PSI_GND_i = eigenkets_H_start[0]
    print(PSI_GND_i)
    E_GND_i = eigenvalues_H_start[0]

    PSI_GND_f = eigenkets_H_prob[0]
    E_GND_f = eigenvalues_H_prob[0]

    #print("START: GROUND STATE ENERGY = %f" % E_GND_i)
    #print("STOP: GROUND STATE ENERGY = %f" % E_GND_f)

    #overlap = (qp.fidelity(PSI_GND_i, PSI_GND_f) ** 2)
    #print("OVERLAP = %f" % overlap)

#-------------------------------------------------------------------------------

#----------------------- STEP 4: MASTER EQUATION ROUTINE -----------------------
    
    qme_solver = MasterEquation(H_START, H_PROB.my_Hamiltonian, H_DRIVE.my_Hamiltonian, PSI_GND_i, T)
    #qme_solver = MasterEquation(H_START, H_PROB.my_Hamiltonian, H_DRIVE, PSI_GND_i, T)
    qme_solver.solve()

#-------------------------------------------------------------------------------

#--------------------- STEP 5: ADIABATIC EVOLUTION ROUTINE ---------------------

    computer = QuantumAnnealer(H_START, H_PROB.my_Hamiltonian, H_DRIVE.my_Hamiltonian, T)
    #computer = QuantumAnnealer(H_START, H_PROB.my_Hamiltonian, H_DRIVE, T)
    
    computer.evolve()
    #computer.print()

    #print(computer.get_gnd_eigvals())

#-------------------------------------------------------------------------------


#----------------------------- STEP 6: COMPARISON ------------------------------

    comparator = Evaluator(computer.current_ground_state_eigenkets, qme_solver.m_states_t, T, PSI_GND_f)
    comparator.evaluate(iteration)
   
#-------------------------------------------------------------------------------

#-------------------------------- STEP 7: LOGGER -------------------------------

    #logger = DataLogger(computer.current_ground_state_eigenkets, qme_solver.m_states, T, PSI_GND_f)
    #logger.solve()

#---------------------------- STEP 8: VISUALIZATION ----------------------------

    if DISPLAY_FLAG:
        comparator.display()

#-------------------------------------------------------------------------------
    stop_time = time.time()

    total_time = stop_time - start_time
    print("TOTAL TIME = %f sec\n" % total_time )

