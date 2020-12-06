
from Ising import Ising
from QuantumAnnealer import QuantumAnnealer
from MasterEquation import MasterEquation

import qutip as qp
import numpy as np
import csv as csv
import matplotlib.pyplot as plt

#CSV_FILE_PATH = "C:\\Users\\nicks\\Documents\\UML\\UML Graduate\\Thesis\\QuantumAnnealer\\Quantum_Information"
CSV_FILE_PATH = "D:\\comp_phys\\Quantum_Information\\output"

class Evaluator:

    def __init__(self, quantum_annealer_gnd_states, master_equation_gnd_states, routine_time, expected_state):

        self.m_quantum_annealer_gnd_states = quantum_annealer_gnd_states
        self.m_master_equation_gnd_states = master_equation_gnd_states
        self.m_routine_time = routine_time
        self.m_expected_state = expected_state

        self.overlap_aqc_qme = np.ndarray(self.m_routine_time, dtype = float)
        self.overlap_aqc_psi = np.ndarray(self.m_routine_time, dtype = float)
        self.overlap_qme_psi = np.ndarray(self.m_routine_time, dtype = float)
        self.time_array = np.linspace(0, self.m_routine_time, self.m_routine_time + 1, dtype = int)


    def par_eval(self, i):
        overlap_aqc_qme = (qp.fidelity(self.m_quantum_annealer_gnd_states[i], self.m_master_equation_gnd_states[i]) ** 2)    
        overlap_aqc_psi = (qp.fidelity(self.m_quantum_annealer_gnd_states[i], self.m_expected_state) ** 2)
        overlap_qme_psi = (qp.fidelity(self.m_master_equation_gnd_states[i], self.m_expected_state) ** 2)

        return overlap_aqc_qme, overlap_aqc_psi, overlap_qme_psi


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

    def display(self):
        #t = np.linspace(0, self.m_routine_time, self.m_routine_time)
        #t_norm = t / self.m_routine_time
        t_norm = np.linspace(0, 1, 100)
        plt.plot(t_norm, self.overlap_qme_psi, 'b-', markersize = 1.0, label = 'OVERLAP_QME_PSI')
        plt.show()
    '''
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.axis([0, 1.0, 0, 1.0])
        plt.title('Overlap between QME and AQC')
        plt.xlabel('Index (t)')
        plt.ylabel('Overlap')
        plt.plot(t_norm, self.overlap_aqc_qme, 'r-', markersize = 1.0, label = 'OVERLAP_AQC_QME')

        plt.subplot(1, 3, 2)
        plt.axis([0, 1.0, 0, 1.0])
        plt.title('Overlap between AQC and PSI')
        plt.xlabel('Index (t)')
        plt.ylabel('Overlap')
        plt.plot(t_norm, self.overlap_aqc_psi, 'g-', markersize = 1.0, label = 'OVERLAP_AQC_PSI')

        plt.subplot(1, 3, 3)
        plt.axis([0, 1.0, 0, 1.0])
        plt.title('Overlap between QME and PSI')
        plt.xlabel('Index (t)')
        plt.ylabel('Overlap')
        plt.plot(t_norm, self.overlap_qme_psi, 'b-', markersize = 1.0, label = 'OVERLAP_QME_PSI')
        print(self.overlap_qme_psi)

        plt.show()
    '''
        

    
        