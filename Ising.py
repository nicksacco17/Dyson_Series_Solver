
# Imports
import numpy as np
import qutip as qp
import time as time
from PauliInteraction import PauliInteraction

N = 7
NUM_INTERACTIONS = int((N * (N - 1)) / 2)

class Ising:

    def __init__(self, num_q, Jij, hi, direction):
         # System parameters
        self.num_qubits = num_q
        self.num_interactions = int(((self.num_qubits - 1) * (self.num_qubits)) / 2)

        # Couplings and coefficients
        self.pairwise_couplings = Jij.copy()
        self.individual_fields = hi.copy()

        self.m_direction = direction

        # Basic tensor network
        self.network = PauliInteraction(self.num_qubits)
        self.my_Hamiltonian = qp.Qobj()

        self.createHamiltonian()

        self.my_eigenvalues, self.my_eigenkets = self.my_Hamiltonian.eigenstates()

    def print_couplings(self):
        print(self.pairwise_couplings)

    def print_fields(self):
        print(self.individual_fields)

    def printHamiltonian(self):
        #np.set_printoptions(threshold = sys.maxsize)
        print(self.my_Hamiltonian)

    # Populate the Hamiltonian with the specified coefficients
    def createHamiltonian(self):

        if self.m_direction == "X":
            J_ij = 0
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    self.my_Hamiltonian += (self.pairwise_couplings[J_ij] * self.network.get_xtensor(i) * self.network.get_xtensor(j))
                    J_ij += 1
            for i in range(self.num_qubits):
                self.my_Hamiltonian += (self.individual_fields[i] * self.network.get_xtensor(i))

        elif self.m_direction == "Y":
            J_ij = 0
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    self.my_Hamiltonian += (self.pairwise_couplings[J_ij] * self.network.get_ytensor(i) * self.network.get_ytensor(j))
                    J_ij += 1
            for i in range(self.num_qubits):
                self.my_Hamiltonian += (self.individual_fields[i] * self.network.get_ytensor(i))

        elif self.m_direction == "Z":
            J_ij = 0
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    self.my_Hamiltonian += (self.pairwise_couplings[J_ij] * self.network.get_ztensor(i) * self.network.get_ztensor(j))
                    J_ij += 1
            for i in range(self.num_qubits):
                self.my_Hamiltonian += (self.individual_fields[i] * self.network.get_ztensor(i))

    def print_eigenvalue(self, index):
        print(self.my_eigenvalues[index])

    def print_eigenkets(self, index):
        print(self.my_eigenkets[index].data)
    
    def print_eigenvalues_ALL(self):
        
        for i in range(len(self.my_eigenvalues)):
            self.print_eigenvalue(i)

    def print_eigenkets_ALL(self):

        for i in range(len(self.my_eigenkets)):
            self.print_eigenkets(i)
            time.sleep(0.25)
   

def main():
    J = np.random.rand(NUM_INTERACTIONS)
    h = np.random.rand(N)

    IsingHamiltonian = Ising(N, J, h, "X")
    #IsingHamiltonian.print_couplings()
    #IsingHamiltonian.printHamiltonian()

    IsingHamiltonian.print_eigenvalues_ALL()

if __name__ == "__main__":
    main()
