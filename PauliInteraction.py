
# Imports
import numpy as np
import qutip as qp

#---------------------------------- CONSTANTS ----------------------------------

N = 7                               # Number of qubits
DEBUG = 0                           # Debugging flag

def debug_print(print_str):
    if (DEBUG):
        print(print_str)

# Pauli Interaction Tensor Network for N qubits
# In a multi-qubit system, the the interaction for a given qubit is described as tensor products
# between the current qubit in the applied direction (i.e. a Pauli Matrix) and (N-1) identities.
# This results in a sparse, complex-valued matrix of size (2^N x 2^N).

class PauliInteraction:

#-------------------------------------------------------------------------------
    # Initialize Pauli Interaction Tensor Network for num_q qubits
    # STATUS - WT
    def __init__(self, num_q):
        self.num_qubits = num_q
        self.num_interactions = int((self.num_qubits * (self.num_qubits - 1)) / 2)
        self.size = 2 ** self.num_qubits
        # Tensor network is length-N array of size 2^N x 2^N quantum objects
        self.x_tensor = np.ndarray(self.num_qubits, dtype = qp.Qobj)
        self.y_tensor = np.ndarray(self.num_qubits, dtype = qp.Qobj)
        self.z_tensor = np.ndarray(self.num_qubits, dtype = qp.Qobj)
       
        self.create_xtensor_network()
        self.create_ytensor_network()
        self.create_ztensor_network()

        self.validate_tensors()
#-------------------------------------------------------------------------------
    # Create the x-tensor network for N qubits
    # STATUS - WT
    def create_xtensor_network(self):
        
        debug_print('<----- CREATING X-TENSOR ----->')
        for i in range(len(self.x_tensor)):
        
            if i == 0:
                current_product = qp.sigmax()
            else:
                current_product = qp.qeye(2)

            for j in range(1, self.num_qubits):
                
                if i != 0 and i == j:
                    current_product = qp.tensor(current_product, qp.sigmax())
                else:
                    current_product = qp.tensor(current_product, qp.qeye(2))
            
            self.x_tensor[i] = current_product
        debug_print('<----- X-TENSOR INITIALIZED ----->')

    # Create the y-tensor network for N qubits
    # STATUS - WT
    def create_ytensor_network(self):

        debug_print('<----- CREATING Y-TENSOR ----->')
        for i in range(len(self.y_tensor)):
        
            if i == 0:
                current_product = qp.sigmay()
            else:
                current_product = qp.qeye(2)

            for j in range(1, self.num_qubits):
                
                if i != 0 and i == j:
                    current_product = qp.tensor(current_product, qp.sigmay())
                else:
                    current_product = qp.tensor(current_product, qp.qeye(2))
            
            self.y_tensor[i] = current_product
        debug_print('<----- Y-TENSOR INITIALIZED ----->')

    # Create the z-tensor network for N qubits
    # STATUS - WT
    def create_ztensor_network(self):  

        debug_print('<----- CREATING Z-TENSOR ----->')
        for i in range(len(self.z_tensor)):
        
            if i == 0:
                current_product = qp.sigmaz()
            else:
                current_product = qp.qeye(2)

            for j in range(1, self.num_qubits):
                
                if i != 0 and i == j:
                    current_product = qp.tensor(current_product, qp.sigmaz())
                else:
                    current_product = qp.tensor(current_product, qp.qeye(2))
            
            self.z_tensor[i] = current_product
        debug_print('<----- Z-TENSOR INITIALIZED ----->') 
#-------------------------------------------------------------------------------
    def get_xtensor(self, index):
        return self.x_tensor[index]

    def get_ytensor(self, index):
        return self.y_tensor[index]

    def get_ztensor(self, index):
        return self.z_tensor[index]

#-------------------------------------------------------------------------------
    # Validate the x-tensor network for the special case of 7 qubits
    # STATUS - WT
    def validate_xtensor(self):

        if self.num_qubits == 7:

            sx0 = qp.tensor(qp.sigmax(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))
            sx1 = qp.tensor(qp.qeye(2), qp.sigmax(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))
            sx2 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.sigmax(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))            
            sx3 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmax(), qp.qeye(2), qp.qeye(2), qp.qeye(2))
            sx4 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmax(), qp.qeye(2), qp.qeye(2))
            sx5 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmax(), qp.qeye(2))
            sx6 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmax())
    
            assert sx0 == self.x_tensor[0], "[ERROR] sx0"
            assert sx1 == self.x_tensor[1], "[ERROR] sx1"
            assert sx2 == self.x_tensor[2], "[ERROR] sx2"
            assert sx3 == self.x_tensor[3], "[ERROR] sx3"
            assert sx4 == self.x_tensor[4], "[ERROR] sx4"
            assert sx5 == self.x_tensor[5], "[ERROR] sx5"
            assert sx6 == self.x_tensor[6], "[ERROR] sx6"

        debug_print("\tX-TENSOR VALIDATION COMPLETE")
        
    # Validate the x-tensor network for the special case of 7 qubits
    # STATUS - WT
    def validate_ytensor(self):
        
        if self.num_qubits == 7:

            sy0 = qp.tensor(qp.sigmay(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))
            sy1 = qp.tensor(qp.qeye(2), qp.sigmay(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))
            sy2 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.sigmay(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))            
            sy3 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmay(), qp.qeye(2), qp.qeye(2), qp.qeye(2))
            sy4 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmay(), qp.qeye(2), qp.qeye(2))
            sy5 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmay(), qp.qeye(2))
            sy6 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmay())
    
            assert sy0 == self.y_tensor[0], "[ERROR] sy0"
            assert sy1 == self.y_tensor[1], "[ERROR] sy1"
            assert sy2 == self.y_tensor[2], "[ERROR] sy2"
            assert sy3 == self.y_tensor[3], "[ERROR] sy3"
            assert sy4 == self.y_tensor[4], "[ERROR] sy4"
            assert sy5 == self.y_tensor[5], "[ERROR] sy5"
            assert sy6 == self.y_tensor[6], "[ERROR] sy6"

        debug_print("\tY-TENSOR VALIDATION COMPLETE")
    
    # Validate the x-tensor network for the special case of 7 qubits
    # STATUS - W
    def validate_ztensor(self):
        
        if self.num_qubits == 7:

            sz0 = qp.tensor(qp.sigmaz(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))
            sz1 = qp.tensor(qp.qeye(2), qp.sigmaz(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))
            sz2 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.sigmaz(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))            
            sz3 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmaz(), qp.qeye(2), qp.qeye(2), qp.qeye(2))
            sz4 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmaz(), qp.qeye(2), qp.qeye(2))
            sz5 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmaz(), qp.qeye(2))
            sz6 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmaz())
    
            assert sz0 == self.z_tensor[0], "[ERROR] sz0"
            assert sz1 == self.z_tensor[1], "[ERROR] sz1"
            assert sz2 == self.z_tensor[2], "[ERROR] sz2"
            assert sz3 == self.z_tensor[3], "[ERROR] sz3"
            assert sz4 == self.z_tensor[4], "[ERROR] sz4"
            assert sz5 == self.z_tensor[5], "[ERROR] sz5"
            assert sz6 == self.z_tensor[6], "[ERROR] sz6"

        debug_print("\tZ-TENSOR VALIDATION COMPLETE")

    # Validate all tensor networks
    # STATUS - WT
    def validate_tensors(self):

        debug_print('<----- BEGIN VALIDATION ----->')
        self.validate_xtensor()
        self.validate_ytensor()
        self.validate_ztensor()
        debug_print('<----- VALIDATION COMPLETE ----->')
#-------------------------------------------------------------------------------      
        

def main():
    network = PauliInteraction(N)
    print(network.validate_tensors())

#def debug_config():
 #   np.set_printoptions(threshold = sys.maxsize)

if __name__ == "__main__":
    #debug_config()
    main()
    
