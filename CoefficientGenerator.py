
# Imports
import numpy as np
import time as time

class CoefficientGenerator:

    def __init__(self, num_Jij, num_hi):

        self.m_num_Jij = num_Jij
        self.m_num_hi = num_hi

        self.m_coupling_coefs = np.ndarray(self.m_num_Jij, dtype = float)
        self.m_field_coefs = np.ndarray(self.m_num_hi, dtype = float)

    def generateCoefficients_gaussian(self, mean, standard_deviation):
        self.m_coupling_coefs = np.random.normal(loc = mean, scale = standard_deviation, size = self.m_num_Jij)
        self.m_fields_coefs = np.random.normal(loc = mean, scale = standard_deviation, size = self.m_num_hi)

    def generateCoefficients_uniform(self, lower_bound, upper_bound):
        self.m_coupling_coefs = np.random.uniform(low = lower_bound, high = upper_bound, size = self.m_num_Jij)
        self.m_fields_coefs = np.random.uniform(low = lower_bound, high = upper_bound, size = self.m_num_hi)

    def printCoefficients(self):
        print(self.m_coupling_coefs)
        print(self.m_field_coefs)

    def optimizeCoefficients(self):
        print("Beep boop doing some machine learning beep boop")

    def get_coupling_coefs(self):
        return self.m_coupling_coefs

    def get_field_coefs(self):
        return self.m_fields_coefs

def main():
    
    couplings = CoefficientGenerator(21, 7)
    couplings.generateCoefficients_gaussian(0.0, 1.0)
    couplings.printCoefficients()
    couplings.optimizeCoefficients()


if __name__ == "__main__":
    main()