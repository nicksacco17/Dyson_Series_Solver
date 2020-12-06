
from PauliInteraction import PauliInteraction
from Ising import Ising
from CoefficientGenerator import CoefficientGenerator
from Evaluator import Evaluator

#from DataVisualizer import DataVisualizer
#from DataLogger import DataLogger

import Driver as Driver_H
import HardInstances as HI

from MasterEquation import MasterEquation
from QuantumAnnealer import QuantumAnnealer
from Test import test

import qutip as qp
import numpy as np
import time as time

X_HAT = "X"
Y_HAT = "Y"
Z_HAT = "Z"

N = 7
T = 100
NUM_INTERACTIONS = int((N * (N - 1)) / 2)
RAND_COEF_FLAG = False
DRIVER_FLAG = False

RAND_INSTANCES = 100
# MAIN TEST CASES FROM SAMPLE CODE

def main():

    for i in range(0, RAND_INSTANCES):
        test(RAND_COEF_FLAG, i, DRIVER_FLAG)

    #test(RAND_COEF_FLAG, 0, DRIVER_FLAG)

if __name__ == "__main__":
    main()


