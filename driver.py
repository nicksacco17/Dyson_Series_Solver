
import numpy as np
import numpy.matlib as mat

import time as time

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

import qutip as qutip

if __name__ == '__main__':

    print("TIME DEPENDENT SCHRODINGER EQUATION SOLVER MODULE")
    print("Approach I: N-th order matrix Trapezoidal method")
    print("Approach II: Dyson Series Approximation, developed for Quantum Computer")
    print("Approach III: QuTiP Verification")
    print("Approach IV: Parellization ")