###################################################################################################################
# Antonio Russo
# Imperial College London
# Date: 14/10/2028
###################################################################################################################
import sys, os, shutil
LibPath = os.environ['SPDE']
sys.path.append(LibPath)

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt,pi,log

#cython: infer_types=False
from cython.parallel import prange

from scipy.optimize import fsolve


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)

def free_energy_derivative_ideal_gas(rho_vec):
    return np.log(rho_vec)

cdef free_energy_derivative_ideal_gas ( np.ndarray[np.float64_t, ndim=1] rho_vec ):
    cdef int dim =len(rho_vec)
    cdef int iter=0
    cdef np.ndarray[np.float64_t, ndim=1] result
    
    for i in range (dim):
        result[i]= log(rho_vec[i])

    return result



    
    

    
    
    
