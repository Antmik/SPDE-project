################################################################################################################
# Antonio Russo
# Imperial College London
# Date: 19/10/2028
################################################################################################################
import sys, os, shutil
LibPath = os.environ['SPDE']
sys.path.append(LibPath)

cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange

from time_integrator  import time_integrator



################################################################################################################
#SOLVER BROWNIAN MOTION
###############################################################################################################
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)

cdef class geometric_brownian2D:

    cdef float value_mu
    cdef float value_sigma
    cdef Py_ssize_t i

    def __cinit__(self, float value_mu, float value_sigma):
        self.value_mu= value_mu
        self.value_sigma= value_sigma

    def __call__(self, np.ndarray[np.float64_t, ndim=1] rho, np.ndarray[np.float64_t, ndim=3] eta, float dt, int n_steps, str time_method='EM', float theta=0, n_traj=1):

        dim=rho.shape[0]
        
        if (dim!=2):
            raise("Max variable lenght for this solver is 2")
        
        mu=space_discretizator_deterministic(self.value_mu)
        sigma=space_discretizator_stochastic(self.value_sigma)

        my_time_integrator=time_integrator( mu, sigma)

        rho_output= np.zeros([dim,n_steps,n_traj],dtype=np.float64)
        for i_traj in range (n_traj):
            rho_output[:,0,i_traj]= my_time_integrator(rho, eta[:,0,i_traj], dt, time_method ,theta)
            for i in range (1,n_steps):
                rho_output[:,i,i_traj]= my_time_integrator(rho_output[:,i-1,i_traj], eta[:,i,i_traj], dt, time_method ,theta)

        return rho_output



################################################################################################################
#SPACE DISCRETIZATORS BROWNIAN MOTION
###############################################################################################################

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)
cdef double[:,::1] compute_1D_sigma( double[:] rho, float value):
    cdef int dim=rho.shape[0]
    cdef Py_ssize_t i_l
    cdef double[:,::1] sigma_value= np.zeros([dim,dim],dtype=np.float64)
    for i_l in range(dim):
        sigma_value[i_l,i_l]= value*rho[i_l]
    return sigma_value



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)
cdef double[:,:,::1] compute_2D_sigma( double[:,::1] rho, float value):
    cdef int dim=rho.shape[0]
    cdef Py_ssize_t i_l, i_m
    cdef double[:,:,::1] sigma_value_extended= np.zeros([dim,dim,dim],dtype=np.float64)
    for i_m in range(dim):
        for i_l in range(dim):
            sigma_value_extended[i_l,i_l,i_m]= value*rho[i_l,i_m]
    return sigma_value_extended





@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)
cdef class space_discretizator_deterministic:
    cdef float value
    def __cinit__(self, float value):
        self.value=value
    
    def __call__(self, double[:] rho):
        mat= np.array([ [ -self.value, self.value ],[ self.value, -self.value] ])
        return np.dot(mat,rho)



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef class space_discretizator_stochastic:
    cdef float value
    cdef double[:,::1] sigma_value
    cdef double[:,:,::1] sigma_value_extended
    
    def __cinit__(self, float value):
        self.value= value

    def __call__(self, rho,  eta ):
        if ( rho.ndim==1):
            sigma_value = compute_1D_sigma(rho, self.value)
            return np.asarray(sigma_value)
        else:
            sigma_value_extended = compute_2D_sigma(rho, self.value)
            return np.asarray(sigma_value_extended)




    
    
