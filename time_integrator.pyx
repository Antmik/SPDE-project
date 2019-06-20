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
from libc.math cimport sqrt,pi
from libc.math cimport isnan

#cython: infer_types=False
from cython.parallel import prange

from scipy.optimize import fsolve

###################################################################################################################
#TIME INTEGRATOR
###################################################################################################################
# The class time_integrator has the following methods:

#   __init__ takes as input:
#       rho= n vector
#       mu(rho)= function returing a n vector function with deterministic contributions
#       sigma(rho, eta)= function returing a n-by-n matrix function with stochastic contributions
#       eta = n vector with standard Gaussian random values
#       dt= timestep

#   solve takes as input:
#       time_method = integration algorithm ( EM, MI, RK)
#       theta= parameter between 0 and 1 determining the implicitness of the algorithm (0= full explicit, 1=full implicit)

###################################################################################################################

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)

cdef secant_method( object f, np.ndarray[np.float64_t, ndim=1] x0, np.ndarray[np.float64_t, ndim=1] x1, int max_iter=100, float max_tol=10**(-16) ):
    cdef int dim =len(x0)
    cdef int iter=0
    cdef float tol=10**(6)
    cdef np.ndarray[np.float64_t, ndim=1] x_temp
    
    while( iter<max_iter and tol>max_tol ):
        while(isnan( any( f(x1)) ) ):
            x1 = x1/2
            print('nan')
        
        x_temp = x1 - 0.1*(x1-x0) / ( f(x1)-f(x0) ) * f(x1)
        x0 = x1
        x1 = x_temp
        tol=max( f(x1)-f(x0) )
        iter += 1
        if ( isnan( any( f(x1)-f(x0) ) ) ):
            print('nan1')

    return x1



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)

cdef class time_integrator:

    cdef np.ndarray rho
    cdef object mu
    cdef object sigma
    cdef np.ndarray eta
    cdef float dt
    cdef object time_method
    cdef float theta
    
    cdef np.ndarray rho_new
    cdef np.ndarray s0
    cdef np.ndarray phi_value

    def __cinit__(self, object mu, object sigma):

        self.mu=mu
        self.sigma=sigma

    
    def __call__(self, np.ndarray[np.float64_t, ndim=1] rho , np.ndarray[np.float64_t, ndim=1] eta,  float dt , str time_method='EM', float theta=0):
        self.rho=rho
        self.theta=theta
        self.eta=eta
        self.dt=dt
        #Euler-Maruyama
        if (time_method == 'EM'):
            #explicit
            if (self.theta==0.0):
                rho_new= self.rho + self.mu(self.rho) * self.dt + sqrt(self.dt)* np.dot(self.sigma(self.rho,self.eta ),self.eta)
            
            #explicit-implicit case
            else:
                def f( np.ndarray[np.float64_t, ndim=1] x ):
                    return -x + self.rho + ( (1-theta)*self.mu(self.rho) + theta*self.mu(x) )* self.dt + sqrt(self.dt)* np.dot(self.sigma(self.rho,self.eta),self.eta)
                        #rho_new= secant_method( f, self.rho , self.rho +  0.5*self.mu(self.rho)* self.dt , max_iter=100, max_tol=10**(-16) )
                rho_new= fsolve(f, self.rho + 0.5*self.mu(self.rho)* self.dt  , maxfev= 1000)
        
        #Milstein algorithm
        elif (time_method == 'MI'):

            s0=self.sigma(self.rho,self.eta)
            sigma1= f_sigma1(self.rho, self.mu, self.sigma, self.eta, self.dt)

            #explicit
            if (self.theta==0.0):
                rho_new= self.rho + self.mu(self.rho) * self.dt + sqrt(self.dt)* np.dot(s0,self.eta) + 1/sqrt(self.dt)*sigma1
            else:
            #semi-implicit
                def f( np.ndarray[np.float64_t, ndim=1] x):
                    return -x + self.rho + ( (1-theta)*self.mu(self.rho) + theta*self.mu(x) )* self.dt + sqrt(self.dt)* np.dot(s0,self.eta) + 1/sqrt(self.dt)*sigma1
                rho_new= fsolve(f, + 0.5*self.mu(self.rho)* self.dt , maxfev= 1000)
                #rho_new= secant_method( f, self.rho , self.rho +  0.5*self.mu(self.rho)* self.dt , max_iter=10, max_tol=10**(-16) )

        #Runge-Kutta 2nd weak order algorithm
        elif (time_method == 'RK2'):

            phi_value=f_phi(self.rho, self.mu, self.sigma, self.eta, self.dt)
            #Predictor stage
            gamma= self.rho + self.mu(self.rho) * self.dt + sqrt(self.dt)* np.dot( self.sigma(self.rho,self.eta) ,self.eta)
            rho_pred= self.rho + 0.5*( self.mu(gamma) + self.mu(self.rho) )* self.dt + phi_value

            #Corrector stage
            rho_new=self.rho + 0.5* ( self.mu(rho_pred) + self.mu(self.rho) )* self.dt + phi_value

        #Runge-Kutta 3rd weak order algorithm for additive noise
        elif (time_method == 'RK3'):
            rho1= self.rho + self.mu(self.rho) * self.dt + sqrt(2*self.dt)* np.dot(self.sigma(self.rho,self.eta),self.eta)
            rho2= 3/4*self.rho + 1/4*rho1 + 1/4 * ( self.mu(rho1) * self.dt + sqrt(2*self.dt)* np.dot(self.sigma(rho1, self.eta),self.eta)  )
            rho_new= 1/3*self.rho + 2/3*rho2 + 2/3 * ( self.mu(rho2) * self.dt + sqrt(2*self.dt)* np.dot(self.sigma(rho2, self.eta),self.eta) )

        else:
            TypeError('Error in the parameter: time_method')


        return rho_new



#Auxiliary function for MI

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
#@cython.infer_types(True)
cdef object f_sigma1( np.ndarray[np.float64_t, ndim=1] rho, object mu, object sigma, np.ndarray[np.float64_t, ndim=1] eta, float dt):

    cdef Py_ssize_t i_l, i_m, i_r

    cdef int dim =len(rho)
    cdef double[:,:] s0=sigma(rho, eta)

    cdef int p=3#max(100,int(1/100/dt))
    cdef np.ndarray[np.float64_t, ndim=1 ,negative_indices=False, mode='c'] vec= 1/np.arange(1,p+1,1)

    cdef np.ndarray[np.float64_t, ndim=2 ,negative_indices=False, mode='c'] xi= np.random.randn(dim,p) #
    #cdef double[:,:] xi = np.random.randn(dim,p)
    #cdef np.ndarray[np.float64_t, ndim=1 ,negative_indices=False, mode='c'] phi0= np.random.randn(dim) #
    cdef double[:] phi = np.random.randn(dim)
    #cdef np.ndarray[np.float64_t, ndim=1, negative_indices=False, mode='c'] zeta0 = np.random.randn(dim) #
    cdef np.ndarray[np.float64_t, ndim=1 ,negative_indices=False, mode='c'] zeta = np.random.randn(dim)
    
    cdef float sqrt_kr = sqrt( 1/12- 1/2/pi**2 * np.sum(vec**2) )
    cdef float sqrt_2 = sqrt(2)
    cdef float inv_pi = 1/pi
    #cdef float el,em
    #cdef np.ndarray xil= np.empty([dim],dtype=np.float32)
    #cdef np.ndarray xim= np.empty([dim],dtype=np.float32)
    
#           Is=np.zeros((dim,dim),dtype=float64)
#            xi= np.sqrt(dt/2/np.pi/vec) * np.random.randn(dim,p) #
#            phi= np.sqrt(dt/2/np.pi/vec) * np.random.randn(dim,p) #
#            a = np.sum( xi/np.sqrt(np.pi *vec) , axis=1)
#            for i_l in range(dim) :
#                for i_m in range(i_l+1) :
#                    el=eta[i_l]
#                    em=eta[i_m]
#                    xil= xi[i_l,:]
#                    xim= xi[i_m,:]
#                    phil= phi[i_l,:]
#                    phim= phi[i_m,:]
#
#                    Is[i_l,i_m]= 0.5* (el*em*dt - np.sqrt(dt)*(a[i_m]*el - a[i_l]*em) ) + np.sum(xil*phim - xim*phil)
#                    Is[i_m,i_l]= 0.5* (em*el*dt - np.sqrt(dt)*(a[i_l]*em - a[i_m]*el) ) + np.sum(xim*phil - xil*phim)

    cdef np.ndarray[np.float64_t, ndim=1] sigma1=np.zeros([dim],dtype=np.float64)
    cdef double[:,:] gamma = np.repeat(rho[:, np.newaxis], dim, axis=1) + np.repeat(mu(rho)[:, np.newaxis], dim, axis=1)* dt + np.sqrt(dt)* s0

    cdef double[:,:] Is= 0.5*( np.diag(eta**2 -1)  ) #+ np.outer(eta,eta)
    cdef double[:] eta_view = eta

    if (dim!=1):
        for i_l in range(dim) :
            for i_m in range(i_l-1) :
                Is[i_l,i_m]=  ( 0.5*eta_view[i_l]*eta_view[i_m] + sqrt_kr*(zeta[i_l]*eta_view[i_m] - zeta[i_m]*eta_view[i_l]) + 0.5*inv_pi*np.sum(vec*(xi[i_l,:]*(sqrt_2*eta_view[i_m]+phi[i_m]) - xi[i_m,:]*(sqrt_2*eta_view[i_l]+phi[i_l]))) )
                Is[i_m,i_l]=  ( 0.5*eta_view[i_m]*eta_view[i_l] + sqrt_kr*(zeta[i_m]*eta_view[i_l] - zeta[i_l]*eta_view[i_m]) + 0.5*inv_pi*np.sum(vec*(xi[i_m,:]*(sqrt_2*eta_view[i_l]+phi[i_l]) - xi[i_l,:]*(sqrt_2*eta_view[i_m]+phi[i_m]))) )


#Is[i_l,i_m]= ( 0.5*eta_view[i_l]*eta_view[i_m] + sqrt_kr*(zeta[i_l]*eta_view[i_m] - zeta[i_m]*eta_view[i_l]) + 0.5*inv_pi*np.sum(vec*(xi[i_l,:]*(sqrt_2*eta_view[i_m]+phi[i_m]) - xi[i_m,:]*(sqrt_2*eta_view[i_l]+phi[i_l]))) )
#Is[i_m,i_l]=  ( 0.5*eta_view[i_m]*eta_view[i_l] + sqrt_kr*(zeta[i_m]*eta_view[i_l] - zeta[i_l]*eta_view[i_m]) + 0.5*inv_pi*np.sum(vec*(xi[i_m,:]*(sqrt_2*eta_view[i_l]+phi[i_l]) - xi[i_l,:]*(sqrt_2*eta_view[i_m]+phi[i_m]))) )


    #cdef double[:,:] Is= 0.5*( np.diag(eta**2 -1) + np.outer(eta,eta) ) + sqrt_kr*(np.outer(zeta,eta) - np.outer(eta,zeta) ) - 0.5*inv_pi* np.matmul(xi[:,:],vec) *(sqrt_2*eta[:]+phi[:])
    #for i_l in range(dim) :
        #Is[i_l,:]= 0.5*eta[i_l]*eta[:]+ sqrt_kr*(zeta[i_l]*eta[:] - zeta[:]*eta[i_l]) + 0.5*inv_pi* np.sum( np.outer(vec*xi[i_l,:],(sqrt_2*eta[:]+phi[:])),axis=0) - 0.5*inv_pi*np.matmul(xi[:,:],vec) *(sqrt_2*eta[i_l]+phi[i_l])
        #Is[i_l,:] += 0.5*inv_pi* np.sum( np.outer(vec*xi[i_l,:],(sqrt_2*eta[:]+phi[:])),axis=0)
        

    #Compute Milstein contribution
    for i_l in range(dim) :
        sigma1[:] +=  dt * np.dot( sigma(gamma[:,i_l],eta)-s0 , np.asarray(Is[i_l,:]) )

    return sigma1



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)
#Auxiliary function for RK
cdef object f_phi( np.ndarray[np.float64_t, ndim=1] rho, object mu, object sigma, np.ndarray[np.float64_t, ndim=1] eta, float dt):

    cdef int dim =len(rho)
    cdef Py_ssize_t i_j, i_r, i_m,i_n
    cdef double[:,:] s0 = sigma(rho, eta)
    cdef float eta_j
    cdef float sqrt_dt=sqrt(dt)
    cdef float s0_mj
    
    
    cdef double[:,:] U_plus= np.zeros([dim,dim],dtype=np.float64) #+ sqrt_dt* s0 + np.repeat(rho[:, np.newaxis], dim, axis=1)
    cdef double[:,:] U_minus= np.zeros([dim,dim],dtype=np.float64)#- sqrt_dt* s0 + np.repeat(rho[:, np.newaxis], dim, axis=1)
    cdef double[:,:] R_plus= np.zeros([dim,dim],dtype=np.float64) #U_plus+ np.repeat(mu(rho)[:, np.newaxis], dim, axis=1)* dt
    cdef double[:,:] R_minus= np.zeros([dim,dim],dtype=np.float64) #U_minus + np.repeat(mu(rho)[:, np.newaxis], dim, axis=1)* dt

    cdef double[:] mu_value = mu(rho)
    
    for i_j in prange(dim,nogil=True):
        for i_m in range(dim):
            s0_mj= s0[i_j,i_m]
            U_plus[i_j,i_m] = rho[i_j] + sqrt_dt* s0_mj
            U_minus[i_j,i_m] = rho[i_j] - sqrt_dt* s0_mj
            R_plus[i_j,i_m] = U_plus[i_j,i_m] + mu_value[i_j]* dt
            R_minus[i_j,i_m] = U_minus[i_j,i_m] + mu_value[i_j]* dt

    #Auxiliary random value
    cdef long[:,:] Vrand=  np.tril( 2*(np.random.randint(2, size=(dim, dim))-0.5), -1).astype(np.int)
    Vrand -= np.transpose(Vrand) - np.diag(np.ones(dim,dtype=int))


    cdef double[:,:,:] bUrPlus =  sigma(U_plus,eta)
    cdef double[:,:,:] bUrMinus=  sigma(U_minus,eta)
    cdef double[:,:,:] bRPlus =  sigma(R_plus,eta)
    cdef double[:,:,:] bRMinus =  sigma(R_minus,eta)


    #Auxiliary function phi
    cdef double[:] phi= np.zeros(dim,dtype=np.float64)


#Not too expensive loop
    for i_j in prange(dim,nogil=True):
        eta_j=eta[i_j]
        for i_m in range(dim):
            s0_mj= 2*s0[i_m,i_j]
            
            phi[i_m] += ( ( bRPlus[i_m,i_j,i_j] + bRMinus[i_m,i_j,i_j] + s0_mj ) * eta_j + ( bRPlus[i_m,i_j,i_j] - bRMinus[i_m,i_j,i_j] ) * ( eta_j*eta_j -1) )* sqrt_dt  - ( ( bUrPlus[i_m,i_j,i_j] + bUrMinus[i_m,i_j,i_j] - s0_mj ) * eta_j + ( bUrPlus[i_m,i_j,i_j] - bUrMinus[i_m,i_j,i_j] ) * ( eta_j*eta_j + Vrand[i_j,i_j])* sqrt_dt )
            for i_n in range(dim):
                phi[i_m] +=   ( bUrPlus[i_m,i_j,i_n] + bUrMinus[i_m,i_j,i_n] - s0_mj ) * eta_j + ( bUrPlus[i_m,i_j,i_n] - bUrMinus[i_m,i_j,i_n] )* ( eta_j*eta[i_n] + Vrand[i_n,i_j])* sqrt_dt


#   for i_j in prange(dim,nogil=True):
#       eta_j=eta[i_j]
#       phi +=  ( bRPlus[:,i_j,i_j] + bRMinus[:,i_j,i_j] + 2 * s0[:,i_j] ) * eta_j*sqrt_dt + ( bRPlus[:,i_j,i_j] - bRMinus[:,i_j,i_j] ) * ( eta_j**2 -1) * sqrt_dt +            np.sum( ( bUrPlus[:,i_j,:] + bUrMinus[:,i_j,:] - 2* s0_extended[:,i_j,:] ) * eta_j , axis=1)*sqrt_dt + np.dot( ( bUrPlus[:,i_j,:] - bUrMinus[:,i_j,:] ) , ( eta_j*eta[:] + Vrand[:,i_j]) ) * sqrt_dt - (( bUrPlus[:,i_j,i_j] + bUrMinus[:,i_j,i_j] - 2* s0[:,i_j] ) * eta_j +( bUrPlus[:,i_j,i_j] - bUrMinus[:,i_j,i_j] ) * ( eta_j*eta_j + Vrand[i_j,i_j]) * sqrt_dt )
#                for i_r in range(dim) :
#                    phi[:] +=  ( bUrPlus[:,i_j,i_r] + bUrMinus[:,i_j,i_r] - 2* s0[:,i_j] ) * self.eta[i_j] + \
#                        ( bUrPlus[:,i_j,i_r] - bUrMinus[:,i_j,i_r] ) * ( self.eta[i_j]*self.eta[i_r] + Vrand[i_r,i_j]) *np.sqrt(self.dt)
#       phi -=  ( bUrPlus[:,i_j,i_j] + bUrMinus[:,i_j,i_j] - 2* s0[:,i_j] ) * eta[i_j] +( bUrPlus[:,i_j,i_j] - bUrMinus[:,i_j,i_j] ) * ( eta[i_j]*eta[i_j] + Vrand[i_j,i_j]) *np.sqrt(dt)

    return np.asarray(phi)/4


    
    

    
    
    
