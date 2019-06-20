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
import sympy as sp
from sympy import I
from scipy import signal
import math, cmath

from cython.parallel import prange

from libc.math cimport sqrt,pi, isnan,isinf

######################################################################################

cdef np.ndarray time_integrator( np.ndarray[np.float64_t, ndim=1] rho , np.ndarray[np.float64_t, ndim=1] mu, np.ndarray[np.float64_t, ndim=1] sigma, np.ndarray[np.float64_t, ndim=1] eta,  float dt ):
    
    cdef np.ndarray[np.float64_t, ndim=1] rho_new
    rho_new= rho + mu * dt + sqrt(dt)* np.multiply(sigma,eta)
    
    return rho_new

######################################################################################

def gamma(nb_neurons,c,lam,omega):
    num=0
    for i_neurons in range(nb_neurons):
        partial_num=-sp.I*c[i_neurons]*(omega+sp.I*lam[i_neurons])+sp.I*c[i_neurons]*(omega-sp.I*lam[i_neurons])
        for j_neurons in range(nb_neurons):
            if( j_neurons!= i_neurons):
                partial_num*=(omega-sp.I*lam[j_neurons])*(omega+sp.I*lam[j_neurons])
        num+=partial_num
    A_coeff=sp.Poly.coeffs(sp.Poly(num,omega))

    beta_coeff=[]
    roots=sp.solvers.solve(num, omega)
    beta_coeff=[x for x in roots if sp.im(x)<0]
    return A_coeff[0], beta_coeff

######################################################################################
def g_build(nb_neurons,A,beta,lam,omega):
    num=np.sqrt(float(A)/2)
    for i_neurons in range(len(beta)):
        num *= (omega-beta[i_neurons])
    
    den=1
    for i_neurons in range(nb_neurons):
        den *= (omega-1.j*lam[i_neurons])
    
    #with simpy
    a=sp.Poly.coeffs(sp.Poly((1.j**(len(beta)-nb_neurons))*num,omega))
    b=sp.Poly.coeffs(sp.Poly(den,omega))
    a= [complex(item) for item in a]
    b= [complex(item) for item in b]
    [r,p,k]=signal.residue(a,b,tol=0.001)
    lam1=np.imag(p)
    
    def locate_min(a):
        smallest = min(a)
        return [index for index, element in enumerate(a) if smallest == element]
    
    lam1_loc=np.zeros((len(lam1),1))
    for i_lam1 in range (len(lam1)):
        lam1_loc[i_lam1]= locate_min(abs(lam-lam1[i_lam1]))
    
    #print(lam1_loc,-np.imag(r))
    
    lam1_loc, b_coeff = zip(*sorted(zip(lam1_loc, -np.imag(r))))
    #print(lam1_loc,b_coeff)
    
    return b_coeff

######################################################################################

def noiseCoeff(nb_neurons=2,c=[1.0,1.0],lam=[1.0,1.0]):
        
    omega=sp.symbols('omega')
    [A,beta_coeff]=gamma(nb_neurons,c,lam,omega)
    
    b_coeff=g_build(nb_neurons,A,beta_coeff,lam,omega)
    #print(b_coeff)
    
    return b_coeff



################################################################################################################
#SOLVER Non_Markovian_FH
###############################################################################################################
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)

cdef class non_markovian_FH:

    cdef np.ndarray mu_rho
    cdef np.ndarray mu_momentum
    cdef np.ndarray mu_S
    cdef np.ndarray sigma_S
    
    cdef np.ndarray x
    cdef str boundaries
    cdef float open_boundary_value
    cdef Py_ssize_t i
    cdef object energy_derivative
    cdef np.ndarray A_vec
    cdef np.ndarray B_vec
    cdef np.ndarray b_vec
    cdef int brownian_bridge_calls
    cdef np.ndarray S
    
    def __cinit__(self, str boundaries, np.ndarray[np.float64_t, ndim=1] x , object energy_derivative , np.ndarray[np.float64_t, ndim=1] A_vec, np.ndarray[np.float64_t, ndim=1] B_vec, float open_boundary_value =0.0):
        
        if ( len(A_vec)!=len(B_vec)):
            raise TypeError('Error in the parameters: A_k and B_k')
        else:
            self.A_vec= A_vec
            self.B_vec= B_vec
            self.b_vec= np.zeros(len(A_vec))#  sqrt( np.multiply( A_vec, -B_vec ))
            
            #find b_k vector
            if ( np.all(self.B_vec) != 0 and np.all(self.A_vec) != 0):
                self.b_vec=np.array( noiseCoeff(len(A_vec),A_vec,-B_vec) )
                print(self.A_vec, self.B_vec, self.b_vec)
        
        
        if (boundaries!='periodic' and boundaries!='noflux' and boundaries!='open' ):
            raise TypeError('Error in the parameter: boundaries')
        else:
            self.boundaries=boundaries
            self.open_boundary_value=open_boundary_value
            print('Boundaries= %s' %self.boundaries)

        self.energy_derivative= energy_derivative
        self.x= x

    def __call__(self, np.ndarray[np.float64_t, ndim=1] rho, np.ndarray[np.float64_t, ndim=1] momentum, np.ndarray[np.float64_t, ndim=3] eta, float dt, int n_steps, n_traj=1):
        

        dim=rho.shape[0]
        n_S= len(self.A_vec)
        rho_output= np.zeros([dim,n_steps,n_traj],dtype=np.float64)
        momentum_output= np.zeros([dim,n_steps,n_traj],dtype=np.float64)
        S_output=np.zeros([dim,n_S,n_steps,n_traj],dtype=np.float64)
        
        #Initialize S
        S=np.zeros([dim,n_S],dtype=np.float64)
        for i_S in range (n_S):
            if (self.B_vec[i_S] != 0):
                S[:,i_S]= self.b_vec[i_S] / np.sqrt( -self.B_vec[i_S]) *np.random.randn(dim)
        
        S_tmp=np.zeros([dim,n_S],dtype=np.float64)
        
        brownian_bridge_calls=0
        for i_traj in range (n_traj):
            #First timestep
            
            #Predictor
            mu_rho= compute_mu_rho1( self.x, momentum, self.boundaries)
            rho_tmp= time_integrator(rho, mu_rho, np.zeros(dim), np.zeros(dim), dt)

            mu_momentum= compute_mu_momentum1(self.x, rho, momentum, np.sum(S, axis=1) , self.energy_derivative, self.boundaries)
            momentum_tmp= time_integrator(momentum, mu_momentum, np.zeros(dim) , np.zeros(dim), dt)

            for i_S in range (n_S):
                mu_S = compute_mu_S(self.x, S[:, i_S], momentum, self.A_vec[i_S], self.B_vec[i_S], self.boundaries)
                sigma_S = compute_sigma_S(self.x, rho, self.b_vec[i_S], self.boundaries)
                S_tmp[:,i_S]= time_integrator(S[:,i_S], mu_S, sigma_S, eta[:,0,i_traj], dt)


            #Corrector
            mu_rho= compute_mu_rho2( self.x, momentum_tmp, self.boundaries)
            rho_output[:,0,i_traj]= 0.5*(rho + time_integrator(rho_tmp, mu_rho, np.zeros(dim), np.zeros(dim), dt) )
            
            mu_momentum= compute_mu_momentum2(self.x, rho_tmp, momentum_tmp, np.sum(S_tmp, axis=1) , self.energy_derivative, self.boundaries)
            momentum_output[:,0,i_traj]= 0.5*(momentum +  time_integrator(momentum_tmp, mu_momentum, np.zeros(dim) , np.zeros(dim), dt) )
            
            for i_S in range (n_S):
                mu_S = compute_mu_S(self.x, S_tmp[:, i_S], momentum_tmp, self.A_vec[i_S], self.B_vec[i_S], self.boundaries)
                sigma_S = compute_sigma_S(self.x, rho_tmp, self.b_vec[i_S], self.boundaries)
                S_output[:,i_S,0,i_traj]= 0.5*(S[:,i_S] + time_integrator(S_tmp[:,i_S], mu_S, sigma_S, eta[:,0,i_traj], dt) )
            
            if ( any ( [ value < 0 or isnan(value) or isinf(value) for value in rho_output[:,0,i_traj]] ) ):
                brownian_bridge_calls +=1
                rho_output[:,0,i_traj],  momentum_output[:,0,i_traj], S_output[:,:,0,i_traj] =brownian_bridge(rho, momentum, S, eta[:,0,i_traj], dt, self.x , self.energy_derivative , self.A_vec, self.B_vec , self.b_vec , self.boundaries)



            for i in range (1,n_steps):
                #Predictor
                mu_rho= compute_mu_rho1( self.x, momentum_output[:,i-1,i_traj], self.boundaries)
                rho_tmp= time_integrator(rho_output[:,i-1,i_traj], mu_rho, np.zeros(dim), np.zeros(dim), dt)
                
                mu_momentum= compute_mu_momentum1(self.x, rho_output[:,i-1,i_traj], momentum_output[:,i-1,i_traj], np.sum(S_output[:,:,i-1,i_traj], axis=1) , self.energy_derivative, self.boundaries)
                momentum_tmp= time_integrator(momentum_output[:,i-1,i_traj], mu_momentum, np.zeros(dim) , np.zeros(dim), dt)
                
                for i_S in range (n_S):
                    mu_S = compute_mu_S(self.x, S_output[:, i_S, i-1, i_traj], momentum_output[:,i-1,i_traj], self.A_vec[i_S], self.B_vec[i_S], self.boundaries)
                    sigma_S = compute_sigma_S(self.x, rho_output[:,i-1,i_traj], self.b_vec[i_S], self.boundaries)
                    S_tmp[:,i_S]= time_integrator(S_output[:,i_S,i-1,i_traj], mu_S, sigma_S, eta[:,i,i_traj], dt)

                
                #Corrector
                mu_rho= compute_mu_rho2( self.x, momentum_tmp, self.boundaries)
                rho_output[:,i,i_traj]= 0.5*(rho_output[:,i-1,i_traj] + time_integrator(rho_tmp, mu_rho, np.zeros(dim), np.zeros(dim), dt) )
                
                mu_momentum= compute_mu_momentum2(self.x, rho_tmp, momentum_tmp, np.sum(S_tmp, axis=1) , self.energy_derivative, self.boundaries)
                momentum_output[:,i,i_traj]= 0.5*(momentum_output[:,i-1,i_traj] +  time_integrator(momentum_tmp, mu_momentum, np.zeros(dim) , np.zeros(dim), dt) )

                for i_S in range (n_S):
                    mu_S = compute_mu_S(self.x, S_tmp[:, i_S], momentum_tmp, self.A_vec[i_S], self.B_vec[i_S], self.boundaries)
                    sigma_S = compute_sigma_S(self.x, rho_tmp, self.b_vec[i_S], self.boundaries)
                    S_output[:,i_S,i,i_traj]= 0.5*(S_output[:,i_S,i-1,i_traj] + time_integrator(S_tmp[:,i_S], mu_S, sigma_S, eta[:,i,i_traj], dt) )

                if ( any ( [ value < 0 or isnan(value) or isinf(value) for value in rho_output[:,i,i_traj]] ) ):
                    brownian_bridge_calls +=1
                    rho_output[:,i,i_traj], momentum_output[:,i,i_traj], S_output[:,:,i,i_traj] =brownian_bridge(rho_output[:,i-1,i_traj], momentum_output[:,i-1,i_traj], S_output[:,:,i-1,i_traj], eta[:,i,i_traj], dt, self.x , self.energy_derivative , self.A_vec, self.B_vec, self.b_vec, self.boundaries)
                        
        print("Brownian bridge calls: %d" %brownian_bridge_calls)
                            
        return np.asarray(rho_output), np.asarray( momentum_output), np.asarray( S_output)



################################################################################################################
# BROWNIAN BRIDGE FOR ADAPTIVE TIME STEP
###############################################################################################################
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
#@cython.infer_types(True)

cdef noise_partition(int dim, np.ndarray[np.float64_t, ndim=2] eta_short, int dt_partition):
    cdef Py_ssize_t i_dt
    cdef np.ndarray[np.float64_t, ndim=2] eta_large=np.zeros((dim,2**dt_partition))
    cdef np.ndarray[np.float64_t, ndim=1] rnd_n
    for i_dt in range(2**(dt_partition-1)):
        rnd_n=np.sqrt(2)*np.random.randn(dim)
        eta_large[:,2*i_dt]=0.5*eta_short[:,i_dt]+0.5*rnd_n
        eta_large[:,2*i_dt+1]=0.5*eta_short[:,i_dt]-0.5*rnd_n
    return eta_large


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
#@cython.infer_types(True)
cdef brownian_bridge( np.ndarray[np.float64_t, ndim=1] rho_start, np.ndarray[np.float64_t, ndim=1] mom_start,np.ndarray[np.float64_t, ndim=2] S_start, np.ndarray[np.float64_t, ndim=1] eta_org, float dt, np.ndarray[np.float64_t, ndim=1] x , object energy_derivative , np.ndarray[np.float64_t, ndim=1] A_vec, np.ndarray[np.float64_t, ndim=1] B_vec, np.ndarray[np.float64_t, ndim=1] b_vec , str boundaries):
    
    cdef int dim=rho_start.shape[0]
    cdef int n_S= len(A_vec)
    cdef Py_ssize_t i_dt
    cdef float dt_try=dt
    cdef np.ndarray rho_try
    cdef np.ndarray momentum_try
    cdef np.ndarray S_try
    cdef np.ndarray[np.float64_t, ndim=2] eta_extended = eta_org.reshape(dim,1)
    cdef int dt_partition=0
    cdef str flag_cfl_false=True
    
    cdef np.ndarray[np.float64_t, ndim=2] S_tmp=np.zeros([dim,n_S],dtype=np.float64)
    
    cdef np.ndarray[np.float64_t, ndim=1] mu_rho
    cdef np.ndarray[np.float64_t, ndim=1] mu_momentum
    cdef np.ndarray[np.float64_t, ndim=1] mu_S
    cdef np.ndarray[np.float64_t, ndim=1] sigma_S
    
    
    while (flag_cfl_false):
        print('BB')
        flag_cfl_false = False
        dt_partition +=1
        dt_try = dt_try/2
        rho_try= rho_start
        
        momentum_try= mom_start
        
        S_try= S_start
        
        eta_extended= noise_partition(dim, eta_extended, dt_partition)

        for i_dt in range(2**dt_partition):
            
            #Predictor
            mu_rho= compute_mu_rho1( x, momentum_try, boundaries)
            rho_tmp= time_integrator(rho_try, mu_rho, np.zeros(dim), np.zeros(dim), dt_try)
            
            mu_momentum= compute_mu_momentum1(x, rho_try, momentum_try, np.sum(S_try, axis=1) , energy_derivative, boundaries)
            momentum_tmp= time_integrator(momentum_try, mu_momentum, np.zeros(dim) , np.zeros(dim), dt_try)
                
            for i_S in range (n_S):
                mu_S = compute_mu_S(x, S_try[:, i_S], momentum_try, A_vec[i_S], B_vec[i_S], boundaries)
                sigma_S = compute_sigma_S(x, rho_try, b_vec[i_S], boundaries)
                S_tmp[:,i_S]= time_integrator(S_try[:,i_S], mu_S, sigma_S, eta_extended[:,i_dt], dt_try)
        
        
            #Corrector
            mu_rho= compute_mu_rho2( x, momentum_tmp, boundaries)
            rho_try= 0.5*(rho_try + time_integrator(rho_tmp, mu_rho, np.zeros(dim), np.zeros(dim), dt_try) )
            
            mu_momentum= compute_mu_momentum2( x, rho_tmp, momentum_tmp, np.sum(S_tmp, axis=1) , energy_derivative, boundaries)
            momentum_try= 0.5*(momentum_try +  time_integrator(momentum_tmp, mu_momentum, np.zeros(dim) , np.zeros(dim), dt_try) )
            
            for i_S in range (n_S):
                mu_S = compute_mu_S(x, S_tmp[:, i_S], momentum_tmp, A_vec[i_S], B_vec[i_S], boundaries)
                sigma_S = compute_sigma_S(x, rho_tmp, b_vec[i_S], boundaries)
                S_try[:,i_S]= 0.5*(S_try[:,i_S] + time_integrator(S_tmp[:,i_S], mu_S, sigma_S, eta_extended[:,i_dt], dt_try) )
            
            if ( any ( [ value < 0 or isnan(value) or isinf(value) for value in rho_try ] )):
                flag_cfl_false = True
                break

    return rho_try, momentum_try, S_try



################################################################################################################
#SPACE DISCRETIZATOR FDDFT
###############################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)

cdef np.ndarray compute_mu_rho1(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] momentum, str boundaries):
    cdef int dim=x.shape[0]
    cdef Py_ssize_t i_l
    cdef double[:] mu_value= np.zeros([dim],dtype=np.float64)
    cdef float dx = x[1]-x[0]
    cdef double[:] Fp= momentum

    for i_l in range(1,dim-1):
        mu_value[i_l]= -( Fp[i_l+1] - Fp[i_l] )/dx
    
    if (boundaries=='periodic'):
        mu_value[0]= -(Fp[1] - Fp[0])/dx
        mu_value[dim-1]= -(Fp[0] - Fp[dim-1])/dx
    elif (boundaries=='noflux'):
        mu_value[0]= -(Fp[1])/dx
        mu_value[dim-1]= 0.0

    return np.asarray(mu_value)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)

cdef np.ndarray compute_mu_rho2(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] momentum, str boundaries):
    cdef int dim=x.shape[0]
    cdef Py_ssize_t i_l
    cdef double[:] mu_value= np.zeros([dim],dtype=np.float64)
    cdef float dx = x[1]-x[0]
    cdef double[:] Fp= momentum
    
    for i_l in range(1,dim-1):
        mu_value[i_l]= -( Fp[i_l] - Fp[i_l-1] )/dx
    
    if (boundaries=='periodic'):
        mu_value[0]= -(Fp[0] - Fp[dim-1])/dx
        mu_value[dim-1]= -(Fp[dim-1] - Fp[dim-2])/dx
    elif (boundaries=='noflux'):
        mu_value[0]= -(Fp[0])/dx
        mu_value[dim-1]= +(Fp[dim-2])/dx
    
    return np.asarray(mu_value)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
#@cython.infer_types(True)
cdef np.ndarray compute_mu_momentum1( np.ndarray[np.float64_t, ndim=1] x, double[:] rho , np.ndarray[np.float64_t, ndim=1] momentum , np.ndarray[np.float64_t, ndim=1] S_sum, object energy_derivative , str boundaries):
    
    #cdef double[:] rho_half
    
    cdef int dim=rho.shape[0]
    cdef Py_ssize_t i_l
    cdef double[:] mu_value= np.zeros([dim],dtype=np.float64)
    cdef float dx = x[1]-x[0]
    cdef double[:] U = energy_derivative(rho,x)
    cdef double[:] CFlux = np.divide( np.multiply(momentum,momentum), rho)
    cdef double[:] Source = np.zeros([dim],dtype=np.float64)
    cdef float f_ext = 0.0 #constant external velocity
    cdef double[:] Fp= np.zeros([dim],dtype=np.float64)

    #Boundary conditions
    if (boundaries=='periodic'):
        
        #Source[0]= - rho[0]*( U[1] - U[dim-1])/(2*dx) + S_sum[0] +f_ext
        Source[0]= - rho[0]*(U[1] - U[0])/dx + S_sum[0] + f_ext
        Fp[0]= CFlux[1]
    
        Source[dim-1]= - rho[dim-1]*(U[0] - U[dim-1])/dx + S_sum[dim-1] +f_ext
        #Source[dim-1]= - rho[dim-1]*( U[0] - U[dim-2])/(2*dx) + S_sum[dim-1] +f_ext
        Fp[dim-1]= CFlux[0]

    elif (boundaries=='noflux'):
        Source[0]= - rho[0]*( U[1] - U[0])/dx + S_sum[0] +f_ext
        Fp[0]=  CFlux[1]
        
        Source[dim-1]= - rho[dim-1]*( U[dim-1] - U[dim-2])/dx + S_sum[dim-1] +f_ext
        Fp[dim-1]= 0.0

    #elif (boundaries=='open'):
        # compute [0]
        #up=  -( U[1] - U[0])/ dx +f_ext
        #Fp[0]= up* (rho[0] + rho[1])/2
        
        #up=  - ( chemical_pot - U[dim-1]) / dx + f_ext
        #Fp[dim-1]= up* ( rho[dim-1] )

    for i_l in range(1,dim-1):
        #Source[i_l]= - rho[i_l]*( U[i_l+1] - U[i_l])/(dx) + S_sum[i_l] +f_ext
        Source[i_l]= -  rho[i_l]*(U[i_l+1] - U[i_l])/dx + S_sum[i_l] +f_ext
        Fp[i_l]=  CFlux[i_l+1]
        
        #Compute mu_value
        mu_value[i_l]= -( Fp[i_l] - Fp[i_l-1] )/dx + Source[i_l]

    if (boundaries=='periodic'):
        mu_value[0]= -(Fp[0] - Fp[dim-1])/dx + Source[0]
        mu_value[dim-1]= -(Fp[dim-1] - Fp[dim-2])/dx + Source[dim-1]
    elif (boundaries=='noflux'):
        mu_value[0]= -(Fp[0] - Fp[dim-1])/dx  + Source[0]
        mu_value[dim-1]= + Fp[dim-2]/dx  + Source[dim-1]

    #elif (boundaries=='open'):
        #mu_value[0]= -(Fp[0] - ( -(rho[0]) * (U[0] - chemical_pot)/dx ) )/dx
        #mu_value[dim-1]= -(Fp[dim-1] - Fp[dim-2])/dx

    return np.asarray( mu_value )





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
#@cython.infer_types(True)
cdef np.ndarray compute_mu_momentum2( np.ndarray[np.float64_t, ndim=1] x, double[:] rho , np.ndarray[np.float64_t, ndim=1] momentum , np.ndarray[np.float64_t, ndim=1] S_sum, object energy_derivative , str boundaries):
    
    #cdef double[:] rho_half
    
    cdef int dim=rho.shape[0]
    cdef Py_ssize_t i_l
    cdef double[:] mu_value= np.zeros([dim],dtype=np.float64)
    cdef float dx = x[1]-x[0]
    cdef double[:] U = energy_derivative(rho,x)
    cdef double[:] CFlux = np.divide( np.multiply(momentum,momentum), rho)
    cdef double[:] Source = np.zeros([dim],dtype=np.float64)
    cdef float f_ext = 0.0 #constant external velocity
    cdef double[:] Fp= np.zeros([dim],dtype=np.float64)
    
    #Boundary conditions
    if (boundaries=='periodic'):
        
        #Source[0]= - rho[0]*( U[1] - U[dim-1])/(2*dx) + S_sum[0] +f_ext
        Source[0]= - rho[0]*(U[0] - U[dim-1] )/dx + S_sum[0] + f_ext
        Fp[0]= CFlux[0]
        
        Source[dim-1]= - rho[dim-1]*(U[dim-1] - U[dim-2])/dx + S_sum[dim-1] +f_ext
        #Source[dim-1]= - rho[dim-1]*( U[0] - U[dim-2])/(2*dx) + S_sum[dim-1] +f_ext
        Fp[dim-1]= CFlux[dim-1]
    
    elif (boundaries=='noflux'):
        Source[0]= - rho[0]*( U[1] - U[0])/(dx) + S_sum[0] +f_ext
        Fp[0]=   CFlux[0]
        
        Source[dim-1]= - rho[dim-1]*( U[dim-1] - U[dim-2])/(dx) + S_sum[dim-1] +f_ext
        Fp[dim-1]= CFlux[dim-1]
    
    #elif (boundaries=='open'):
    # compute [0]
    #up=  -( U[1] - U[0])/ dx +f_ext
    #Fp[0]= up* (rho[0] + rho[1])/2
    
    #up=  - ( chemical_pot - U[dim-1]) / dx + f_ext
    #Fp[dim-1]= up* ( rho[dim-1] )
    
    for i_l in range(1,dim-1):
        #Source[i_l]= - rho[i_l]*( U[i_l+1] - U[i_l])/(dx) + S_sum[i_l] +f_ext
        Source[i_l]= - rho[i_l]*(U[i_l] - U[i_l-1])/dx + S_sum[i_l] +f_ext
        Fp[i_l]=  CFlux[i_l]
        
        #Compute mu_value
        mu_value[i_l]= -( Fp[i_l] - Fp[i_l-1] )/dx + Source[i_l]
    
    if (boundaries=='periodic'):
        mu_value[0]= -(Fp[0] - Fp[dim-1])/dx + Source[0]
        mu_value[dim-1]= -(Fp[dim-1] - Fp[dim-2])/dx + Source[dim-1]
    elif (boundaries=='noflux'):
        mu_value[0]= -Fp[0]/dx + Source[0]
        mu_value[dim-1]=  -(Fp[dim-1] - Fp[dim-2])/dx + Source[dim-1]
    
    #elif (boundaries=='open'):
    #mu_value[0]= -(Fp[0] - ( -(rho[0]) * (U[0] - chemical_pot)/dx ) )/dx
    #mu_value[dim-1]= -(Fp[dim-1] - Fp[dim-2])/dx
    
    return np.asarray( mu_value )




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)
cdef np.ndarray compute_mu_S(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] S_vec, np.ndarray[np.float64_t, ndim=1] momentum_vec, float A, float B, str boundaries):
    cdef int dim=x.shape[0]
    cdef Py_ssize_t i_l
    cdef double[:] mu_value= np.zeros([dim],dtype=np.float64)
    cdef float dx = x[1]-x[0]
    
    for i_l in range(dim):
        mu_value[i_l]= B * S_vec[i_l] - A * momentum_vec[i_l]

    return np.asarray(mu_value)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)
cdef np.ndarray compute_sigma_S( np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] rho_vec, float b, str boundaries):
    
    cdef int dim=x.shape[0]
    cdef Py_ssize_t i_l
    cdef double[:] sigma_value= np.zeros([dim],dtype=np.float64)
    cdef float dx = x[1]-x[0]
    
    for i_l in range(dim):
        sigma_value[i_l]= b * sqrt( 2/dx*rho_vec[i_l] )

    return np.asarray(sigma_value)











    
    
