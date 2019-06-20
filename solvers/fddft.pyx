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
from libc.math cimport sqrt,pi, isnan,isinf


################################################################################################################
#SOLVER FDDFT
###############################################################################################################
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)

cdef class fddft:

    cdef object mu
    cdef np.ndarray x
    cdef str boundaries
    cdef float open_boundary_value
    cdef Py_ssize_t i
    cdef object energy_derivative
    cdef int brownian_bridge_calls

    def __cinit__(self, str boundaries, np.ndarray[np.float64_t, ndim=1] x , object energy_derivative , float open_boundary_value =0.0):
        
        if (boundaries!='periodic' and boundaries!='noflux' and boundaries!='open' ):
            raise TypeError('Error in the parameter: boundaries')
        else:
            self.boundaries=boundaries
            self.open_boundary_value=open_boundary_value
            print('Boundaries= %s' %self.boundaries)

        self.energy_derivative= energy_derivative
        self.x= x

    def __call__(self, np.ndarray[np.float64_t, ndim=1] rho, np.ndarray[np.float64_t, ndim=3] eta, float dt, int n_steps, str space_method='CD', str time_method='EM', float theta=0, n_traj=1):
        
        adaptive_timestep=True
        
        if (space_method!='FO' and space_method!='CD' and space_method !='PR' and space_method!='UW'): # FO=forward , CD=centered difference, PR=parabolic
            raise TypeError('Error in the parameter: space_method')

        if (time_method!='EM' and time_method!='MI' and time_method!='RK2' and time_method!='RK3'): # EM= Euler Maruyama, MI= Milstein, RK2=Runge-Kutta
            raise TypeError('Error in the parameter: time_method')
    
        #Compute mu function
        mu=space_discretizator_deterministic(space_method, self.boundaries, self.x, self.energy_derivative, self.open_boundary_value)
        
        #Compute sigma function
        sigma=space_discretizator_stochastic(space_method, self.boundaries, self.x, self.energy_derivative)

        #Integrate in time
        my_time_integrator=time_integrator( mu, sigma)
        dim=rho.shape[0]
        rho_output= np.zeros([dim,n_steps,n_traj],dtype=np.float64)
  
#if(adaptive_timestep):
        brownian_bridge_calls=0
        for i_traj in range (n_traj):
            #First timestep
            rho_output[:,0,i_traj]= my_time_integrator(rho, sqrt(2)*eta[:,0,i_traj], dt, time_method ,theta)
            if ( any ( [ value < 0 or isnan(value) or isinf(value) for value in rho_output[:,0,i_traj]] ) ):
                rho_output[:,0,i_traj]=brownian_bridge(my_time_integrator, rho, sqrt(2)*eta[:,0,i_traj], dt, time_method ,theta)

            for i in range (1,n_steps):
                #timestep 'i'
                rho_output[:,i,i_traj]= my_time_integrator(rho_output[:,i-1,i_traj], sqrt(2)*eta[:,i,i_traj], dt, time_method ,theta)
                if ( any ( [ value < 0 or isnan(value) or isinf(value) for value in rho_output[:,i,i_traj]] ) ):
                    brownian_bridge_calls +=1
                    rho_output[:,i,i_traj]=brownian_bridge(my_time_integrator, rho_output[:,i-1,i_traj], sqrt(2)*eta[:,i,i_traj], dt, time_method ,theta)
        print("Brownian bridge calls: %d" %brownian_bridge_calls)
                
            #else:
            #for i_traj in range (n_traj):
            #First timestep
                #rho_output[:,0,i_traj]= my_time_integrator(rho, sqrt(2)*eta[:,0,i_traj], dt, time_method ,theta)
                #for i in range (1,n_steps):
                    #timestep 'i'
                    #rho_output[:,i,i_traj]= my_time_integrator(rho_output[:,i-1,i_traj], sqrt(2)*eta[:,i,i_traj], dt, time_method ,theta)
        
        return rho_output



################################################################################################################
# BROWNIAN BRIDGE FOR ADAPTIVE TIME STEP FDDFT
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

cdef brownian_bridge( object my_time_integrator, np.ndarray[np.float64_t, ndim=1] rho_start, np.ndarray[np.float64_t, ndim=1] eta_org, float dt, str time_method , float theta):
    
    cdef int dim=rho_start.shape[0]
    cdef Py_ssize_t i_dt
    cdef float dt_tmp=dt
    cdef np.ndarray[np.float64_t, ndim=1] rho_tmp
    cdef np.ndarray[np.float64_t, ndim=2] eta_extended = eta_org.reshape(dim,1)
    cdef int dt_partition=0
    cdef str flag_cfl_false=True
    
    while (flag_cfl_false):
        flag_cfl_false = False
        dt_partition +=1
        dt_tmp = dt_tmp/2
        rho_tmp= rho_start
        
        eta_extended= noise_partition(dim, eta_extended, dt_partition)

        for i_dt in range(2**dt_partition):
            rho_tmp= my_time_integrator(rho_tmp, eta_extended[:,i_dt], dt_tmp , time_method ,theta)
            if ( any ( [ value < 0 or isnan(value) or isinf(value) for value in rho_tmp ] )):
                flag_cfl_false = True
                break
    return rho_tmp

################################################################################################################
# DETERMINISTIC SPACE DISCRETIZATOR FDDFT
###############################################################################################################

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)

cdef class space_discretizator_deterministic:
    cdef np.ndarray x
    cdef str boundaries
    cdef Py_ssize_t i
    cdef object energy_derivative
    cdef str space_method
    cdef float open_boundary_value
    cdef int dim
    cdef float dx
    
    def __cinit__(self, str space_method, str boundaries, np.ndarray[np.float64_t, ndim=1] x, object energy_derivative, float open_boundary_value):
        self.boundaries=boundaries
        self.energy_derivative= energy_derivative
        self.open_boundary_value = open_boundary_value
        self.x= x
        self.space_method=space_method
    
    def __call__(self, double[:] rho):
        return  np.asarray( compute_mu( rho, self.boundaries, self.x, self.energy_derivative, self.open_boundary_value) )


#AUXILIARY FUNCTION FOR DETERMINISTIC SPACE DISCRETIZATOR
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)
cdef float f_minmod(float a, float b, float c):
    if (a>0 and b>0 and c>0):
        return min(a,b,c)
    elif (a<0 and b<0 and c<0):
        return max(a,b,c)
    else:
        return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)
cdef float myMean( np.ndarray[np.float32_t, ndim=1] a):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = a.shape[0]
    cdef double m = 0.0
    for i in range(n):
        m += a[i]
    m /= n
    return m

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
#@cython.infer_types(True)
cdef double[:] compute_mu( double[:] rho, str boundaries, np.ndarray[np.float64_t, ndim=1] x, object energy_derivative , float open_boundary_value):
    cdef int dim=rho.shape[0]
    cdef Py_ssize_t i_l
    cdef double[:] mu_value= np.zeros([dim],dtype=np.float64)
    cdef float dx = x[1]-x[0]
    cdef double[:] U = energy_derivative(rho,x)
    cdef float up
    cdef float upjp
    cdef float umjp
    cdef float k_par=3  # Upwind -> k_par=0    Central ->k_par=Infty
    cdef float f_ext =0.0 #constant external velocity
    cdef int w =5 #window of upwind 2 order
    cdef double[:] Fp= np.zeros([dim],dtype=np.float64)
    cdef np.ndarray[np.float32_t, ndim=1] grad_rho= np.zeros([dim],dtype=np.float32)
    cdef np.ndarray[np.int_t, ndim=1] theta_space= np.zeros([dim],dtype=int)
    cdef float chemical_pot= energy_derivative( open_boundary_value * np.ones(dim) ,x) [0]
    #print(chemical_pot)
    #chemical_pot= energy_derivative( open_boundary_value * np.ones(dim) ,x) [0]
    #print(chemical_pot)
    
    cdef float[:] rho_xj = np.zeros([dim],dtype=np.float32)
    
    for i_l in range(1,dim-1):
       rho_xj[i_l]= f_minmod( (rho[i_l+1]-rho[i_l]), (rho[i_l+1]-rho[i_l-1])/4, (rho[i_l]-rho[i_l-1]) )
       grad_rho[i_l]= abs(rho[i_l+1] - rho[i_l])
       
    if (boundaries=='periodic'):
       rho_xj[0]= f_minmod( (rho[1]-rho[0]), (rho[1]-rho[dim-1])/4, (rho[0]-rho[dim-1]))
       grad_rho[0]= abs(rho[1] - rho[0])

       rho_xj[dim-1]= f_minmod( (rho[0]-rho[dim-1]) , (rho[0]-rho[dim-2])/4, (rho[dim-1]-rho[dim-2]) )
       grad_rho[dim-1]= abs(rho[0] - rho[dim-1])

    elif(boundaries=='noflux'):
       rho_xj[0]=0
       rho_xj[dim-1]=0

    elif(boundaries=='open'):
        rho_xj[0]=0
        rho_xj[dim-1]=0

    for i_l in range(0 , w):
        if( grad_rho[i_l] >= k_par * myMean(grad_rho[i_l:i_l+2*w]) ):
           theta_space[i_l]=1
    
    for i_l in range(w ,dim-w):
        if( grad_rho[i_l] >= k_par*myMean(grad_rho[i_l-w:i_l+w])  ):
            theta_space[i_l-w:i_l+w]=1
        
    for i_l in range(dim-w ,dim):
        if( grad_rho[i_l] >= k_par*myMean(grad_rho[i_l-2*w:i_l])  ):
           theta_space[i_l]=1

#print(theta_space)

    #Boundary conditions
    if (boundaries=='periodic'):
        # compute mu[0]
        up=  -( U[1] - U[0])/ dx +f_ext
        
        if (theta_space[0]==0):
            #CD order start
            Fp[0]= up* (rho[0] + rho[1])/2
        else:
            #Upwind 2 order start
            upjp=max(up,0.0)
            umjp=min(up,0.0)
            Fp[0]= upjp* ( rho[0]+rho_xj[0] ) + umjp * (rho[1]-rho_xj[1])

        # compute mu[dim-1]
        up=  -( U[0] - U[dim-1]) / dx + f_ext
        
        if (theta_space[dim-1]==0):
            #CD order
            Fp[dim-1]= up* (rho[dim-1] + rho[0])/2
        
        else:
            #Upwind 2 order
            upjp=max(up,0.0)
            umjp=min(up,0.0)
            Fp[dim-1]= upjp* ( rho[dim-1] + rho_xj[dim-1] ) + umjp * ( rho[0] - rho_xj[0] )

    elif (boundaries=='noflux'):
        
        # compute mu[0]
        up=  -( U[1] - U[0] )/dx
    
        if (theta_space[0]==0):
            #CD order start
            Fp[0]= up* (rho[0] + rho[1])/2
        else:
            #Upwind 2 order start
            upjp=max(up,0.0)
            umjp=min(up,0.0)
            Fp[0]= upjp* ( rho[0]+rho_xj[0] ) + umjp * (rho[1]-rho_xj[1])
        
        # compute mu[dim-1]
        up=  -( U[0] - U[dim-1]) / dx + f_ext
        Fp[dim-1]= 0

    elif (boundaries=='open'):
        # compute [0]
        up=  -( U[1] - U[0])/ dx +f_ext
        
        if (theta_space[0]==0):
            #CD order start
            Fp[0]= up* (rho[0] + rho[1])/2
        else:
            #Upwind 2 order start
            upjp=max(up,0.0)
            umjp=min(up,0.0)
            Fp[0]= upjp* ( rho[0]+rho_xj[0] ) + umjp * (rho[1]-rho_xj[1])

        # compute [dim-1]

        up=  - ( chemical_pot - U[dim-1]) / dx + f_ext
        #CD order
        Fp[dim-1]= up* ( rho[dim-1] + open_boundary_value )/2
        #Fp[dim-1]= up* ( rho[dim-1] )

    for i_l in range(1,dim-1):
        #Compute u(j+1/2) and u(j-1/2)
        up=  - ( U[i_l+1] - U[i_l])/ dx + f_ext

        #Compute theta_space
        if (theta_space[i_l]==0):
            #CD order start
            Fp[i_l]= up* (rho[i_l] + rho[i_l+1])/2
        else:
            #Upwind 2 order start
            upjp=max(up,0.0)
            umjp=min(up,0.0)
            Fp[i_l]= upjp* ( rho[i_l] + rho_xj[i_l] ) + umjp * ( rho[i_l+1] - rho_xj[i_l+1] )

        #Compute mu_value
        mu_value[i_l]= -( Fp[i_l] - Fp[i_l-1] )/dx

    if (boundaries=='periodic'):
        mu_value[0]= -(Fp[0] - Fp[dim-1])/dx
        mu_value[dim-1]= -(Fp[dim-1] - Fp[dim-2])/dx
    elif (boundaries=='noflux'):
        mu_value[0]= -Fp[0]/dx
        mu_value[dim-1]= +Fp[dim-2]/dx
    elif (boundaries=='open'):
        mu_value[0]= -(Fp[0] - ( -(rho[0]) * (U[0] - chemical_pot)/dx ) )/dx
        mu_value[dim-1]= -(Fp[dim-1] - Fp[dim-2])/dx


    return np.asarray( mu_value )



################################################################################################################
# STOCHASTIC SPACE DISCRETIZATOR FDDFT
###############################################################################################################

#AUXILIARY FUNCTION FOR STOCHASTIC SPACE DISCRETIZATOR
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)

cdef double[:,:] compute_1D_sigma( double[:] rho, double[:] eta, str space_method, str boundaries, float dx):
    cdef int dim=rho.shape[0]
    cdef Py_ssize_t i_l
    cdef double[:,:] sigma_value= np.zeros([dim,dim],dtype=np.float64)
    cdef float dx_eff= dx*sqrt(dx)
    
    #LINEARIZED
    #            Fps= np.multiply(ups, np.sqrt(rho_ref))
    #            Fms= np.multiply(ums, np.sqrt(rho_ref))


    #forward stochastic discretization
    if (space_method=='FO'):
        for i_l in range(1,dim-1):
            sigma_value[i_l,i_l]= -sqrt(rho[i_l]) / dx_eff
            sigma_value[i_l,i_l+1]= sqrt( rho[i_l+1] ) / dx_eff
        
        if (boundaries=='periodic'):
            sigma_value[0,0]= -sqrt(rho[0]) / dx_eff
            sigma_value[0,1]= sqrt( rho[1] ) / dx_eff
            
            sigma_value[dim-1,dim-1]=-sqrt(rho[dim-1]) / dx_eff
            sigma_value[dim-1,0]=sqrt( rho[0] ) / dx_eff
        elif (boundaries=='noflux'):
            sigma_value[0,0]=0.0
            sigma_value[0,1]=sqrt( rho[1] ) / dx_eff
            
            sigma_value[dim-1,dim-1]=-sqrt(rho[dim-1]) / dx_eff
            sigma_value[dim-1,0]=0.0

    #central difference stochastic discretization
    if (space_method=='CD'):
        for i_l in range(1,dim-1):
            if (rho[i_l]>10**(-8)):
                sigma_value[i_l,i_l]= -sqrt((rho[i_l] + rho[i_l-1])/2) / dx_eff
                sigma_value[i_l,i_l+1]= sqrt((rho[i_l] + rho[i_l+1])/2) / dx_eff

        if (boundaries=='periodic'):
            if (rho[0]>10**(-8)):
                sigma_value[0,0]= -sqrt((rho[0] + rho[dim-1])/2) / dx_eff
                sigma_value[0,1]= sqrt((rho[0] + rho[1])/2) / dx_eff
            
            if (rho[dim-1]>10**(-8)):
                sigma_value[dim-1,dim-1]=-sqrt((rho[dim-1] + rho[dim-2])/2) / dx_eff
                sigma_value[dim-1,0]=sqrt((rho[dim-1] + rho[0])/2) / dx_eff
                    
        elif (boundaries=='noflux' or boundaries=='open'):
            sigma_value[0,0]=0.0
            sigma_value[0,1]=sqrt((rho[0] + rho[1])/2) / dx_eff
            
            sigma_value[dim-1,dim-1]=-sqrt((rho[dim-1] + rho[dim-2])/2) / dx_eff
            sigma_value[dim-1,0]=0.0


#parabolic stochastic discretization
    if (space_method=='PR'):
        a1=(1+sqrt(3))/4
        a2=(1-sqrt(3))/4
        for i_l in range(2,dim-2):
            sigma_value[i_l,i_l]= -sqrt( a1* (rho[i_l] + rho[i_l-1]) + a2*(rho[i_l+1] + rho[i_l-2])  ) / dx_eff
            sigma_value[i_l,i_l+1]= sqrt(  a1* (rho[i_l] + rho[i_l+1]) + a2 *(rho[i_l-1] + rho[i_l+2])  ) / dx_eff

        if (boundaries=='periodic'):
                
            #cell 0
            sigma_value[0,0]= -sqrt( a1* (rho[0] + rho[dim-1]) + a2*(rho[+1] + rho[dim-2])  ) / dx_eff
            sigma_value[0,1]= sqrt( a1* (rho[0] + rho[+1]) + a2 *(rho[dim-1] + rho[+2])  ) / dx_eff
            
            #cell 1
            sigma_value[1,1]= -sqrt( a1* (rho[1] + rho[0]) + a2*(rho[2] + rho[dim-1])  ) / dx_eff
            sigma_value[1,2]= sqrt( a1* (rho[1] + rho[2]) + a2 *(rho[0] + rho[3])  ) / dx_eff
            
            #cell dim-1
            sigma_value[dim-1,dim-1]= - sqrt( a1* (rho[dim-1] + rho[dim-2]) + a2*(rho[0] + rho[dim-3])  ) / dx_eff
            sigma_value[dim-1,0]= sqrt(  a1* (rho[dim-1] + rho[0]) + a2 *(rho[dim-2] + rho[1])  ) / dx_eff
            
            #cell dim-2
            sigma_value[dim-2,dim-2]= - sqrt( a1* (rho[dim-2] + rho[dim-3]) + a2*(rho[dim-1] + rho[dim-4])  ) / dx_eff
            sigma_value[dim-2,dim-1]= sqrt(  a1* (rho[dim-2] + rho[dim-1]) + a2 *(rho[dim-3] + rho[0])  ) / dx_eff
    
        elif (boundaries=='noflux'):
            #cell 0
            sigma_value[0,0]= 0.0
            sigma_value[0,1]= sqrt( a1* (rho[0] + rho[+1]) + a2 *(0.0 + rho[+2])  ) / dx_eff
            
            #cell 1
            sigma_value[1,1]= -sqrt( a1* (rho[1] + rho[0]) + a2*(rho[2] + 0.0)  ) / dx_eff
            sigma_value[1,2]= sqrt( a1* (rho[1] + rho[2]) + a2 *(rho[0] + rho[3])  ) / dx_eff
            
            #cell dim-1
            sigma_value[dim-1,dim-1]= - sqrt( a1* (rho[dim-1] + rho[dim-2]) + a2*(0.0 + rho[dim-3])  ) / dx_eff
            sigma_value[dim-1,0]= 0.0
            
            #cell dim-2
            sigma_value[dim-2,dim-2]= - sqrt( a1* (rho[dim-2] + rho[dim-3]) + a2*(rho[dim-1] + rho[dim-4]) ) / dx_eff
            sigma_value[dim-2,dim-1]= sqrt(  a1* (rho[dim-2] + rho[dim-1]) + a2 *(rho[dim-3] + 0.0)  ) / dx_eff


    #upwind stochastic discretization
    if (space_method=='UW'):
        for i_l in range(1,dim-1):
            if ( eta[i_l]<=0 ):
                sigma_value[i_l,i_l]= -sqrt(rho[i_l] ) / dx_eff
            else:
                sigma_value[i_l,i_l]= -sqrt(rho[i_l-1] ) / dx_eff

            if ( eta[i_l+1]<=0 ):
                sigma_value[i_l,i_l+1]= sqrt(rho[i_l+1] ) / dx_eff
            else:
                sigma_value[i_l,i_l+1]= sqrt(rho[i_l] ) / dx_eff

        if (boundaries=='periodic'):
            
            #cell 0
            if ( eta[0]<=0 ):
                sigma_value[0,0]= -sqrt(rho[0] ) / dx_eff
            else:
                sigma_value[0,0]= -sqrt(rho[dim-1] ) / dx_eff

            if ( eta[1]<=0 ):
                sigma_value[0,1]= sqrt(rho[1] ) / dx_eff
            else:
                sigma_value[0,1]= sqrt(rho[0] ) / dx_eff

            #cell dim-1
            if ( eta[dim-1]<=0 ):
                sigma_value[dim-1,dim-1]= -sqrt(rho[dim-1] ) / dx_eff
            else:
                sigma_value[dim-1,dim-1]= -sqrt(rho[dim-2] ) / dx_eff
            
            if ( eta[0]<=0 ):
                sigma_value[dim-1,0]= sqrt(rho[0] ) / dx_eff
            else:
                sigma_value[dim-1,0]= sqrt(rho[dim-1] ) / dx_eff

        elif (boundaries=='noflux'):
            #cell 0
            sigma_value[0,0]=0.0
            
            if ( eta[1]<=0 ):
                sigma_value[0,1]= sqrt(rho[1] ) / dx_eff
            else:
                sigma_value[0,1]= sqrt(rho[0] ) / dx_eff
            
            #cell dim-1
            if ( eta[dim-1]<=0 ):
                sigma_value[dim-1,dim-1]= -sqrt(rho[dim-1] ) / dx_eff
            else:
                sigma_value[dim-1,dim-1]= -sqrt(rho[dim-2] ) / dx_eff
            
            sigma_value[dim-1,0]=0.0


    return sigma_value






#AUXILIARY FUNCTION FOR STOCHASTIC SPACE DISCRETIZATOR
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.infer_types(True)
cdef double[:,:,:] compute_2D_sigma( double[:,:] rho, double[:] eta, str space_method, str boundaries, float dx):
    cdef int dim=rho.shape[0]
    cdef Py_ssize_t i_l, i_m
    cdef double[:,:,:] sigma_value_extended= np.zeros([dim,dim,dim],dtype=np.float64)
    cdef float dx_eff= dx*sqrt(dx)
    
    #forward stochastic discretization
    if (space_method=='FO'):
        for i_m in range(dim):
            for i_l in range(1,dim-1):
                sigma_value_extended[i_l,i_l,i_m]= -sqrt(rho[i_l,i_m]) / dx_eff
                sigma_value_extended[i_l,i_l+1,i_m]= sqrt( rho[i_l+1,i_m] ) / dx_eff
            
            if (boundaries=='periodic'):
                sigma_value_extended[0,0,i_m]= -sqrt(rho[0,i_m]) / dx_eff
                sigma_value_extended[0,1,i_m]= sqrt( rho[1,i_m] ) / dx_eff
                
                sigma_value_extended[dim-1,dim-1,i_m]=-sqrt(rho[dim-1,i_m]) / dx_eff
                sigma_value_extended[dim-1,0,i_m]=sqrt( rho[0,i_m] ) / dx_eff
            elif (boundaries=='noflux'):
                sigma_value_extended[0,0,i_m]=0.0
                sigma_value_extended[0,1,i_m]=sqrt( rho[1,i_m] ) / dx_eff
                
                sigma_value_extended[dim-1,dim-1,i_m]=-sqrt(rho[dim-1,i_m]) / dx_eff
                sigma_value_extended[dim-1,0,i_m]=0.0
    
    #central difference stochastic discretization
    if (space_method=='CD'):
        for i_m in range(dim):
            for i_l in range(1,dim-1):
                if (rho[i_l,i_m]>10**(-8)):
                    sigma_value_extended[i_l,i_l,i_m]= -sqrt((rho[i_l,i_m] + rho[i_l-1,i_m])/2) / dx_eff
                    sigma_value_extended[i_l,i_l+1,i_m]= sqrt((rho[i_l,i_m] + rho[i_l+1,i_m])/2) / dx_eff
        
        if (boundaries=='periodic'):
            for i_m in range(dim):
                if (rho[dim-1,i_m]>10**(-8)):
                    sigma_value_extended[0,0,i_m]= -sqrt((rho[0,i_m] + rho[dim-1,i_m])/2) / dx_eff
                    sigma_value_extended[0,1,i_m]= sqrt((rho[0,i_m] + rho[1,i_m])/2) / dx_eff

                if (rho[i_l,i_m]>10**(-8)):
                    sigma_value_extended[dim-1,dim-1,i_m]=-sqrt((rho[dim-1,i_m] + rho[dim-2,i_m])/2) / dx_eff
                    sigma_value_extended[dim-1,0,i_m]=sqrt((rho[dim-1,i_m] + rho[0,i_m])/2) / dx_eff
            
        elif (boundaries=='noflux' or boundaries=='open' ):
            for i_m in range(dim):
                sigma_value_extended[0,0,i_m]=0.0
                sigma_value_extended[0,1,i_m]=sqrt((rho[0,i_m] + rho[1,i_m])/2) / dx_eff
                
                sigma_value_extended[dim-1,dim-1,i_m]=-sqrt((rho[dim-1,i_m] + rho[dim-2,i_m])/2) / dx_eff
                sigma_value_extended[dim-1,0,i_m]=0.0


    #parabolic stochastic discretization
    if (space_method=='PR'):
        a1=(1+sqrt(3))/4
        a2=(1-sqrt(3))/4
        for i_m in range(dim):
            for i_l in range(2,dim-2):
                sigma_value_extended[i_l,i_l,i_m]= -sqrt( a1* (rho[i_l,i_m] + rho[i_l-1,i_m]) + a2*(rho[i_l+1,i_m] + rho[i_l-2,i_m])  ) / dx_eff
                sigma_value_extended[i_l,i_l+1,i_m]= sqrt(  a1* (rho[i_l,i_m] + rho[i_l+1,i_m]) + a2 *(rho[i_l-1,i_m] + rho[i_l+2,i_m])  ) / dx_eff
        
        if (boundaries=='periodic'):
            for i_m in range(dim):
                
                #cell 0
                sigma_value_extended[0,0,i_m]= -sqrt( a1* (rho[0,i_m] + rho[dim-1,i_m]) + a2*(rho[+1,i_m] + rho[dim-2,i_m])  ) / dx_eff
                sigma_value_extended[0,1,i_m]= sqrt( a1* (rho[0,i_m] + rho[+1,i_m]) + a2 *(rho[dim-1,i_m] + rho[+2,i_m])  ) / dx_eff
 
                #cell 1
                sigma_value_extended[1,1,i_m]= -sqrt( a1* (rho[1,i_m] + rho[0,i_m]) + a2*(rho[2,i_m] + rho[dim-1,i_m])  ) / dx_eff
                sigma_value_extended[1,2,i_m]= sqrt( a1* (rho[1,i_m] + rho[2,i_m]) + a2 *(rho[0,i_m] + rho[3,i_m])  ) / dx_eff
                
                #cell dim-1
                sigma_value_extended[dim-1,dim-1,i_m]= - sqrt( a1* (rho[dim-1,i_m] + rho[dim-2,i_m]) + a2*(rho[0,i_m] + rho[dim-3,i_m])  ) / dx_eff
                sigma_value_extended[dim-1,0,i_m]= sqrt(  a1* (rho[dim-1,i_m] + rho[0,i_m]) + a2 *(rho[dim-2,i_m] + rho[1,i_m])  ) / dx_eff

                #cell dim-2
                sigma_value_extended[dim-2,dim-2,i_m]= - sqrt( a1* (rho[dim-2,i_m] + rho[dim-3,i_m]) + a2*(rho[dim-1,i_m] + rho[dim-4,i_m])  ) / dx_eff
                sigma_value_extended[dim-2,dim-1,i_m]= sqrt(  a1* (rho[dim-2,i_m] + rho[dim-1,i_m]) + a2 *(rho[dim-3,i_m] + rho[0,i_m])  ) / dx_eff
                    
        elif (boundaries=='noflux'):
            for i_m in range(dim):
                #cell 0
                sigma_value_extended[0,0,i_m]= 0.0
                sigma_value_extended[0,1,i_m]= sqrt( a1* (rho[0,i_m] + rho[+1,i_m]) + a2 *(0.0 + rho[+2,i_m])  ) / dx_eff
                
                #cell 1
                sigma_value_extended[1,1,i_m]= -sqrt( a1* (rho[1,i_m] + rho[0,i_m]) + a2*(rho[2,i_m] + 0.0)  ) / dx_eff
                sigma_value_extended[1,2,i_m]= sqrt( a1* (rho[1,i_m] + rho[2,i_m]) + a2 *(rho[0,i_m] + rho[3,i_m])  ) / dx_eff
                
                #cell dim-1
                sigma_value_extended[dim-1,dim-1,i_m]= - sqrt( a1* (rho[dim-1,i_m] + rho[dim-2,i_m]) + a2*(0.0 + rho[dim-3,i_m])  ) / dx_eff
                sigma_value_extended[dim-1,0,i_m]= 0.0
                
                #cell dim-2
                sigma_value_extended[dim-2,dim-2,i_m]= - sqrt( a1* (rho[dim-2,i_m] + rho[dim-3,i_m]) + a2*(rho[dim-1,i_m] + rho[dim-4,i_m])  ) / dx_eff
                sigma_value_extended[dim-2,dim-1,i_m]= sqrt(  a1* (rho[dim-2,i_m] + rho[dim-1,i_m]) + a2 *(rho[dim-3,i_m] + 0.0)  ) / dx_eff

    #upwind stochastic discretization
    if (space_method=='UW'):
        
        for i_m in range(dim):
        
            for i_l in range(1,dim-1):
                if ( eta[i_l]<=0 ):
                    sigma_value_extended[i_l,i_l,i_m]= -sqrt(rho[i_l,i_m] ) / dx_eff
                else:
                    sigma_value_extended[i_l,i_l,i_m]= -sqrt(rho[i_l-1,i_m] ) / dx_eff
                
                if ( eta[i_l+1]<=0 ):
                    sigma_value_extended[i_l,i_l+1,i_m]= sqrt(rho[i_l+1,i_m] ) / dx_eff
                else:
                    sigma_value_extended[i_l,i_l+1,i_m]= sqrt(rho[i_l,i_m] ) / dx_eff

            if (boundaries=='periodic'):
                #cell 0
                if ( eta[0]<=0 ):
                    sigma_value_extended[0,0,i_m]= -sqrt(rho[0,i_m] ) / dx_eff
                else:
                    sigma_value_extended[0,0,i_m]= -sqrt(rho[dim-1,i_m] ) / dx_eff
        
                if ( eta[1]<=0 ):
                    sigma_value_extended[0,1,i_m]= sqrt(rho[1,i_m] ) / dx_eff
                else:
                    sigma_value_extended[0,1,i_m]= sqrt(rho[0,i_m] ) / dx_eff
            
                #cell dim-1
                if ( eta[dim-1]<=0 ):
                    sigma_value_extended[dim-1,dim-1,i_m]= -sqrt(rho[dim-1,i_m] ) / dx_eff
                else:
                    sigma_value_extended[dim-1,dim-1,i_m]= -sqrt(rho[dim-2,i_m] ) / dx_eff

                if ( eta[0]<=0 ):
                    sigma_value_extended[dim-1,0,i_m]= sqrt(rho[0,i_m] ) / dx_eff
                else:
                    sigma_value_extended[dim-1,0,i_m]= sqrt(rho[dim-1,i_m] ) / dx_eff
            
            elif (boundaries=='noflux'):
                #cell 0
                sigma_value_extended[0,0,i_m]=0.0
                
                if ( eta[1]<=0 ):
                    sigma_value_extended[0,1,i_m]= sqrt(rho[1,i_m] ) / dx_eff
                else:
                    sigma_value_extended[0,1,i_m]= sqrt(rho[0,i_m] ) / dx_eff
                
                #cell dim-1
                if ( eta[dim-1]<=0 ):
                    sigma_value_extended[dim-1,dim-1,i_m]= -sqrt(rho[dim-1,i_m] ) / dx_eff
                else:
                    sigma_value_extended[dim-1,dim-1,i_m]= -sqrt(rho[dim-2,i_m] ) / dx_eff
                
                sigma_value_extended[dim-1,0,i_m]=0.0
    

    return sigma_value_extended




#CLASS STOCHASTIC SPACE DISCRETIZATOR
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef class space_discretizator_stochastic:
    cdef np.ndarray x
    cdef float dx
    cdef str boundaries
    cdef Py_ssize_t i
    cdef str space_method
    cdef double[:,:] sigma_value
    cdef double[:,:,:] sigma_value_extended
    
    def __cinit__(self, str space_method, str boundaries, np.ndarray[np.float64_t, ndim=1] x, object energy_derivative ):
        self.boundaries=boundaries
        self.x= x
        self.dx= x[1]-x[0]
        self.space_method=space_method

    def __call__(self, rho, eta):
        if ( rho.ndim==1):
            sigma_value = compute_1D_sigma(rho, eta, self.space_method, self.boundaries, self.dx )
            return np.asarray(sigma_value)
        else:
            sigma_value_extended = compute_2D_sigma(rho, eta, self.space_method, self.boundaries, self.dx)
            return np.asarray(sigma_value_extended)







    
    
