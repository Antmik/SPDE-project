###################################################################################################################
# Antonio Russo
# Imperial College London
# Date: 26/10/2028
###################################################################################################################
import sys, os, shutil
LibPath = os.environ['SPDE']
sys.path.append(LibPath)

import numpy as np
import matplotlib.pyplot as plt
import time

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

from time_integrator  import time_integrator
from solvers.geometric_brownian  import geometric_brownian

###############################################################################################################

#Weak Error
def compute_weak_error(approx_sol, exact_sol): #approx_sol=[dim1] and exact_sol=[dim2]
    if (approx_sol.ndim==1 and exact_sol.ndim==1):
        return  abs( np.mean(approx_sol) -  np.mean(exact_sol))
    elif (approx_sol.ndim==2 and exact_sol.ndim==2):
#        return  np.amax( abs( np.mean(approx_sol,axis=1) -  np.mean(exact_sol,axis=1)) )
        return  np.mean( abs( np.mean(approx_sol,axis=1) -  np.mean(exact_sol,axis=1)) )
    else:
        raise ("Error in dimensions")

#Strong Error
def compute_strong_error(approx_sol, exact_sol): #approx_sol=[dim] and exact_sol=[dim]
    if (approx_sol.ndim==1 and exact_sol.ndim==1):
        return  np.amax(np.abs(approx_sol-exact_sol))
    elif (approx_sol.ndim==2 and exact_sol.ndim==2):
#        return  np.amax( np.mean(np.abs(approx_sol-exact_sol), axis=1) )
        return  np.mean( np.mean(np.abs(approx_sol-exact_sol), axis=1) )
    else:
        raise ("Error in dimensions")



#cpu time
def compute_cpu_time(process):
    t0 = time.time()
    output = process
    t1 = time.time()
    total_time = t1-t0
    return  output , total_time


#Structure factor
def structure_factor( rho , x ):
    dim=rho.shape[0]
    n_traj=rho.shape[1]
    dx=x[1]-x[0]
    V=dim*dx

    k= 2*np.pi/V*np.arange(0,int(dim/2))
    j= np.arange(0,dim)
    Re_rho= np.zeros(n_traj)
    Im_rho= np.zeros(n_traj)
    S=np.zeros(int(dim/2))

    for i_k in range(int(dim/2)):
        for i_traj in range(n_traj):
            Re_rho[i_traj]= 1/V*dx* np.sum( rho[:,i_traj]*np.cos(j*dx*k[i_k]) )
            Im_rho[i_traj]= -1/V*dx* np.sum( rho[:,i_traj]*np.sin(j*dx*k[i_k]) )
        
        Re_rho -= np.mean(Re_rho)
        Im_rho -= np.mean(Im_rho)

        S[i_k]= V*np.mean( Re_rho**2 + Im_rho**2)

    #Plot
    fig = plt.figure()
    plt.plot(k/k[-1],S , ls='-')
    plt.legend()
    #    plt.yscale('log')
#    plt.rc('text',usetex=True)
#    plt.rc('font',family='serif')
    plt.ylabel(r'$S(k)$')
    plt.xlabel(r'k / k_max')
    plt.tight_layout()
    plt.savefig('Structure_factor.pdf')

    return k,S












    
    

    
    
    
