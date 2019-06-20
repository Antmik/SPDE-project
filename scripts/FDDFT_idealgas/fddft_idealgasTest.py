###################################################################################################################
# Antonio Russo
# Imperial College London
# Date: 14/10/2028
###################################################################################################################
import sys, os, shutil
LibPath = os.environ['SPDE']
sys.path.append(LibPath)

import numpy as np
import matplotlib.pyplot as plt
import time

import pyximport
pyximport.install(pyimport=False , setup_args={'include_dirs': np.get_include()})

from time_integrator  import time_integrator
from solvers.fddft  import fddft
from utils.utils  import compute_weak_error, compute_strong_error, compute_cpu_time,structure_factor

######################################################################################
## Test
######################################################################################
if __name__ == '__main__':

    #Data
    n=50
    n_steps=1000
    n_steps_eq=10
    n_traj=10
    rho_ref=0.5
    dx=100
    boundaries='periodic'
    
    rho0= rho_ref*np.ones(n) + np.sqrt( (1-1/n) * rho_ref /dx ) *np.random.randn(n)
    x=np.linspace(-(n-1)*dx/2,(n-1)*dx/2,n-1)
    
    def energy_derivative(rho_vec, x_vec):
        return np.log(rho_vec)
    
    dt= 0.5*dx*dx #1* 10**(-3)
    eta=np.random.randn(n,n_steps,n_traj)
    
    #Initialize fddft class
    my_fddft= fddft(boundaries, x , energy_derivative)
    
    t0 = time.time()
    rhoEM, total_timeEM = compute_cpu_time( my_fddft(rho0, eta, dt, n_steps,space_method='UW', time_method='EM', theta=0.5, n_traj=n_traj) )
    t1 = time.time()
    total_timeEM=t1-t0
    print("Time:%f" %total_timeEM)

    std= np.mean( np.std( rhoEM[:,-1,:] ,axis=1) )
    std_Theory=np.sqrt( (1-1/n) * rho_ref / (dx) )
    print("Std:%.8f" %std)
    print("Std_theory:%.8f" %std_Theory)

    SF=structure_factor( np.reshape( rhoEM[:,-n_steps_eq-1:-1,:], (n, n_traj*n_steps_eq) ) , x )
#    t0 = time.time()
#    rhoMI = my_geometric_brownian(rho0, eta, dt, n_steps, time_method='MI', theta=1)
#    t1 = time.time()
#    total_timeMI = t1-t0
#    print("MIim:",total_timeMI)
#
#    t0 = time.time()
#    rhoRK2 = my_geometric_brownian(rho0, eta, dt, n_steps, time_method='RK2')
#    t1 = time.time()
#    total_timeRK2 = t1-t0
#    print("RK2:",total_timeRK2)
#
#    t0 = time.time()
#    rhoRK3 = my_geometric_brownian(rho0, eta, dt, n_steps, time_method='RK3')
#    t1 = time.time()
#    total_timeRK3 = t1-t0
#    print("RK3:",total_timeRK3)


#######################################################################################################################
#    Plots
#######################################################################################################################
fig = plt.figure()
plt.title(r'$\rho(x)$')
plt.plot(rhoEM[:,-1], ls='-', label="Euler-Maruyama")
#    plt.plot(rhoMI[:,-1] , ls='--' ,label="Milstein")
#    plt.plot(rhoRK2[:,-1] , ls=':' , label="Runge-Kutta 2")
#    plt.plot(rhoRK3 , ls=':' , label="Runge-Kutta 3")
#    plt.plot(rhoTheory[:,-1] , ls=':', label="Theory")
plt.legend()
#    plt.yscale('log')
#plt.rc('text',usetex=True)
#plt.rc('font',family='serif')
plt.xlabel(r'$\rho(x)$')
plt.ylabel(r'x')
plt.tight_layout()
plt.savefig('rho.png')

    
    
    
