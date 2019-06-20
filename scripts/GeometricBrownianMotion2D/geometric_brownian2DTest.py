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
pyximport.install(setup_args={'include_dirs': np.get_include()})

from time_integrator  import time_integrator
from solvers.geometric_brownian2D  import geometric_brownian2D

######################################################################################
## Test
######################################################################################
if __name__ == '__main__':

    #Data
    n=2
    n_steps=1
    n_traj=1
    rho0= np.ones(n) #+ 0.1*np.random.randn(n)
    value_mu=-1.0
    value_sigma=0.5
    dt= 1* 10**(-11)
    eta=np.random.randn(n,n_steps,n_traj)
    
    P_mat = np.array([[1,1],[-1,1]])
    #Time for every method
    W= np.cumsum(np.sqrt(dt)*eta,axis=1)
    rhoTheory= rho0[0] * np.exp( value_mu*n_steps*dt - 0.5 * (value_sigma**2)*n_steps*dt +  value_sigma*W )
    my_geometric_brownian= geometric_brownian2D(value_mu, value_sigma)

    t0 = time.time()
    rhoEM = my_geometric_brownian(rho0, eta, dt, n_steps, time_method='EM', theta=1)
    t1 = time.time()
    total_timeEM = t1-t0
    print("EMim:",total_timeEM)

    t0 = time.time()
    rhoMI = my_geometric_brownian(rho0, eta, dt, n_steps, time_method='MI', theta=1)
    t1 = time.time()
    total_timeMI = t1-t0
    print("MIim:",total_timeMI)

    t0 = time.time()
    rhoRK2 = my_geometric_brownian(rho0, eta, dt, n_steps, time_method='RK2')
    t1 = time.time()
    total_timeRK2 = t1-t0
    print("RK2:",total_timeRK2)

    t0 = time.time()
    rhoRK3 = my_geometric_brownian(rho0, eta, dt, n_steps, time_method='RK3')
    t1 = time.time()
    total_timeRK3 = t1-t0
    print("RK3:",total_timeRK3)

    
    #Weak and Strong Errors
    n_traj=5000
    def compute_weak_error(approx_sol, exact_sol): #approx_sol=[dim1] and exact_sol=[dim2]
        return  abs(np.mean(approx_sol) - np.mean(exact_sol))
    
    def compute_strong_error(approx_sol, exact_sol): #approx_sol=[dim] and exact_sol=[dim]
        return  np.mean(np.abs(approx_sol-exact_sol))
    
    dt_vec=2.0**(-np.asarray([3,4,5,6,7]))

    weak_errorEM1=np.zeros(len(dt_vec))
    weak_errorEM2=np.zeros(len(dt_vec))
    weak_errorEM3=np.zeros(len(dt_vec))
    weak_errorMI1=np.zeros(len(dt_vec))
    weak_errorMI2=np.zeros(len(dt_vec))
    weak_errorMI3=np.zeros(len(dt_vec))
    weak_errorRK2=np.zeros(len(dt_vec))
    weak_errorRK3=np.zeros(len(dt_vec))
    
    strong_errorEM1=np.zeros(len(dt_vec))
    strong_errorEM2=np.zeros(len(dt_vec))
    strong_errorEM3=np.zeros(len(dt_vec))
    strong_errorMI1=np.zeros(len(dt_vec))
    strong_errorMI2=np.zeros(len(dt_vec))
    strong_errorMI3=np.zeros(len(dt_vec))
    strong_errorRK2=np.zeros(len(dt_vec))
    strong_errorRK3=np.zeros(len(dt_vec))
    
    for i_dt in range(len(dt_vec)):
        i_steps=int(1/dt_vec[i_dt])
        eta=np.random.randn(n,i_steps,n_traj)
        W= np.cumsum(np.sqrt(dt_vec[i_dt])*eta,axis=1)
        rhoTheory= rho0[0] * np.exp( value_mu*i_steps*dt_vec[i_dt] - 0.5 * (value_sigma**2)*i_steps*dt_vec[i_dt] +  value_sigma*W )
#        rhoTheory_mean= rho0[0] * np.exp( value_mu*i_steps*dt_vec[i_dt])
        my_geometric_brownian= geometric_brownian2D(value_mu, value_sigma)
        
        rhoEM1_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='EM', theta=0, n_traj=n_traj)
        rhoEM2_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='EM', theta=0.5, n_traj=n_traj)
        rhoEM3_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='EM', theta=1, n_traj=n_traj)
        rhoMI1_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='MI', theta=0, n_traj=n_traj)
        rhoMI2_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='MI', theta=0.5, n_traj=n_traj)
        rhoMI3_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='MI', theta=1, n_traj=n_traj)
        rhoRK2_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='RK2', theta=1, n_traj=n_traj)
#        rhoRK3_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='RK3', theta=1, n_traj=n_traj)

        weak_errorEM1[i_dt]=compute_weak_error(rhoEM1_conv[0,-1,:],rhoTheory[0,-1,:])
        weak_errorEM2[i_dt]=compute_weak_error(rhoEM2_conv[0,-1,:],rhoTheory[0,-1,:])
        weak_errorEM3[i_dt]=compute_weak_error(rhoEM3_conv[0,-1,:],rhoTheory[0,-1,:])
        weak_errorMI1[i_dt]=compute_weak_error(rhoMI1_conv[0,-1,:],rhoTheory[0,-1,:])
        weak_errorMI2[i_dt]=compute_weak_error(rhoMI2_conv[0,-1,:],rhoTheory[0,-1,:])
        weak_errorMI3[i_dt]=compute_weak_error(rhoMI3_conv[0,-1,:],rhoTheory[0,-1,:])
        weak_errorRK2[i_dt]=compute_weak_error(rhoRK2_conv[0,-1,:],rhoTheory[0,-1,:])
#        weak_errorRK3[i_dt]=compute_weak_error(rhoRK3_conv[0,-1,:],rhoTheory[0,-1,:])

        strong_errorEM1[i_dt]=compute_strong_error(rhoEM1_conv[0,-1,:],rhoTheory[0,-1,:])
        strong_errorEM2[i_dt]=compute_strong_error(rhoEM2_conv[0,-1,:],rhoTheory[0,-1,:])
        strong_errorEM3[i_dt]=compute_strong_error(rhoEM3_conv[0,-1,:],rhoTheory[0,-1,:])
        strong_errorMI1[i_dt]=compute_strong_error(rhoMI1_conv[0,-1,:],rhoTheory[0,-1,:])
        strong_errorMI2[i_dt]=compute_strong_error(rhoMI2_conv[0,-1,:],rhoTheory[0,-1,:])
        strong_errorMI3[i_dt]=compute_strong_error(rhoMI3_conv[0,-1,:],rhoTheory[0,-1,:])
        strong_errorRK2[i_dt]=compute_strong_error(rhoRK2_conv[0,-1,:],rhoTheory[0,-1,:])
#        strong_errorRK3[i_dt]=compute_strong_error(rhoRK3_conv[0,-1,:],rhoTheory[0,-1,:])

#    print(weak_errorEM)
#    print(strong_errorEM)
    #######################################################################################################################
    #    Plots
    #######################################################################################################################
    fig = plt.figure()
    plt.title(r'$\rho(x)$')
    plt.plot(rhoEM[:,-1], ls='-', label="Euler-Maruyama")
    plt.plot(rhoMI[:,-1] , ls='--' ,label="Milstein")
    plt.plot(rhoRK2[:,-1] , ls=':' , label="Runge-Kutta 2")
    #    plt.plot(rhoRK3 , ls=':' , label="Runge-Kutta 3")
    plt.plot(rhoTheory[:,-1] , ls=':', label="Theory")
    plt.legend()
    #    plt.yscale('log')
    #plt.rc('text',usetex=True)
    #plt.rc('font',family='serif')
    plt.xlabel(r'$\rho(x)$')
    plt.ylabel(r'x')
    plt.tight_layout()
    plt.savefig('rho.png')


    fig = plt.figure()
    plt.title(r'$\epsilon_w$')
    plt.plot(dt_vec,weak_errorEM1, ls='-', label="Euler-Maruyama ex")
    plt.plot(dt_vec,weak_errorEM2, ls='-', label="Euler-Maruyama cn")
    plt.plot(dt_vec,weak_errorEM3, ls='-', label="Euler-Maruyama im")
    plt.plot(dt_vec,weak_errorMI1, ls='-', label="Milstein ex")
    plt.plot(dt_vec,weak_errorMI2, ls='-', label="Milstein cn")
    plt.plot(dt_vec,weak_errorMI3, ls='-', label="Milstein im")
    plt.plot(dt_vec,weak_errorRK2, ls='-', label="Runge-Kutta 2")
#    plt.plot(dt_vec,weak_errorRK3, ls='-', label="Runge-Kutta 3")
    plt.plot(dt_vec,0.1*dt_vec, ls='--', label="1 order line")
    plt.plot(dt_vec,0.01*dt_vec**2, ls='--', label="2 order line")
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    #plt.rc('text',usetex=True)
    #plt.rc('font',family='serif')
    plt.xlabel(r'$\epsilon_w$')
    plt.ylabel(r'$\Delta \ t $')
    plt.tight_layout()
    plt.savefig('weak.pdf')




    fig = plt.figure()
    plt.title(r'$\epsilon_s$')
    plt.plot(dt_vec,strong_errorEM1, ls='-', label="Euler-Maruyama ex")
    plt.plot(dt_vec,strong_errorEM2, ls='-', label="Euler-Maruyama cn")
    plt.plot(dt_vec,strong_errorEM3, ls='-', label="Euler-Maruyama im")
    plt.plot(dt_vec,strong_errorMI1, ls='-', label="Milstein ex")
    plt.plot(dt_vec,strong_errorMI2, ls='-', label="Milstein cn")
    plt.plot(dt_vec,strong_errorMI3, ls='-', label="Milstein im")
    plt.plot(dt_vec,strong_errorRK2, ls='-', label="Runge-Kutta 2")
#    plt.plot(dt_vec,strong_errorRK3, ls='-', label="Runge-Kutta 3")
    plt.plot(dt_vec,0.1*dt_vec, ls='--', label="1 order line")
    plt.plot(dt_vec,0.1*dt_vec**(1/2), ls='--', label="0.5 order line")
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    #plt.rc('text',usetex=True)
    #plt.rc('font',family='serif')
    plt.xlabel(r'$\epsilon_s$')
    plt.ylabel(r'$\Delta \ t $')
    plt.tight_layout()
    plt.savefig('strong.pdf')

    
    

    
    
    
