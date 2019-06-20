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
from solvers.geometric_brownian  import geometric_brownian

figWidth = 8.6*0.39
figHeigh = 6.88*0.39

######################################################################################
## Test
######################################################################################
if __name__ == '__main__':

    #Data
    n=1
    n_steps=1
    n_traj=1
    rho0= np.ones(n) #+ 0.1*np.random.randn(n)
    value_mu=-1.0
    value_sigma=0.5
    dt= 1*2**(-15)
    eta=np.random.randn(n,n_steps,n_traj)
    
    #Time for every method
    W= np.cumsum(np.sqrt(dt)*eta,axis=1)
    rhoTheory= rho0[0] * np.exp( value_mu*n_steps*dt - 0.5 * (value_sigma**2)*n_steps*dt +  value_sigma*W )
    my_geometric_brownian= geometric_brownian(value_mu, value_sigma)

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


    
    #Weak and Strong Errors
    n_traj=10000
    def compute_weak_error(approx_sol, exact_sol): #approx_sol=[dim1] and exact_sol=[dim2]
        return  abs(np.mean(approx_sol) - np.mean(exact_sol))
    
    def compute_strong_error(approx_sol, exact_sol): #approx_sol=[dim] and exact_sol=[dim]
        return  np.mean(np.abs(approx_sol-exact_sol))
    
    dt_vec=2.0**(-np.asarray([5,6,7]))

    weak_errorEM1=np.zeros(len(dt_vec))
    weak_errorEM2=np.zeros(len(dt_vec))
    weak_errorEM3=np.zeros(len(dt_vec))
    weak_errorMI1=np.zeros(len(dt_vec))
    weak_errorMI2=np.zeros(len(dt_vec))
    weak_errorMI3=np.zeros(len(dt_vec))
    weak_errorRK2=np.zeros(len(dt_vec))
    
    strong_errorEM1=np.zeros(len(dt_vec))
    strong_errorEM2=np.zeros(len(dt_vec))
    strong_errorEM3=np.zeros(len(dt_vec))
    strong_errorMI1=np.zeros(len(dt_vec))
    strong_errorMI2=np.zeros(len(dt_vec))
    strong_errorMI3=np.zeros(len(dt_vec))
    strong_errorRK2=np.zeros(len(dt_vec))
    
    i_steps=int(.5/dt)
    eta=np.random.randn(n,i_steps,n_traj)
    my_geometric_brownian= geometric_brownian(value_mu, value_sigma)
    rhoTheory=my_geometric_brownian(rho0, eta, dt , i_steps, time_method='RK2', theta=0, n_traj=n_traj)
    
    for i_dt in range(len(dt_vec)):
        i_steps=int(.5/dt_vec[i_dt])
        eta=np.random.randn(n,i_steps,n_traj)
#        W= np.cumsum(np.sqrt(dt_vec[i_dt])*eta,axis=1)
#        rhoTheory= rho0[0] * np.exp( value_mu*i_steps*dt_vec[i_dt] - 0.5 * (value_sigma**2)*i_steps*dt_vec[i_dt] +  value_sigma*W )
#        rhoTheory_mean= rho0[0] * np.exp( value_mu*i_steps*dt_vec[i_dt])

        my_geometric_brownian= geometric_brownian(value_mu, value_sigma)
        
#        rhoEM1_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='EM', theta=0, n_traj=n_traj)
#        rhoEM2_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='EM', theta=0.5, n_traj=n_traj)
#        rhoEM3_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='EM', theta=1, n_traj=n_traj)
#        rhoMI1_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='MI', theta=0, n_traj=n_traj)
#        rhoMI2_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='MI', theta=0.5, n_traj=n_traj)
#        rhoMI3_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='MI', theta=1, n_traj=n_traj)
        rhoRK2_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='RK2', theta=1, n_traj=n_traj)
#        rhoRK3_conv=my_geometric_brownian(rho0, eta, dt_vec[i_dt], i_steps, time_method='RK3', theta=1, n_traj=n_traj)

#        weak_errorEM1[i_dt]=compute_weak_error(rhoEM1_conv[0,-1,:],rhoTheory[0,-1,:])
#        weak_errorEM2[i_dt]=compute_weak_error(rhoEM2_conv[0,-1,:],rhoTheory[0,-1,:])
#        weak_errorEM3[i_dt]=compute_weak_error(rhoEM3_conv[0,-1,:],rhoTheory[0,-1,:])
#        weak_errorMI1[i_dt]=compute_weak_error(rhoMI1_conv[0,-1,:],rhoTheory[0,-1,:])
#        weak_errorMI2[i_dt]=compute_weak_error(rhoMI2_conv[0,-1,:],rhoTheory[0,-1,:])
#        weak_errorMI3[i_dt]=compute_weak_error(rhoMI3_conv[0,-1,:],rhoTheory[0,-1,:])
        weak_errorRK2[i_dt]=compute_weak_error(rhoRK2_conv[0,-1,:],rhoTheory[0,-1,:])
#        weak_errorRK3[i_dt]=compute_weak_error(rhoRK3_conv[0,-1,:],rhoTheory[0,-1,:])

#        strong_errorEM1[i_dt]=compute_strong_error(rhoEM1_conv[0,-1,:],rhoTheory[0,-1,:])
#        strong_errorEM2[i_dt]=compute_strong_error(rhoEM2_conv[0,-1,:],rhoTheory[0,-1,:])
#        strong_errorEM3[i_dt]=compute_strong_error(rhoEM3_conv[0,-1,:],rhoTheory[0,-1,:])
#        strong_errorMI1[i_dt]=compute_strong_error(rhoMI1_conv[0,-1,:],rhoTheory[0,-1,:])
#        strong_errorMI2[i_dt]=compute_strong_error(rhoMI2_conv[0,-1,:],rhoTheory[0,-1,:])
#        strong_errorMI3[i_dt]=compute_strong_error(rhoMI3_conv[0,-1,:],rhoTheory[0,-1,:])
        strong_errorRK2[i_dt]=compute_strong_error(rhoRK2_conv[0,-1,:],rhoTheory[0,-1,:])
#        strong_errorRK3[i_dt]=compute_strong_error(rhoRK3_conv[0,-1,:],rhoTheory[0,-1,:])

#    print(weak_errorEM)
#    print(strong_errorEM)
    #######################################################################################################################
    #    Plots
    #######################################################################################################################
#    fig = plt.figure()
#    plt.title(r'$\rho(x)$')
#    plt.plot(rhoEM[:,-1], ls='-', label="Euler-Maruyama")
#    plt.plot(rhoMI[:,-1] , ls='--' ,label="Milstein")
#    plt.plot(rhoRK2[:,-1] , ls=':' , label="Runge-Kutta 2")
#    #    plt.plot(rhoRK3 , ls=':' , label="Runge-Kutta 3")
#    plt.plot(rhoTheory[:,-1] , ls=':', label="Theory")
#    plt.legend()
#    #    plt.yscale('log')
#    #plt.rc('text',usetex=True)
#    #plt.rc('font',family='serif')
#    plt.xlabel(r'$\rho(x)$')
#    plt.ylabel(r'x')
#    plt.tight_layout()
#    plt.savefig('rho.png')

    #######################################################################################################
    color_vec=['tab:red',  'tab:orange', 'tab:blue','tab:green', 'tab:cyan' ,'tab:brown', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:olive']
    line_style=['-','--','-.',':']
    marker_style=['x','o','+','d','v','<','>','^']
    
    ##########################################################################################
    #########################################################################################
    
    fig=plt.figure(num=None, figsize=(figWidth,figHeigh), dpi=300, facecolor='w', edgecolor='k')
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

#    plt.plot(dt_vec,weak_errorEM1, lw=.5,ls=line_style[0],marker=marker_style[0],ms=2,mew=.1,mfc='none', color=color_vec[0], label=r"EM $\theta=0$")
#    plt.plot(dt_vec,weak_errorEM2,lw=.5,ls=line_style[0],marker=marker_style[1],ms=2,mew=.1,mfc='none', color=color_vec[1], label=r"EM $\theta=0.5$")
#    plt.plot(dt_vec,weak_errorEM3,lw=.5,ls=line_style[0],marker=marker_style[2],ms=2,mew=.1,mfc='none', color=color_vec[2], label=r"EM $\theta=1$")
#    plt.plot(dt_vec,weak_errorMI1,lw=.5,ls=line_style[0],marker=marker_style[3],ms=2,mew=.1,mfc='none', color=color_vec[3], label=r"MI $\theta=0$")
#    plt.plot(dt_vec,weak_errorMI2,lw=.5,ls=line_style[0],marker=marker_style[4],ms=2,mew=.1,mfc='none', color=color_vec[4], label=r"MI $\theta=0.5$")
#    plt.plot(dt_vec,weak_errorMI3,lw=.5,ls=line_style[0],marker=marker_style[5],ms=2,mew=.1,mfc='none', color=color_vec[5], label=r"MI $\theta=1$")
    plt.plot(dt_vec,weak_errorRK2,lw=.5,ls=line_style[0],marker=marker_style[6],ms=2,mew=.1,mfc='none', color=color_vec[6], label=r"RK")

    plt.plot(dt_vec,0.1*dt_vec ,lw=.5,ls=line_style[1],marker=marker_style[0],ms=0,mew=.1,mfc='none', color=color_vec[7], label=r"order 1")
    plt.plot(dt_vec,0.1*dt_vec**2,lw=.5,ls=line_style[1],marker=marker_style[0],ms=0,mew=.1,mfc='none', color=color_vec[8], label=r"order 2")

    plt.legend(loc='lower right', fontsize=6, edgecolor='k', frameon=False, ncol=2)
    plt.yscale('log')
    plt.xscale('log')
#    plt.ylim(ymin=0.3*10**(-7))
    plt.ylabel(r'$\epsilon_w$', fontsize=12)
    plt.xlabel(r'$\Delta \ t $', fontsize=12)
    plt.tight_layout()
    plt.savefig('weak.pdf')


#########################################################################################


    fig=plt.figure(num=None, figsize=(figWidth,figHeigh), dpi=300, facecolor='w', edgecolor='k')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
#    plt.plot(dt_vec,strong_errorEM1, lw=.5,ls=line_style[0],marker=marker_style[0],ms=2,mew=.1,mfc='none', color=color_vec[0], label=r"EM $\theta=0$")
#    plt.plot(dt_vec,strong_errorEM2,lw=.5,ls=line_style[0],marker=marker_style[1],ms=2,mew=.1,mfc='none', color=color_vec[1], label=r"EM $\theta=0.5$")
#    plt.plot(dt_vec,strong_errorEM3,lw=.5,ls=line_style[0],marker=marker_style[2],ms=2,mew=.1,mfc='none', color=color_vec[2], label=r"EM $\theta=1$")
#    plt.plot(dt_vec,strong_errorMI1,lw=.5,ls=line_style[0],marker=marker_style[3],ms=2,mew=.1,mfc='none', color=color_vec[3], label=r"MI $\theta=0$")
#    plt.plot(dt_vec,strong_errorMI2,lw=.5,ls=line_style[0],marker=marker_style[4],ms=2,mew=.1,mfc='none', color=color_vec[4], label=r"MI $\theta=0.5$")
#    plt.plot(dt_vec,strong_errorMI3,lw=.5,ls=line_style[0],marker=marker_style[5],ms=2,mew=.1,mfc='none', color=color_vec[5], label=r"MI $\theta=1$")
    plt.plot(dt_vec,strong_errorRK2,lw=.5,ls=line_style[0],marker=marker_style[6],ms=2,mew=.1,mfc='none', color=color_vec[6], label=r"RK")
    
    plt.plot(dt_vec,1*dt_vec**(1/2),lw=.5,ls=line_style[1],marker=marker_style[0],ms=0,mew=.1,mfc='none', color=color_vec[7], label=r"order 0.5")
    plt.plot(dt_vec,1*dt_vec,lw=.5,ls=line_style[1],marker=marker_style[0],ms=0,mew=.1,mfc='none', color=color_vec[8], label=r"order 1")
    
    plt.legend(loc='lower right', fontsize=6, edgecolor='k', frameon=False, ncol=2)
    plt.yscale('log')
    plt.xscale('log')

#    plt.ylim(ymin=2*10**(-5))

    plt.ylabel(r'$\epsilon_s$', fontsize=12)
    plt.xlabel(r'$\Delta \ t $', fontsize=12)
    plt.tight_layout()

    plt.savefig('strong.pdf')

    
    

    
    
    
