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
import cmath

import pyximport
pyximport.install(pyimport=False , setup_args={'include_dirs': np.get_include()})

from time_integrator  import time_integrator
from solvers.non_markovian_FH  import non_markovian_FH
from utils.utils  import compute_weak_error, compute_strong_error, compute_cpu_time,structure_factor

######################################################################################
## Test
######################################################################################
if __name__ == '__main__':

    #Data
    n=40
    n_steps=100000
    n_steps_eq=100
    n_traj=1000#0
    rho_ref=1.0
    dx=100
    boundaries='periodic'
    
    A_vec=np.array([1.0])#np.array([1,1])#
    B_vec=np.array([-2.0])#np.array([-1,-4])#

    rho0= rho_ref*np.ones(n) + np.sqrt( (1-1/n) * rho_ref /dx ) *np.random.randn(n)
    mom0= 0*np.ones(n) + np.sqrt( (1-1/n) * rho_ref /dx ) *np.random.randn(n)
    x=np.linspace(-(n-1)*dx/2,(n-1)*dx/2,n)
    
    def energy_derivative(rho_vec, x_vec):
        return np.log(rho_vec)
    
    dt= 0.1#*dx*dx #1* 10**(-3)

    #Initialize fddft class
    my_fh= non_markovian_FH (boundaries, x , energy_derivative, A_vec, B_vec )
    
    
    #Simulate for equilibrium
    eta_eq=np.random.randn(n,n_steps,1)
    a_tmp, b_tmp, _ = my_fh(rho0, mom0, eta_eq, 10*dt, n_steps, n_traj=1)
    
    rho_eq=a_tmp[:,-1,0]
    mom_eq=b_tmp[:,-1,0]
    
    del a_tmp,b_tmp
    #Simulate for n_traj
    
    t0 = time.time()
    eta=np.random.randn(n,n_steps_eq+1000,n_traj)
    rho_tmp, mom_tmp, S = my_fh(rho_eq, mom_eq, eta, dt, n_steps_eq+1000, n_traj=n_traj)
    rho= rho_tmp[:,-n_steps_eq:,:]
    mom= mom_tmp[:,-n_steps_eq:,:]
    
    t1 = time.time()
    total_timeEM=t1-t0
    print("Time:%f" %total_timeEM)

    #Preliminary
    std_rho= np.mean( np.std( rho[:,-1,:] ,axis=1) )
    std_mom= np.mean( np.std( mom[:,-1,:] ,axis=1) )
    
    std_rho_Theory=np.sqrt( (1-1/n) * rho_ref / (dx) )
    std_mom_Theory=np.sqrt(  rho_ref / (dx) ) #(1-1/n) *
    print("Std_rho:%.8f" %std_rho)
    print("Std_rho_theory:%.8f" %std_rho_Theory)
    print("Std_mom:%.8f" %std_mom)
    print("Std_mom_theory:%.8f" %std_mom_Theory)

    #Correlations rho
    rho_mean=np.zeros((n,n_steps_eq))
    rho_std=np.zeros((n,n_steps_eq))
    rho_spacecorr= np.zeros(n)
    rho_timecorr= np.zeros(n_steps_eq)
    rho_spacecorr_Theory= np.zeros(n)

    for i_bin_pos in range(n) :
        for i_step in range(n_steps_eq):
            rho_mean[i_bin_pos, i_step] =  np.mean( rho[i_bin_pos,i_step,:])
            rho_std[i_bin_pos, i_step] =  np.std ( rho[i_bin_pos,i_step,:] - rho_mean[i_bin_pos, i_step] )
    
    #Compute time correlations at equilibrium
    for i_step in range(n_steps_eq):
        for i_bin in range(n) :
            rho_timecorr[i_step] += 1/n* np.mean( (rho[i_bin,i_step,:]- rho_mean[i_bin, i_step] ) * (rho[i_bin,0,:]- rho_mean[i_bin, 0] ) ) / np.mean( (rho[i_bin,0,:]- rho_mean[i_bin, 0] )*(rho[i_bin,0,:] - rho_mean[i_bin, 0] ) )

    #Compute spatial correlations with middle cell
    for i_bin_pos in range (n):
        for i_step in range(n_steps_eq) :
            rho_spacecorr[i_bin_pos] += 1/n_steps_eq * np.mean( ( rho[i_bin_pos,i_step,:]- rho_mean[i_bin_pos,i_step]) * (rho[int(n/2),i_step,:]-  rho_mean[int(n/2),i_step] ) ) #/ np.mean( ( rho[int(n/2),i_step,:]- rho_mean[int(n/2),i_step]) * (rho[int(n/2),i_step,:]-  rho_mean[int(n/2),i_step] ) )
            
        rho_spacecorr_Theory[i_bin_pos]=  -1/n * rho_ref / (dx)
            
    rho_spacecorr_Theory[int(n/2)]=  (1-1/n) * rho_ref / (dx)
#rho_spacecorr_Theory /= rho_spacecorr_Theory[int(n/2)]


    #Correlations mom
    mom_mean=np.zeros((n,n_steps_eq))
    mom_std=np.zeros((n,n_steps_eq))
    mom_spacecorr= np.zeros(n)
    mom_timecorr= np.zeros(n_steps_eq)
    mom_spacecorr_Theory= np.zeros(n)

    for i_bin_pos in range(n) :
        for i_step in range(n_steps_eq):
            mom_mean[i_bin_pos, i_step] =  np.mean( mom [i_bin_pos,i_step,:])
            mom_std[i_bin_pos, i_step] =  np.std ( mom[i_bin_pos,i_step,:] - mom_mean[i_bin_pos, i_step] )
        
    #Compute time correlations at equilibrium
    for i_step in range(n_steps_eq):
        for i_bin in range(n) :
            mom_timecorr[i_step] += 1/n* np.mean( (mom[i_bin,i_step,:]- mom_mean[i_bin, i_step] ) * (mom[i_bin,0,:]- mom_mean[i_bin, 0] ) ) #/ np.mean( (mom[i_bin,0,:]- mom_mean[i_bin, 0] )*(mom[i_bin,0,:] - mom_mean[i_bin, 0] ) )

    #Compute spatial correlations with middle cell
    cell_center= int(n/2)
    for i_bin_pos in range (n):
        for i_step in range(n_steps_eq) :
            mom_spacecorr[i_bin_pos] += 1/n_steps_eq * np.mean( ( mom[i_bin_pos,i_step,:]- mom_mean[i_bin_pos,i_step]) * (mom[cell_center,i_step,:]-  mom_mean[cell_center,i_step] ) ) #/ np.mean( ( mom[int(n/2),i_step,:]- mom_mean[int(n/2),i_step]) * (mom[int(n/2),i_step,:] - mom_mean[int(n/2),i_step] ) )
            
            mom_spacecorr_Theory[i_bin_pos]=  0* rho_ref / (dx) #-1/n *
        
            mom_spacecorr_Theory[int(n/2)]=   rho_ref / (dx) #(1-1/n) *
        #mom_spacecorr_Theory /= rho_spacecorr_Theory[int(n/2)]


#    mom_corr=np.zeros(n_steps_eq)
#    for i_step in range(n_steps-n_steps_eq,n_steps):
#        #autv[i_step,i_dim]=correlation(np.asarray(v[:,0,i_step,i_dim+1]),np.asarray(v[:,0,0,i_dim+1]))
#        mom_corr[i_step-(n_steps-n_steps_eq)]=np.mean( mom[:,i_step,:] * mom[:,0,:] )/np.mean( mom[:,0,:] * mom[:,0,:] )


    #Analytical autocorrelation
    mom_corr_Theory=np.zeros(n_steps_eq,dtype=np.complex_)
    Time=np.linspace(0,dt*n_steps_eq,n_steps_eq)
    if (len(A_vec)==1):
        Omega=cmath.sqrt( np.asscalar(A_vec)- (np.asscalar(B_vec)**2)/4)
        if (Omega==0):
            mom_corr_Theory=  rho_ref / (dx) * np.exp(B_vec*Time/2)*(1-B_vec*Time/2) #(1-1/n) *
        else:
            #mom_corr_Theory=np.exp(B_vec*Time/2)*(np.cos(Omega*Time) - B_vec/(2*Omega)*np.sin(Omega*Time))
            for i_t in range(n_steps_eq):
                mom_corr_Theory[i_t]=  rho_ref / (dx) * cmath.exp(np.asscalar(B_vec) *Time[i_t]/2)*(cmath.cos(Omega*Time[i_t])- np.asscalar(B_vec) /(2*Omega)*cmath.sin(Omega*Time[i_t])) #(1-1/n) *

#SF=structure_factor( np.reshape( rhoEM[:,-n_steps_eq-1:-1,:], (n, n_traj*n_steps_eq) ) , x )
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
#######################################################################################################
figWidth = 8.6*0.39
figHeigh = 6.88*0.39
color_vec=['tab:red','tab:blue', 'tab:green', 'tab:orange', 'tab:cyan' ,'tab:brown', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:olive']
line_style=['-','-','-','-']
marker_style=['x','o','+','d','v','<','>','^']
#######################################################################################################################
#fig = plt.figure()
#plt.title(r'$\rho(x)$')
#plt.plot(rho[:,-1], ls='-', label="Euler-Maruyama")
##    plt.plot(rhoMI[:,-1] , ls='--' ,label="Milstein")
##    plt.plot(rhoRK2[:,-1] , ls=':' , label="Runge-Kutta 2")
##    plt.plot(rhoRK3 , ls=':' , label="Runge-Kutta 3")
##    plt.plot(rhoTheory[:,-1] , ls=':', label="Theory")
#plt.legend()
##    plt.yscale('log')
##plt.rc('text',usetex=True)
##plt.rc('font',family='serif')
#plt.xlabel(r'$\rho(x)$')
#plt.ylabel(r'x')
#plt.tight_layout()
#plt.savefig('rho.png')
#
#
#fig = plt.figure()
#plt.title(r'$j(x)$')
#plt.plot(mom[:,-1], ls='-', label="Euler-Maruyama")
##    plt.plot(rhoMI[:,-1] , ls='--' ,label="Milstein")
##    plt.plot(rhoRK2[:,-1] , ls=':' , label="Runge-Kutta 2")
##    plt.plot(rhoRK3 , ls=':' , label="Runge-Kutta 3")
##    plt.plot(rhoTheory[:,-1] , ls=':', label="Theory")
#plt.legend()
##    plt.yscale('log')
##plt.rc('text',usetex=True)
##plt.rc('font',family='serif')
#plt.xlabel(r'$\rho(x)$')
#plt.ylabel(r'x')
#plt.tight_layout()
#plt.savefig('mom.png')

#######################################################################################################################
#    Plots
#######################################################################################################################

fig=plt.figure(num=None, figsize=(figWidth,figHeigh), dpi=300, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax1 = plt.subplot()
ax1.plot(Time, mom_timecorr ,lw=0,ls=line_style[0],marker=marker_style[0],ms=3,mew=.2,mfc='none', color=color_vec[0], label=r"Simulation" )
ax1.plot(Time,mom_corr_Theory,lw=0.5,ls=line_style[1],marker=marker_style[1],ms=0,mew=.1,mfc='none', color=color_vec[1], label=r"Theory")


plt.xlabel(r'$t$', fontsize=12)
ax1.set_ylabel(r'$  \langle \delta j_j(t) \delta j_j(0) \rangle $', fontsize=12)

ax1.legend(loc='upper right', fontsize=7, edgecolor='k', frameon=False)
#    ax1.set_ylim(ymin=0)
plt.tick_params(axis='both', labelsize=8,color='k' , direction='in')
#    plt.locator_params(nbins=6, axis='x')
#    plt.locator_params(nbins=4, axis='y')
#    plt.ylim(-0.05,1.05)
#    plt.xlim(-0.05,time_corr[max_step]+0.05)
fig.tight_layout()
fig.savefig('momentum_timecorr.pdf',dpi=fig.dpi)
#plt.show()

##########################################################################################
#density-spacecorr
#########################################################################################

fig=plt.figure(num=None, figsize=(figWidth,figHeigh), dpi=300, facecolor='w', edgecolor='k')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


ax1 = plt.subplot()

ax1.plot(x, mom_spacecorr ,lw=0,ls=line_style[0],marker=marker_style[0],ms=4,mew=.2,mfc='none', color=color_vec[0], label=r"Simulation" )
ax1.plot(x, mom_spacecorr_Theory,lw=.5,ls=line_style[0], color=color_vec[1], label=r"Theory" )

plt.xlabel(r'$x$', fontsize=12)
ax1.set_ylabel(r'$  \langle \delta j_i(t) \delta j_j(t) \rangle $', fontsize=12)

ax1.legend(loc='upper right', fontsize=7, edgecolor='k', frameon=False)

#    ax1.set_ylim(ymin=0)
plt.tick_params(axis='both', labelsize=8,color='k' , direction='in')
plt.locator_params(nbins=6, axis='x')
plt.locator_params(nbins=4, axis='y')
fig.tight_layout()
fig.savefig('density_spacecorr.pdf',dpi=fig.dpi)
