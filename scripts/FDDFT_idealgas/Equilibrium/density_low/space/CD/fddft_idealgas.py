###################################################################################################################
# Antonio Russo
# Imperial College London
# Date: 14/10/2028
###################################################################################################################
#######################################################################################################
figWidth = 9.0*0.39
figHeigh = 7.6*0.39
color_vec=['tab:red','tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
line_style=['-','-','-','-']
marker_style=['x','o','+','d']
#######################################################################################################

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
    
    n=25
    n_steps=2000
    n_steps_eq=100
    n_traj=100
    rho_ref=10.
    dx=np.array([50,100,200,400])
    boundaries='periodic'
    space_method='CD'
    time_method='RK2'
    theta=0
    dt= 0.1*np.amin(dx)**2
    figure='True'

    def energy_derivative(rho_vec, x_vec):
        return np.log(rho_vec)
 
    time_vec = np.linspace(0, n_steps*dt, n_steps)
    time_corr = np.linspace(0, n_steps_eq*dt, n_steps_eq)

#    t0 = time.time()
#    rhoEM, total_timeEM = compute_cpu_time( my_fddft(rho0, eta, dt, n_steps,space_method='UW', time_method='EM', theta=0.5, n_traj=n_traj) )
#    t1 = time.time()
#    total_timeEM=t1-t0
#    print("Time:%f" %total_timeEM)


    rho_std_Theory=np.array([])
    rho_std_Sim=np.array([])
    for i_dx in dx:
        
        rho0= rho_ref*np.ones(n) + np.sqrt( (1-1/n) * rho_ref /i_dx ) *np.random.randn(n)
        x=np.linspace(-(n-1)*i_dx/2,(n-1)*i_dx/2,n)
        eta=np.random.randn(n,n_steps,n_traj)
        print(x)
        
        rho_mean=np.zeros((n,n_steps))
        rho_std=np.zeros((n,n_steps))
        
        rho_spacecorr= np.zeros(n)
        rho_timecorr= np.zeros(n_steps_eq)
        rho_spacecorr_Theory= np.zeros(n)
    

        
        #Simulate for i_dx
        my_fddft= fddft(boundaries, x , energy_derivative)
        rho = my_fddft(rho0, eta, dt, n_steps, space_method=space_method, time_method=time_method, theta=theta, n_traj=n_traj)
        
        for i_bin_pos in range(n) :
            for i_step in range(n_steps_eq,n_steps):
                rho_mean[i_bin_pos, i_step] =  np.mean( rho[i_bin_pos,i_step,:])
                rho_std[i_bin_pos, i_step] =  np.std ( rho[i_bin_pos,i_step,:] - rho_mean[i_bin_pos, i_step] )

        rho_std_Sim= np.append(rho_std_Sim, np.mean(rho_std[:,-1]) )
        rho_std_Theory= np.append(rho_std_Theory, np.sqrt( (1-1/n) * rho_ref / (i_dx) ) )
        
        #Compute time correlations at equilibrium
        for i_step in range(n_steps-n_steps_eq,n_steps):
            rho_timecorr[i_step-(n_steps-n_steps_eq)]= np.mean( (rho[:,i_step,:]- rho_ref ) * (rho[:,-n_steps_eq,:]- rho_ref ) ) / np.mean( (rho[:,-n_steps_eq,:]- rho_ref)*(rho[:,-n_steps_eq,:] - rho_ref) )

        #Compute spatial correlations with middle cell
        for i_bin_pos in range (n):
            rho_spacecorr[i_bin_pos]= np.mean( ( rho[i_bin_pos,-1,:]- rho_mean[i_bin_pos,-1]) * (rho[int(n/2),-1,:]-  rho_mean[i_bin_pos,-1] ) )
            rho_spacecorr_Theory[i_bin_pos]=  -1/n * rho_ref / (i_dx)
        rho_spacecorr_Theory[int(n/2)]=  (1-1/n) * rho_ref / (i_dx)
        
        #Compute Structure factor
        SF=structure_factor( np.reshape( rho[:,-n_steps_eq-1:-1,:], (n, n_traj*n_steps_eq) ) , x )
        
        np.savetxt('density_mean%d.txt'%(i_dx), rho_mean)
        np.savetxt('density_std%d.txt'%(i_dx), rho_std)
        np.savetxt('density_corr%d.txt'%(i_dx), rho_timecorr)
        np.savetxt('density_spacecorr%d.txt'%(i_dx), rho_spacecorr)
        np.savetxt('structure_factor%d.txt'%(i_dx), SF)


#######################################################################################################################
#    Plots
###################################################################################################

        if(figure):
            ##########################################################################################
            #density-std relaxation
            #########################################################################################

            fig=plt.figure(num=None, figsize=(figWidth,figHeigh), dpi=300, facecolor='w', edgecolor='k')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            ax1 = plt.subplot()
            ax1.plot( time_vec,np.mean(rho_std, axis= 0) ,lw=.5,ls=line_style[0],marker=marker_style[0],ms=2,mew=.1,mfc='none', color=color_vec[0], label=r"std" )
            ax1.plot( time_vec,np.sqrt( (1-1/n)*rho_ref/i_dx )*np.ones(n_steps) ,lw=.5,ls=line_style[1],marker=marker_style[1],ms=2,mew=.1,mfc='none', color=color_vec[1], label=r"theoretical std" )

            plt.xlabel(r'$y-y_c$', fontsize=12)
            ax1.set_ylabel(r'$ std( \rho(t) )$', fontsize=12)
            ax1.legend(loc='upper right', fontsize=7, edgecolor='k', frameon=False)
            ax1.set_ylim(ymin=0)
            plt.tick_params(axis='both', labelsize=8,color='k' , direction='in')
            plt.locator_params(nbins=6, axis='x')
            plt.locator_params(nbins=4, axis='y')
            fig.tight_layout()
            fig.savefig('density_std%d.pdf'%(i_dx),dpi=fig.dpi)

        ##########################################################################################
        #density-corr
        #########################################################################################

            fig=plt.figure(num=None, figsize=(figWidth,figHeigh), dpi=300, facecolor='w', edgecolor='k')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            
            ax1 = plt.subplot()
            ax1.plot(time_corr, rho_timecorr,lw=.5,ls=line_style[0],marker=marker_style[0],ms=2,mew=.1,mfc='none', color=color_vec[0], label=r"fddft" )
            
            plt.xlabel(r'$t$', fontsize=12)
            ax1.set_ylabel(r'$  \langle \delta \rho_i(t) \delta \rho_j(0) \rangle $', fontsize=12)
            ax1.legend(loc='upper right', fontsize=7, edgecolor='k', frameon=False)
            #    ax1.set_ylim(ymin=0)
            plt.tick_params(axis='both', labelsize=8,color='k' , direction='in')
            plt.locator_params(nbins=6, axis='x')
            plt.locator_params(nbins=4, axis='y')
            fig.tight_layout()
            fig.savefig('density_corr%d.pdf'%(i_dx),dpi=fig.dpi)
            #plt.show()
            
            ##########################################################################################
            #density-corr
            #########################################################################################
            
            fig=plt.figure(num=None, figsize=(figWidth,figHeigh), dpi=300, facecolor='w', edgecolor='k')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            
            ax1 = plt.subplot()
            ax1.plot(x, rho_spacecorr,lw=.5,ls=line_style[0],marker=marker_style[0],ms=2,mew=.1,mfc='none', color=color_vec[0], label=r"fddft" )
            ax1.plot(x, rho_spacecorr_Theory,lw=.5,ls=line_style[1],marker=marker_style[1],ms=2,mew=.1,mfc='none', color=color_vec[1], label=r"theory" )

            plt.xlabel(r'$x$', fontsize=12)
            ax1.set_ylabel(r'$  \langle \delta \rho_i(t) \delta \rho_j(t) \rangle $', fontsize=12)
            
            ax1.legend(loc='upper right', fontsize=7, edgecolor='k', frameon=False)
            #    ax1.set_ylim(ymin=0)
            plt.tick_params(axis='both', labelsize=8,color='k' , direction='in')
            plt.locator_params(nbins=6, axis='x')
            plt.locator_params(nbins=4, axis='y')
            fig.tight_layout()
            fig.savefig('density_spacecorr%d.pdf'%(i_dx),dpi=fig.dpi)
        #plt.show()



##########################################################################################
#density-std relaxation
#########################################################################################
    fig=plt.figure(num=None, figsize=(figWidth,figHeigh), dpi=300, facecolor='w', edgecolor='k')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax1 = plt.subplot()
    ax1.plot( dx, rho_std_Sim ,lw=.5,ls=line_style[0],marker=marker_style[0],ms=2,mew=.1,mfc='none', color=color_vec[0], label=r"std" )
    ax1.plot( dx, rho_std_Theory ,lw=.5,ls=line_style[1],marker=marker_style[1],ms=2,mew=.1,mfc='none', color=color_vec[1], label=r"theoretical std" )

    plt.xlabel(r'$y-y_c$', fontsize=12)
    ax1.set_ylabel(r'$ std( \rho(t) )$', fontsize=12)
    ax1.legend(loc='upper right', fontsize=7, edgecolor='k', frameon=False)
    ax1.set_ylim(ymin=0)
    plt.tick_params(axis='both', labelsize=8,color='k' , direction='in')
    plt.locator_params(nbins=6, axis='x')
    plt.locator_params(nbins=4, axis='y')
    fig.tight_layout()
    fig.savefig('density_std.pdf',dpi=fig.dpi)
