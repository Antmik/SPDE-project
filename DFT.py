import sys, os, shutil
LibPath = os.environ['SPDE']
sys.path.append(LibPath)

#!/usr/bin/env python
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.signal import argrelextrema
import sympy as sp
from scipy.optimize import fsolve


class DFT(object):

    def __init__(self, functional, functional_derivative, x):
        self.functional=functional
        self.functional_derivative=functional_derivative
        self.x=x
        self.dim= x.shape[0]
    


    def coexistance(self,temp,rho1=0.001*np.ones(self.dim),rho2=0.8*np.ones(self.dim),mu=0.0*np.ones(self.dim)):
        
        def system_coexistance(self,p):
            rhoV,rhoL,mu = p
            eqn1= self.functional_derivative(rhoV,self.x) - mu
            eqn2= self.functional_derivative(rhoL,self.x) - mu
            eqn3= self.functional(rhoV,self.x) - mu*rhoV - ( self.functional(rhoL,self.x) - mu*rhoL )
            return (eqn1,eqn2,eqn3)
        
        sol,infodict,ier,mesg =fsolve(self.system_coexistance, (rho1,rho2,mu), full_output=True, xtol=1e-10, maxfev=10000, factor=0.1)
        
        if (ier!=1):
            print('No coexisting densities found for this T')
            return 0
        
        else:
            self.rhoV,self.rhoL,self.mu = sol

            if(figure):
                #Plot
                rho=np.linspace(0.001,0.8,3000)
                F= self.fHS(rho) - (self.mu)*rho/self.K/self.temp + self.U_integral()/self.K/self.temp * rho**2
                #F= self.fHS(rho) - (-2.5)*rho/self.K/self.temp + self.U_integral()/self.K/self.temp * rho**2

                fig = plt.figure(1)
                plt.title(r'$F[\rho]$')
                plt.plot(rho,F)
                plt.xlabel(r'$\rho(x)$')
                plt.ylabel(r'$F[\rho]$')
                ##plt.ylim(ymax=0.988)
                ##plt.ylim(ymin=1)
                plt.pause(0.0001)
                plt.tight_layout()
                plt.savefig('Free energy density.png')
    #            plt.close()


                dF_drho= self.fHS_prime(rho) - (self.mu)/self.K/self.temp + 2*self.U_integral()/self.K/self.temp * rho
                fig = plt.figure(2)
                plt.title(r'$\delta F[\rho] / \delta \rho$')
                plt.plot(rho,dF_drho)
                plt.xlabel(r'$\rho(x)$')
                plt.ylabel(r'$\delta F[\rho] / \delta \rho$')
                ##plt.ylim(ymax=0.988)
                ##plt.ylim(ymin=1)
                plt.pause(0.0001)
                plt.tight_layout()
                plt.savefig('Free energy density derivative.png')
    #            plt.close()

            return self.rhoV,self.rhoL,self.mu


    def phase_diagram(self, T_start=0.3, T_end=0.45):
        T=np.linspace(T_start, T_end,50)
        rhoV_vec=np.zeros(len(T))
        rhoL_vec=np.zeros(len(T))
        mu_vec=np.zeros(len(T))
        
        for i in range(len(T)):
            if ( np.isscalar (self.coexistance(T[i])) ==False):
                if ( i!= 0 and rhoL_vec[i-1]!=0 and rhoL_vec[i-1]!=rhoV_vec[i-1]):
                    rhoV_vec[i],rhoL_vec[i], mu_vec[i] = self.coexistance(T[i],rhoV_vec[i-1],rhoL_vec[i-1], mu_vec[i-1] )
                else:
                    rhoV_vec[i],rhoL_vec[i], mu_vec[i] = self.coexistance(T[i])
        #Plot
            
        fig = plt.figure()
        plt.title(r'$F[\rho]$')
        plt.scatter(rhoV_vec[rhoV_vec>0],T[rhoV_vec>0])       
        plt.scatter(rhoL_vec[rhoV_vec>0],T[rhoV_vec>0])
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$T$')
        ##plt.ylim(ymax=0.988)
        ##plt.ylim(ymin=1)
        plt.pause(0.0001)
        plt.tight_layout()
        plt.savefig('Phase diagram.png')       

        return 0


    def convolution_matrix(self, x, dx, F_cut):
        #periodic
##        F_cut=10.0
        n=len(x)
        L= x[-1]-x[0]+dx
        W=np.zeros((n,n))
        for j in range(n):
            dist= x-x[j]

            if (self.U_type =='LJ'):
                W[j,:]= dx*1/2* (self.U_LJ(dist+0.5*dx) + self.U_LJ(dist-0.5*dx) )
            elif (self.U_type =='WF'):
                W[j,:]= dx*1/2* (self.U_WF(dist+0.5*dx) + self.U_WF(dist-0.5*dx) )


##            for i in range(n):
##                if( i == j ):
##                    dist=0
##                    W[i,j]= 0 #-epsilon*dx
##                elif (abs(x[i]-x[j])<=F_cut):
##                    
##                    W[i,j]= dx*1/2*( w_approx(dist+0.5*dx,r_star,F_cut,epsilon,alpha) + w_approx(dist-0.5*dx,r_star,F_cut,epsilon,alpha))
####                elif (abs(x[i]+L-x[j])<=F_cut):
    ##                dist= x[i]+L-x[j]
    ##                #W[i]= dx*1/2*( w_approx(dist+0.5*dx,r_star,F_cut,sigma,epsilon,alpha) + w_approx(dist-0.5*dx,r_star,F_cut,sigma,epsilon,alpha))
    ##            elif (abs(x[i]-L-x[j])<=F_cut):
    ##                dist= x[i]-L-x[j]
    ##                #W[i]= dx*1/2*( w_approx(dist+0.5*dx,r_star,F_cut,sigma,epsilon,alpha) + w_approx(dist-0.5*dx,r_star,F_cut,sigma,epsilon,alpha))

############################################################################################################
if __name__ == '__main__':


    bulk=EOS('CS','LJ',1.0)

    plt.figure()
    r=np.linspace(0,5,1000)
    force_approx=-np.diff(bulk.U_WF(r))/(r[2]-r[1])
    plt.plot(bulk.force_WF(r) )
    plt.plot(force_approx)
    plt.ylim(-10,5)
##    plt.show()
    plt.close()

#    r_try=np.array(1.1011011)
#    print(bulk.U_WF(r_try))
#    print(bulk.force_WF(r_try))

    v,l,mu=bulk.coexistance(0.8)
    bulk.phase_diagram(0.5,0.8)
    print(v,l,mu)
#            plt.close()

##    def eq_prime(x):
##        return x**3 - x +1/5
##    
##    def eq(x):
##        return 1/4*x**4 -1/2*x**2 + x/5 
##        
##    def equations(p):
##        #x,y,z = p
##        x,y=p
##        eqn1= eq(x) #-z
##        eqn2= eq(y) #-z #- (1/4*x**4 -1/2*x**2)
##        #eqn3= 1/4*x**4 -1/2*x**2 + x/5 - z*x - (1/4*y**4 -1/2*y**2 + y/5 -z*y )
##        return (eqn1,eqn2)
##
##    #if (fsolve(equations, (-2,2,0)).ier!=1 ):
##    x,infodict,ier,mesg =fsolve(equations, (-0,0), full_output=True)
##    print(ier)
