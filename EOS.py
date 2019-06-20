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


class EOS(object):
    """
    Attributes:
    """
    
    K=1.0
    sigma=1.0
    epsilon=1.0
    alpha=10.0 #unused fol LJ

    def __init__(self, fHS_approx, U_type, temp):
        """Return a new object."""
        self.fHS_approx = fHS_approx
        self.U_type = U_type
        self.temp=temp
        
        if (fHS_approx !='PY' and fHS_approx !='CS'):
            print('Error in the parameter: fHS_approx')
        if (U_type !='LJ' and U_type!='WF' and U_type!='linear'):
            print('Error in the parameter: U')

    def U_WF(self,r):
        V=r.copy()
        V[r<=self.sigma]= np.inf
        V[r>self.sigma]= 4* self.epsilon / self.alpha/ self.alpha*( (((r[r>self.sigma]/self.sigma)**2)-1)**(-6) - self.alpha*(((r[r>self.sigma]/self.sigma)**2)-1)**(-3))
        return V

    def force_WF(self,r):
        V=r.copy()
        V[r<=self.sigma]= np.inf
        V[r>self.sigma]= 4* self.epsilon / self.alpha/ self.alpha*( 12*r[r>self.sigma]*(( (r[r>self.sigma]/self.sigma)**2)-1)**(-7) - 6*r[r>self.sigma]*self.alpha*(((r[r>self.sigma]/self.sigma)**2)-1)**(-4))
        return V

    def U_LJ(self,r):
        V=r.copy()
#        V= 4*self.epsilon*( (r/self.sigma)**(-12) - (r/self.sigma)**(-6) )
        V= np.pi* ( 0.8*(r**(-10)) - 2*(r**(-4)) )
        return V
    
    def force_LJ(self,r):
        V=r.copy()
        V= 4*self.epsilon*( 12*(r/self.sigma)**(-13) - 6*(r/self.sigma)**(-7) )
        return V

    def U_linear(self,r):
        V=r.copy()
#        V[r<np.sqrt(3)]= 1*( r[r<np.sqrt(3)]-2 )
        V[r<np.sqrt(3)]=-2*np.pi* ( 1/3*((r[r<np.sqrt(3)])**2+1)**(3/2) - (r[r<np.sqrt(3)])**2+1/3)
        V[r>np.sqrt(3)]= 0* r[r>np.sqrt(3)]
        return V
    
# diameterHS
#    BH
#   ...
#
    def BarkerHenderson(self):
        if (self.U_type =='LJ'):
#            r_end= 5.0
#            n_steps = 10000
#            r=np.linspace(0.00000000000000000000000000001,r_end,n_steps)
#            r1 = self.sigma
#            V= self.U_LJ(r)
#            dBH= np.trapz(1-np.exp(-V[r<r1]/ self.K/self.temp),x=r[r<r1])
            dBH=1.0

        if (self.U_type =='linear'):
            dBH=1.0
            
        elif (self.U_type =='WF'):
            r_end= 5.0
            n_steps = 10000
            r=np.linspace(0.0,r_end,n_steps)
            r1= self.sigma * np.sqrt(1+(1/self.alpha)**(1/3))
            V=self.U_WF(r)
            dBH= np.trapz(1-np.exp(-V[r<r1]/ self.K/self.temp),x=r[r<r1])

        #print('dBH = '+ str(dBH))
        return dBH

    def fHS(self,rho):
        dBH= self.BarkerHenderson()
        eta = np.pi/6*rho*(dBH**3)
        if (self.fHS_approx =='PY'):
            return rho*np.log(rho)- rho - rho*np.log(1-eta) +3/2*(2-eta)*eta*rho/(1-eta)**2
        elif (self.fHS_approx =='CS'):
            return rho*np.log(rho)- rho + (4-3*eta)*eta*rho/(1-eta)**2

    def fHS_prime (self,rho):
        dBH= self.BarkerHenderson()
        eta = np.pi/6*rho*(dBH**3)
        if (self.fHS_approx =='PY'):    
            return np.log(rho) - np.log(1-eta) + 1/2 *( 14*eta - 13*eta**2 + 5*eta**3 )/((1-eta)**3) 
        elif (self.fHS_approx =='CS'):
            return np.log(rho) + ( 8*eta - 9*eta**2 + 3*eta**3 )/((1-eta)**3) 

    def U_integral(self):
        try:
            self.w_int
        except:
            r_end= 10.0
            n_steps = 2000
            r=np.linspace(0.0,r_end,n_steps)
            w=np.zeros(n_steps)
            
            if (self.U_type =='LJ'):
                #Minimum of interaction potential
                r_star=  self.sigma #* 2**(1/6)
                V = self.U_LJ(r)
                w[r-r_star >0] = V[r-r_star >0]
                w[r_star-r >=0] = -1.2* np.pi
            
            if (self.U_type =='linear'):
                #Minimum of interaction potential
                r_star=  1 #* 2**(1/6)
                V = self.U_linear(r)
                w[r-r_star >0] = V[r-r_star >0]
                w[r_star-r >=0] = -4/3* np.pi
            
            elif (self.U_type =='WF'):
                #Minimum of interaction potential
                r_star= self.sigma*np.sqrt(1+(2/self.alpha)**(1/3))
                V = self.U_WF(r)

            #pair potential function
            
#            if (self.fHS_approx =='PY'):
#                w[r_star-r >=0] = - self.epsilon
#            elif (self.fHS_approx =='CS'):
##                w[r_star-r >=0] = -4/3* np.pi
#                w[r_star-r >=0] = -1.2* np.pi


#            w_int= 1/2*4*np.pi*np.trapz(w*r*r,x=r)
            w_int= 1/2* 2 *np.trapz(w,x=r)

#            if (self.U_type =='linear'):
#                w_int= 1/2*4*np.pi*(-0.91666666)
            print('w_int = '+ str(w_int))

            #print('r* = '+ str(r_star ))
            
            self.r_star=r_star
            self.w_int= w_int

        return self.w_int

    def system_coexistance(self,p):
        rhoV,rhoL,mu = p
        eqn1= self.fHS_prime(rhoV) - mu/self.K/self.temp + self.U_integral() * 2/self.K/self.temp* rhoV
        eqn2= self.fHS_prime(rhoL) - mu/self.K/self.temp +  self.U_integral() * 2/self.K/self.temp* rhoL
        eqn3= self.fHS(rhoV) - mu*rhoV/self.K/self.temp + self.U_integral()/self.K/self.temp * rhoV**2 - ( self.fHS(rhoL) - mu*rhoL/self.K/self.temp + self.U_integral()/self.K/self.temp* rhoL**2 )
        return (eqn1,eqn2,eqn3)


    
    def coexistance(self,temp,rho1=0.001,rho2=0.6,mu=0.0):
        """Return coexisting densities for a given temperature"""
        self.temp= temp
        sol,infodict,ier,mesg =fsolve(self.system_coexistance, (rho1,rho2,mu), full_output=True, xtol=1e-10, maxfev=10000, factor=0.1)
        
        if (ier!=1):
            print('No coexisting densities found for this T')
            return 0
        
        else:
            self.rhoV= sol[0]
            self.rhoL= sol[1]
            self.mu= sol[2]

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
