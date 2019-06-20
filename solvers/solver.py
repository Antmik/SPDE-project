import sys, os, shutil
LibPath = os.environ['SPDE']
sys.path.append(LibPath)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from EOS import EOS
from time_integrator import time_integrator


class fddft(object):

    def __init__(self,space_method='FD', time_method='EM', stochastic_par=0 ,temp=1.0, gamma=1.0, m=1.0, K=1.0):
        """Return a new object."""

        if (space_method!='FO' and space_method!='CD' and space_method !='PR'): # FO=forward , CD=centered difference, PR=parabolic
            raise TypeError('Error in the parameter: space_method')
        else:
            self.space_method = space_method


        if (time_method!='EM' and time_method!='MI' time_method!='RK'): # EM= Euler Maruyama, MI= Milstein, RK=Runge-Kutta
            raise TypeError('Error in the parameter: time_method')
        else:
            self.time_method = time_method

        if (stochastic_par!=0 and stochastic_par!=1): # 0=deterministic and 1=stochastic
            raise TypeError('Error in the parameter: stochastic_par ')
        else:
            self.stochastic_par = stochastic_par
        
        self.temp = temp
        self.gamma = gamma
        self.m = m
        self.K = K



###################################################################################################################
#    CREATE GEOMETRY
###################################################################################################################
    def create_geometry(self, n=50, box=[0,10], boundaries='periodic'):

        #checks on n
        n_new=np.atleast_1d(np.squeeze(np.array(n,dtype=int)))
        self.dim = len(n_new) #dimensionality of the problem
        print('Dimension=%d'%self.dim)
        if (self.dim==1):
            self.nx=n_new[0]
            print('nx=%d'%self.nx)
        elif(self.dim==2):
            self.nx=n_new[0]
            self.ny=n_new[1]
            print('nx=%d and ny=%d'%(self.nx,self.ny) )
        else:
            raise TypeError('Error in the parameter: n --too much dimentions or wrong inputs')

        #checks on box
        if len(np.atleast_1d(box))== 2*self.dim :

            if (self.dim==1 and box[1]>box[0] ):
                self.x=np.linspace(box[0],box[1],self.nx)
                self.dx= (box[1]-box[0])/(self.nx)
                
            elif(self.dim==2 and box[1]>box[0] and box[3]>box[2] ):
                x_tmp = np.linspace(box[0], box[1], self.nx)
                y_tmp = np.linspace(box[2], box[3], self.ny)
                self.x, self.y = np.meshgrid(x_tmp, y_tmp)
                self.dx= (box[1]-box[0])/(self.nx)
                self.dy= (box[3]-box[2])/(self.ny)
                
            else:
                raise TypeError('Error in the parameter: box')
        else:
           raise TypeError('Error in the parameter: box')

        
        if (boundaries!='periodic' and boundaries!='wall'): # FD=finite diff, UP=upwind
            raise TypeError('Error in the parameter: boundaries')
        else:
            self.boundaries=boundaries
            print('Boundaries= %s' %self.boundaries)



###################################################################################################################
#    CREATE INITIAL CONDITIONS
###################################################################################################################
    def create_initial_conditions(self, value, shape='constant'):
        if (shape!='constant' and shape!='gaussian'):
            raise TypeError('Error in the parameter: create_initial_conditions: shape')
        else:
            self.shape=shape

        if (self.shape=='constant' and np.isscalar(value)):
            rho = value*np.ones(self.nx)
        else:
            raise TypeError('Error in the parameter: create_initial_conditions')
        
        return rho
  
  
  
########################################################################################
#    SPACE DISCRETIZATION
#######################################################################################
    def space_discretization(self, rho, U):

        #Forward
        if (self.space_method=='FO'):
            pass
        
        #Centered Differences
        if (self.space_method=='CD'):
            
            up=  1/ self.dx*( np.roll(U, -1, 0) - U)
            um=  1/ self.dx*( U - np.roll(U, 1, 0))

            Fp= np.multiply(up, (np.roll(rho, -1, 0)+rho)/2)
            Fm= np.multiply(um, (np.roll(rho, 1, 0)+rho)/2) 

            if (self.boundaries=='wall'):
                Fm[0]=0.0
                Fp[-1]=0.0
            
            mu= 1/self.dx*(Fp-Fm)

            sigma=np.zeros((self.nx,self.nx))

            if (self.stochastic_par==1):
                sigma += 1/ self.dx*np.diag( -np.sqrt( (np.roll(rho, 1, 0)+rho)/2 ), 0)
                sigma += 1/ self.dx*np.diag( np.sqrt( (np.roll(rho[:-1], -1, 0)+rho[:-1])/2 ), 1)
                sigma[self.nx-1,0]= 1/ self.dx * np.sqrt( (rho[0]+rho[-1])/2 )

            
                if (self.boundaries=='wall'):
                    sigma[0,0]=0.0
                    sigma[self.nx-1,0]=0.0

        
        #Parabolic
        if (self.space_method=='PR'):
            pass

        return mu, sigma


###################################################################################################################
#TIME INTEGRATOR
###################################################################################################################
    def time_integrator(self, rho, mu , sigma, dt, eta):
        
        if (self.time_method == 'EM'):            
            rho_new= rho + mu* dt + np.sqrt(dt)*np.matmul(sigma,eta)
            #print(sigma)
            #print(np.transpose(sigma))

        if (self.time_method == 'MI'):
            rho_new= rho + mu* dt + np.sqrt(dt)*np.matmul(sigma,eta)
 
 
        if (self.time_method == 'RK'):
            rho_new= rho + mu* dt + np.sqrt(dt)*np.matmul(sigma,eta)
         
        return rho_new

###################################################################################################################
#STEP
###################################################################################################################
    def step(self, rho, dt_ref):
        
        def noise_partition(eta_short, dt_partition):
            eta_large=np.zeros((self.nx,2**dt_partition))
            for i in range(2**(dt_partition-1)):
                rnd_n=np.sqrt(2*self.K*self.temp*self.m/self.gamma)*np.random.randn(self.nx)
                eta_large[:,2*i]=0.5*eta_short[:,i]+0.5*rnd_n
                eta_large[:,2*i+1]=0.5*eta_short[:,i]-0.5*rnd_n
            return eta_large
        
        dt=dt_ref*2 #because dividing by 2 later

        eta=np.sqrt(2*self.K*self.temp*self.m/self.gamma)*np.random.randn(self.nx,1)
        
        dt_partition=-1
        flag_cfl_false =1
        
        while (flag_cfl_false==1):
            flag_cfl_false =0
            dt = dt / 2
            dt_partition +=1
            rho_tmp=rho.copy()
            integral_tmp=0.0
            
            if dt_partition!=0:
                eta= noise_partition(eta_old, dt_partition)
            eta_old=eta

            for i_dt in range(2**dt_partition):

                U_tmp = self.potential(rho_tmp)

                #discretization in space                           
                mu,sigma= self.space_discretization(rho_tmp, U_tmp )
                
                #Integration in time                
                rho_tmp = self.time_integrator(rho_tmp, mu , sigma, dt, eta[:,i_dt])
                integral_tmp = np.dot(rho_tmp,mu)*self.dx*dt + 0.5*(2*self.K*self.temp*self.m/self.gamma)*np.trace(np.matmul(np.transpose(sigma),sigma))*self.dx*dt

                #check positivity
                if ( any ( [ i < 0 for i in rho_tmp ] )):
                    flag_cfl_false =1
                    break 

        rho1=rho_tmp
        integral += integral_tmp
        U= self.potential(rho1)

        return (rho1, U, integral)


    def run(self, rho_ini, dt_ref , endTime , saveTime=1.0, trajectories=1, verbose='True', figure='True'):

        def evolveTime(rho_var):
            time = 0
            integral=0.0
            U=0.0
            yield rho_var, U, time, integral # return initial conditions as first state in sequence
            
            while(True):
                rho_var, U, integral_tmp= self.step(rho_var, dt_ref)
                time += dt_ref
                integral += integral_tmp
                yield rho_var, U, time, integral

                
        def run_evolve_time(rho_var):
            trajectory = evolveTime(rho_var)
            
            # Burn some time
            time = 0
            while(time < saveTime):
                _ ,_, time, integral = trajectory.__next__()

            #After some time has passed
            rho_var, U, time,integral = trajectory.__next__()
            
            return rho_var, U, integral        

        n_steps= int(endTime/saveTime)
        rhonorm= np.zeros((n_steps,trajectories))
        integral_vec= np.zeros((n_steps,trajectories))

        for i_traj in range(trajectories):
            np.random.seed(12345*i_traj+i_traj)
            U_ini=self.potential(rho_ini)
            rho=rho_ini.copy()
            rhonorm[0,i_traj]=np.sum(rho_ini*rho_ini)*self.dx
            for i_dt in range(1, n_steps):
                rho, U, integral =run_evolve_time(rho)
                rho_sum=rho.sum()/(self.nx)
                rho_std=np.std(rho)
                
                rhonorm[i_dt,i_traj]=np.sum(rho*rho)*self.dx
                integral_vec[i_dt,i_traj] = integral_vec[i_dt-1,i_traj] + integral
                
                if (verbose):
                    print('Time: %d  Average density: %d  Density Std: %d ' %(i_dt*endTime,rho_sum,rho_std))

                if (figure):
                    fig = plt.figure(1)
                    #plt.clf()
                    plt.title(r'$U(x)$')
                    plt.plot(self.x,U_ini)
                    plt.plot(self.x,U)
                    #plt.rc('text',usetex=True)
                    #plt.rc('font',family='serif')
                    plt.xlabel(r'$U(x)$')
                    plt.ylabel(r'y')
                    #plt.show()
                    plt.pause(0.0001)
                    plt.tight_layout()
                    plt.savefig('UTime%d.png'%i_dt)

                    fig2 = plt.figure(2)
                    #plt.clf()
                    plt.title(r'$\rho(x)$')
                    plt.plot(self.x,rho_ini)
                    plt.plot(self.x,rho)
                    #plt.rc('text',usetex=True)
                    #plt.rc('font',family='serif')
                    plt.xlabel(r'$\rho(x)$')
                    plt.ylabel(r'x')
                    #plt.show(block = False)
                    plt.pause(0.0001)
                    plt.tight_layout()
                    plt.savefig('rhoTime%d.png'%i_dt)

        if (trajectories!=1):
##            print(np.shape(rhonorm))
##            print(np.shape(integral_vec))
            norm_mean=np.mean(rhonorm, axis=1)
            integral_mean=np.mean(integral_vec, axis=1)
        else:
            norm_mean=rhonorm
            integral_mean=integral_vec         

        fig3 = plt.figure(3)
        #plt.clf()
        #plt.title(r'$diffrence$')
        plt.plot(0.5*norm_mean-0.5*norm_mean[0])
        plt.plot(integral_mean)
        #plt.rc('text',usetex=True)
        #plt.rc('font',family='serif')
        plt.xlabel(r'$time$')
        plt.ylabel(r'$||\rho^2(x,t)||-||\rho^2(x,0)||')
        #plt.show(block = False)
        #plt.pause(0.0001)
        plt.tight_layout()
        plt.savefig('check.png')
        plt.close()
    
        return rhonorm , integral_vec


    def potential(self, rho):
        ##        return -np.cos(40*(rho-0.2))
        return + np.log(rho) #+ 1/2*np.power( self.x , 2) 

    
##################################################################################
##########################################################################

##def step_upwind(rho, U, n, dx, dt_ref,integral1):
##    dt=dt_ref*2 #because dividing by 2 later
##
##    eta=np.sqrt(2*K*temp*m/gamma)*np.random.randn(n,1)
##    x= np.linspace(-n*dx/2,n*dx/2,n)
##
##    dt_partition=-1
##    flag_cfl_false =1
##    
##    while (flag_cfl_false==1):
##        flag_cfl_false =0
##        dt = dt / 2
##        dt_partition +=1
##        rho_tmp=rho.copy()
##        integral_tmp=0.0
##
##        if dt_partition!=0:
##            eta= noise_partition(eta_old, dt_partition)
##        eta_old=eta
##        #create and save vector of noises in time
##        #eta= 0.5*eta+0.5*np.sqrt(2*K*temp*m/gamma)*np.random.randn(n)
##        for i_dt in range(2**dt_partition):
##
##            U_tmp = potential(rho_tmp,x)# + 1/4*np.power(x,4)-1/2*np.power(x,2)).reshape(n)
##            
##            ##Compute u(j+1/2) and u(j-1/2)
##            up= - 1/ dx*( roll(U_tmp, -1, 0) - U_tmp  + dx/np.sqrt(dt)*( np.divide(eta[:,i_dt],np.sqrt(rho_tmp))) )#check
##            um= - 1/ dx*( U_tmp -roll(U_tmp, 1, 0) + dx/np.sqrt(dt)*( np.divide(roll(eta[:,i_dt], 1, 0),roll(np.sqrt(rho_tmp), 1, 0))))
##
##            upd= - 1/ dx*( roll(U_tmp, -1, 0) - U_tmp  )#check
##            umd= - 1/ dx*( U_tmp -roll(U_tmp, 1, 0) )
##            
##            
##            ##Compute u(j+1/2)+,u(j+1/2)-,u(j-1/2)+ and u(j-1/2)- 
##            upjp=np.maximum(up,np.zeros(n))
##            umjp=np.minimum(up,np.zeros(n))
##            
##            upjm=np.maximum(um,np.zeros(n))
##            umjm=np.minimum(um,np.zeros(n))
##               
##            #a= np.maximum(np.amax(upjp),np.amax(- umjm)) # upjp = u(j+1/2)+  and umjm=  u(j-1/2)-
##            a= np.amax(upjp- umjm)
##
##            if (dt > dx / (2* a)):
##                flag_cfl_false =1
##                break
##            else:
##                Fp= np.multiply(upjp,rho_tmp) + np.multiply(umjp,roll(rho_tmp, -1, 0))
##                Fm= np.multiply(umjm,rho_tmp) + np.multiply(upjm,roll(rho_tmp, 1, 0))
##                
##                #integral_tmp += np.sum(rho_tmp*d_dx(d_dx(U_tmp)*rho_tmp))*dx*dt#+ 0.5*np.sum((d_dx(rho)*d_dx(rho)))*dx*dt
##                #integral_tmp -= np.sum(rho_tmp/dx * (Fp - Fm)) *dx*dt
##                
##                upjpd=np.zeros(n)
##                umjpd=np.zeros(n)
##                umjmd=np.zeros(n)
##                upjmd=np.zeros(n)
##                
##                upjpd[upjp>0]=upd[upjp>0]
##                umjpd[umjp<0]=upd[umjp<0]
##                umjmd[umjm<0]=umd[umjm<0]
##                upjmd[upjm>0]=umd[upjm>0]
##                
##                Fpd=np.multiply(upjpd,rho_tmp) + np.multiply(umjpd,roll(rho_tmp, -1, 0))
##                Fmd=np.multiply(umjmd,rho_tmp) + np.multiply(upjmd,roll(rho_tmp, 1, 0))
##                integral_tmp -= np.sum(rho_tmp/dx * (Fpd - Fmd)) *dx*dt
##                
##                rho_tmp= rho_tmp - dt/dx * (Fp - Fm)# Fp= F(j+1/2) and Fm= F(j-1/2)
##                
##    rho1=rho_tmp
##    U=potential(rho1,x)
##    integral1 += integral_tmp
##
##    return (rho1, U, dt,integral1)


############################################################################################################
def solver(seed,n_steps):
    
    np.random.seed(seed)
    
    # Initial Conditions
    global n,temp,K,gamma,m,dx,verbose
    verbose = False #True
    n = 50
    Kcgs=1.0 #1.38*10**(-16)
    epsilon=1.0# 1.67*10**(-14)
    sigma=1.0#3.66*10**(-8)
    mass=1.0#6.63*10**(-23)
    
    temp=0.0001 #273#*Kcgs/epsilon#0.01
    K=1*Kcgs
    gamma=1.0 #6*3.1415*sigma/2/mass#*sigma**2/epsilon * np.sqrt(epsilon/mass)
    #deltaV=1.96*10**(-16)/n #/sigma**3/n
    m=1*mass #overdumped limit m/gamma->0

    # Parameters describing simulation
    #dx = 0.04 #2.7*10**(-6) #/sigma #box_size / n
    box_size=4.0
    dx= box_size /(n)
    dt_ref = 0.00001#Initial dt
    t_end= 0.001
    #n_steps = 100

    rho= np.zeros(n)
    x= np.linspace(-n*dx/2,n*dx/2,n)

    rho= 0.1/np.sqrt(2*3.1415*0.2)*np.exp(-((x)**2)/2*0.2)
    
    rho_ref=1.0#1.78*10**(-3)#*(sigma**3)/mass
    #rho = rho_ref*np.ones((n,1))#+ np.sqrt(rho_ref/10000000)*np.random.randn(n,1) # rho = uniform with a perturbation in the center
    #rho[20:40,0]=10.0*np.ones(20)#+0.1*np.sin(np.linspace(0,100,n))
    
    #x,y = np.mgrid[:n,:n]
    #Add Perturbation to rho
    #droplet_x, droplet_y = n/2, n/2
    #rr = (x-droplet_x)**4 + (y-droplet_y)**2
    #rho[rr<5**2] = 20.0 # add a perturbation in density surface

    print(gamma,rho_ref, temp, dx)
    
    rho0 = rho
    U = potential(rho,x)#+ 1/4*np.power(x,4)-1/2*np.power(x,2)).reshape(n) #np.linspace(0,1,n)#rho
    U0 = U.copy()
    
    rhonorm=np.zeros(n_steps)
    rhonorm[0]=np.sum(rho0**2)*dx
    integral1=0.0
    integral1_vec=np.zeros(n_steps)
    
    #rho=demo(rho_start, u_start, v_start, n, dt, dt)
    for i_dt in range(1, n_steps):
        rho, U, integral1 =demo(rho , U, n, dx, t_end, dt_ref, integral1)
        rho_sum=rho.sum()/(n)
        rho_std=np.std(rho[:])
        
        rhonorm[i_dt]=np.sum(rho**2)*dx
        
        integral1_vec[i_dt] =integral1 #[i_dt-1] + np.sum(rho*d_dx(d_dx(U)*rho))*dx*t_end
        
        if (verbose):
            print(i_dt*t_end,rho_sum,rho_std)
            
        
        #fig = plt.figure(1)
        #plt.clf()
        #plt.title(r'$U(x)$')
        #plt.plot(U0)
        #plt.plot(U)
        #plt.rc('text',usetex=True)
        #plt.rc('font',family='serif')
        #plt.xlabel(r'$U(x)$')
        #plt.ylabel(r'y')
        #plt.show()
        #plt.pause(0.0001)
        #plt.tight_layout()
        #plt.savefig('UTime%d.png'%i_dt)

        
        # We plot it using matplotlib
        #fig2 = plt.figure(2)
        #plt.clf()
        #plt.title(r'$\rho(x)$')
        #plt.plot(rho0)
        #plt.plot(rho)
        #plt.rc('text',usetex=True)
        #plt.rc('font',family='serif')
        #plt.xlabel(r'$\rho(x)$')
        #plt.ylabel(r'x')
        #plt.show(block = False)
        #plt.pause(0.0001)
        #plt.tight_layout()
        #plt.savefig('rhoTime%d.png'%i_dt)
        
        
    #fig3 = plt.figure(3)
    #plt.clf()
    #plt.title(r'$diffrence$')
    ##plt.plot(0.5*rhonorm-0.5*rhonorm[0])
    #plt.plot(integral1_vec)
    #plt.rc('text',usetex=True)
    #plt.rc('font',family='serif')
    #plt.xlabel(r'$time$')
    #plt.ylabel(r'$||\rho^2(x,t)||-||\rho^2(x,0)||')
    #plt.show(block = False)
    #plt.pause(0.0001)
    #plt.tight_layout()
    #plt.savefig('checkDet.png')
    
    norm_diff= 0.5*rhonorm #-0.5*rhonorm[0]
    return norm_diff, integral1_vec

def main():
    n_traj=100
    n_steps = 30
    norm= np.zeros((n_steps,n_traj))
    integral= np.zeros((n_steps,n_traj))
    for i_traj in range(n_traj):
        norm[:,i_traj],integral[:,i_traj] = solver(i_traj*1234,n_steps)
    
    norm_mean=np.mean(norm, axis=1)
    integral_mean=np.mean(integral, axis=1)

    x_int= np.linspace(-n*dx/2,n*dx/2,n)
    rho0 = (0.1/np.sqrt(2*3.1415*0.2)*np.exp(-((x_int)**2)/2*0.2) )
    rho0_sq= rho0**2
    normzero=0.5*np.trapz(rho0_sq,x=x_int)
    print(normzero)
    #print(integral_mean)
    
    print(norm_mean[-1]-integral_mean[-1]-normzero)
    
    fig2 = plt.figure(2)
    #plt.clf()
    #plt.title(r'$diffrence$')
    plt.plot(norm_mean-norm_mean[0])
    plt.plot(integral_mean)
    #plt.rc('text',usetex=True)
    #plt.rc('font',family='serif')
    plt.xlabel(r'$time$')
    plt.ylabel(r'$||\rho^2(x,t)||-||\rho^2(x,0)||')
    #plt.show(block = False)
    #plt.pause(0.0001)
    plt.tight_layout()
    plt.savefig('check.png')
    
##    fig3 = plt.figure(3)
##    #plt.clf()
##    plt.title(r'$diffrence$')
##    plt.plot(norm_mean-integral_mean- normzero)
##    #plt.rc('text',usetex=True)
##    #plt.rc('font',family='serif')
##    plt.xlabel(r'$time$')
##    plt.ylabel(r'$||\rho^2(x,t)||-||\rho^2(x,0)||')
##    #plt.show(block = False)
##    #plt.pause(0.0001)
##    plt.tight_layout()
##    plt.savefig('check_difference.png')
    
    
if __name__ == '__main__':
    #main()
    
    temp=0.0001
    n=50
    box=[-2,2]
    rho_ave=.1
    dt_ref=0.0001
    endTime=.0002
    
    fddft=fddft(temp=temp,stochastic_par=1)
    fddft.create_geometry(n, box)
    rho= fddft.create_initial_conditions(rho_ave,'constant')
    
##    rho, U, integral = fddft.step(rho, 0.0001)
    
    a,b=fddft.run(rho, dt_ref , endTime, saveTime=0.0001,trajectories=50)

##    bulk=EOS('CS', 'LJ', temp)
##    rho_v, rho_l, mu = bulk.coexistance(temp)
    
    

    
    
    
