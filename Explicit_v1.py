import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from   scipy.sparse.linalg import spsolve, cg, bicgstab, spilu, LinearOperator
import math

# Initial settings
dt = 0.000025
tmax = 0.1
timesteps = int(tmax/dt)      # number of time steps
gridsteps = 100          # number of steps in the space direction
Da   = 1000        # Damkohler number (used to change k), basically no effect below 100 from my testing


# Create the arrays for all space and time steps
x = np.linspace(0, 1, gridsteps)
dl = x[1]-x[0]   # define dl, just step between first and second gridpoint 
t = np.linspace(0, tmax, timesteps+1)
Phi_A = np.zeros((len(t),len(x)))
Phi_B = np.zeros((len(t),len(x)))

#Analyticalt, Analyticalx, AnalyticalT = determineT_analytical()

def determinePhi_exp(x,time, dt, dl, Da, Phi_A, Phi_B ):
    
    #set initial boundary conditions, j = 0 cannot be in loop
    Phi_A[0,0] = 1 
    Phi_B[0,len(x)-1] = 1 
    
    # time loop over desired number of timesteps
    for j in range (1, int(time/dt)):
        for i in range(len(x)):
            if i == 0:
                Phi_A[:,i] = 1                        # boundary condition y = 0 : Phi_A = 1
            elif i == len(x)-1:
                Phi_B[:,i] = 1                       # boundary condition y = L: Phi_B = 1
            else:
                Phi_A[j+1,i] = (2*dt/(3*dl**2))*(Phi_A[j,i+1]-2*Phi_A[j,i]+Phi_A[j,i-1]) +4*Phi_A[j,i]/3-Phi_A[j-1,i]/3 - 2*dt*Da*Phi_A[j,i]*Phi_B[j,i]/3
                Phi_B[j+1,i] = (2*dt/(3*dl**2))*(Phi_B[j,i+1]-2*Phi_B[j,i]+Phi_B[j,i-1]) +4*Phi_B[j,i]/3-Phi_B[j-1,i]/3 - 2*dt*Da*Phi_B[j,i]*Phi_A[j,i]/3
    return Phi_A,Phi_B

#call function 
Phi_A, Phi_B = determinePhi_exp(x, tmax, dt, dl, Da, Phi_A, Phi_B) 

#plotting
plt.figure(figsize=(12,8))
for j in range(0,len(t),500):
    plt.plot(x, Phi_A[j,:], label=f'A, t={t[j]:.2f}')
    plt.plot(x, Phi_B[j,:], '--', label=f'B, t={t[j]:.2f}')


plt.xlabel('x')
plt.ylabel('Phi')
plt.legend(fontsize=8, ncol = 2)
plt.show()