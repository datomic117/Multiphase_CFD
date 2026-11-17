import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from   scipy.sparse.linalg import spsolve, cg, bicgstab, spilu, LinearOperator
import math

# Initial settings
timesteps = 5001        # number of time steps
tmax      = 50          # end time of the simulation, s
gridsteps = 52          # number of steps in the space direction
Tmax      = 100         # maximum temperature initially
alpha     = 0.01        # thermal diffusivity, cm^2/s
u         = 0.1         # velocity, cm/s
Fo        = alpha*tmax/timesteps*gridsteps**2 # Fourier number
Pe        = u/alpha     # Peclet number
CFL       = u*tmax/timesteps*gridsteps # Courant number

# Create the arrays for all space and time steps
x = np.linspace(0, 1, gridsteps)
t = np.linspace(0, tmax, timesteps)
Phi_A = np.zeros((len(t),len(x)))
Phi_B = np.zeros((len(t,len(x))))
times = np.array([0,0.5,1,2,5,10])

#Analyticalt, Analyticalx, AnalyticalT = determineT_analytical()

def determinePhi_exp(x,time, dt, dl, Da, Phi_A, Phi_B ):
    Phi_A[0,0] = 1 
    Phi_B[0,len(x)-1] = 1 
    for j in range (time/dt):
        for i,x in enumerate(x):
            if i == 0:
                Phi_A[:,i] = 1                        # boundary condition y = 0 : Phi_A = 1
            elif i == len(x)-1:
                Phi_B[:,i] = 1                       # boundary condition y = L: Phi_B = 1
            else:
                Phi_A[j+1,i] = (2*dt/(3*dl**2))*(Phi_A[j,i+1]-2*Phi_A[j,i]+Phi_A[j,i-1]) +4*Phi_A[j,i]-Phi_A[j-1,i] - 2*dt*Da*Phi_A[j,i]*Phi_B[j,i]
                Phi_B[i] = (2*dt/(3*dl**2))*(Phi_B[j,i+1]-2*Phi_B[j,i]+Phi_B[j,i-1]) +4*Phi_B[j,i]-Phi_B[j-1,i] - 2*dt*Da*Phi_B[j,i]*Phi_A[j,i]

            return Phi_A,Phi_B
  

plt.plot(x, T[0,:], 'b.', label='t=0 sec')
plt.plot(x, T[50,:], 'g.', label='t=0.5 sec')
plt.plot(x, T[100,:], 'r.', label='t=1.0 sec')
plt.plot(x, T[200,:], 'c.', label='t=2.0 sec')
plt.plot(x, T[500,:], 'm.', label='t=5.0 sec')
plt.plot(x, T[1000,:], 'k.', label='t=10.0 sec')


plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()