import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from   scipy.sparse.linalg import spsolve, cg, bicgstab, spilu, LinearOperator
import math

def initialize(Tmax, xdata):
    T = [Tmax * x for x in xdata]
    return T

def determineT_FTCS(xdata, Told, Fo, CFL):
    Tnew = np.zeros_like(xdata)
    for i,x in enumerate(xdata):
        if i == 0:
            Tnew[i] = 0                        # boundary condition y = 0 : T = 0
        elif i == len(xdata)-1:
            Tnew[i] = 100                        # boundary condition y = L: T = 0
        else:
            Tnew[i] = (-CFL/2+Fo) * Told[i+1] + (1 - 2*Fo) * Told[i] + (CFL/2+Fo) * Told[i-1]

    return Tnew


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
T = np.zeros((len(t),len(x)))
times = np.array([0,0.5,1,2,5,10])

#Analyticalt, Analyticalx, AnalyticalT = determineT_analytical()

for j, time in enumerate(t):
    if j == 0:
        T[j,:] = initialize (Tmax, x)
    else:
        T[j,:] = determineT_FTCS(x, T[j-1,:], Fo, CFL)      

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