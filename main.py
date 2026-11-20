import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from   scipy.sparse.linalg import spsolve, cg, bicgstab, spilu, LinearOperator
import math

def determinePhi_expl(gridsteps, t_end, dt, Da):

    """Determine the solution of two concentration matrices of both species at 
    every gridpoint in space and time. This is an explicit method.
    The differential equation was dimensionalised with a first-order scheme for the
    derivative in time and with a second-order scheme for the derivative in space.

    :param int gridsteps: the number of gridsteps for the 1D problem
    :param float t_end: the total span of time from the beginning
    :param float dt: represents the step in Fourier number - dimensionless timestep
    :param float Da: the Damköhler number - dimensionless diffusion coefficient
    :return: the space mesh, time mesh, the concetration 2D arrays of A, B
    :rtype: tuple(np.array(1D), np.array(1D), np.array(2D), np.array(2D))
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, t_end, dt is nonpositive 
    :raises ValueError: if Da is negative
    
    """

    #Define ValueErrors
    if not isinstance(gridsteps, (int, np.integer)):
        print ("Arg gridteps has to be an integer!")
        raise ValueError

    if (gridsteps <= 0) or (t_end <= 0) or (dt <= 0):
        print ("Args gridsteps, t_end, dt have to be bigger than zero!")
        raise ValueError

    if (Da < 0):
        print("Arg Da cannot be negative")
        raise ValueError

    
    # Create the arrays for all space and time steps
    x = np.linspace(0, 1, gridsteps) #create coordinate mesh
    dl = x[1]-x[0] # define dl, just step between first and second gridpoint 
    timesteps = int(t_end/dt) # number of time steps     
    t = np.linspace(0, t_end, timesteps+1) #create time mesh

    #Initialise concentraton arrays
    Phi_A = np.zeros((len(t),len(x))) 
    Phi_B = np.zeros((len(t),len(x)))
    
    #Set initial boundary conditions, j = 0 cannot be in loop
    Phi_A[0,0] = 1 
    Phi_B[0,len(x)-1] = 1 
    
    # time loop over desired number of timesteps
    for j in range (1, timesteps):
        for i in range(len(x)):
            if i == 0:
                Phi_A[:,i] = 1                        # boundary condition y = 0 : Phi_A = 1
            elif i == len(x)-1:
                Phi_B[:,i] = 1                       # boundary condition y = L: Phi_B = 1
            else:
                Phi_A[j+1,i] = (2*dt/(3*dl**2))*(Phi_A[j,i+1]-2*Phi_A[j,i]+Phi_A[j,i-1]) +4*Phi_A[j,i]/3-Phi_A[j-1,i]/3 - 2*dt*Da*Phi_A[j,i]*Phi_B[j,i]/3
                Phi_B[j+1,i] = (2*dt/(3*dl**2))*(Phi_B[j,i+1]-2*Phi_B[j,i]+Phi_B[j,i-1]) +4*Phi_B[j,i]/3-Phi_B[j-1,i]/3 - 2*dt*Da*Phi_B[j,i]*Phi_A[j,i]/3
    
    return x, t, Phi_A, Phi_B

# def determinePhi_impl(gridsteps,  dt, Da, Told = np.array):
    
#     Tnew = np.zeros_like(gridsteps)
#     matrix = sps.diags([-Fo, 1+2*Fo-CFL, CFL-Fo], [-1, 0, 1], shape=(len(gridsteps),
#     len(gridsteps)))
#     matrix = sps.csr_matrix(matrix)
#     matrix[0,0] = 1.0 # boundary condition y = 0 : T = 0
#     matrix[0,1] = 0.0
#     matrix[len(gridsteps)-1,len(gridsteps)-1] = 1.0 # boundary condition y = L : T = 0
#     matrix[len(gridsteps)-1,len(gridsteps)-2] = 0.0
#     Tnew = sps.linalg.spsolve(matrix, Told)
#     return Tnew

def plot(x, t, Phi_A, Phi_B, N_snapshots = 10):
    """Plots the solution of the two concetration profiles 
    by showing a number of snapshots of the system for different times. It uses
    the result of determinePhi_expl() or determinePhi_impl() and the number of snapshots. 
    :param np.array(1D) x: the space coordinate
    :param np.array(1D) t: the time coordinate
    :param np.array(2D) Phi_A: the concetration profile of species A in space and time
    :param np.array(2D) Phi_B: the concetration profile of species B in space and time
    :param int N_snapshots: the number of snapshots taken from the system, default is 10
    :return: None

    """ 

    idx = np.linspace(0, len(t)-1, N_snapshots, dtype=int) #snapshots at equally spaced times
    colors = plt.cm.rainbow(np.linspace(0, 1, N_snapshots)) #one color for each snapshot using a colormap

    plt.figure(figsize=(12, 8))
    for j, c in zip(idx, colors):
        plt.plot(x, Phi_A[j, :], color=c, label=f't={t[j]:.3f}')   # A: solid
        plt.plot(x, Phi_B[j, :], '--', color=c)                    # B: dashed

    plt.title("Dimensionless concentration of A (solid) and B (dashed)")
    plt.xlabel('x')
    plt.ylabel('Phi')
    plt.legend(fontsize=8, ncol = 2)
    plt.show()

    return None


#-------------------------------------RUN EXPLICIT SCHEME-----------------------------------------------
dt = 0.000025
tmax = 0.1
gridsteps = 100    # number of steps in the space direction
Da = 100 # Damkohler number (used to change k), basically no effect below 100 from my testing

#-------------------------------------Determine explicit stability in terms of delta_tau----------------
#Turns out tau plays the role of a Fourier number, so it defines stability, through the test shown below
#we found out the criteria is around dt > 1*10**(-4). Run below to see the error blow up.

Da = 0 #remove reaction, although it does not matter
#David wants to prove this more eleganly
for coef in list([4.085, 4.1]):
    dt_test = coef*dt
    x, t, Phi_A, Phi_B = determinePhi_expl(gridsteps, tmax, dt_test, Da)
    plot(x, t, Phi_A, Phi_B)


