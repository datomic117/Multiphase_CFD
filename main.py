import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from   scipy.sparse.linalg import spsolve, cg, bicgstab, spilu, LinearOperator
import math

def determinePhi_expl(gridsteps, t_end, dt, Da):

    """Determine the solution of two concentration matrices of both species at 
    every gridpoint in space and time. This is an explicit method.
    The differential equation was dimensionalised with a second order backward scheme for the
    derivative in time and with a second-order scheme for the derivative in space.

    :param int gridsteps: the number of gridsteps for the 1D problem
    :param float t_end: the total span of time from the beginning
    :param float dt: represents the step in Fourier number - dimensionless timestep
    :param float Da: the Damköhler number - dimensionless diffusion coefficient/dimensionless reaction rate coefficient
    :return: the space mesh, time mesh, the concetration 2D arrays of A, B
    :rtype: tuple(np.array(1D), np.array(1D), np.array(2D), np.array(2D))
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, t_end, dt is nonpositive 
    :raises ValueError: if Da is negative
    
    """

    #Define ValueErrors
    if not isinstance(gridsteps, (int, np.integer)):
        print ("Arg gridsteps has to be an integer!")
        raise ValueError

    if (gridsteps <= 0) or (t_end <= 0) or (dt <= 0):
        print ("Args gridsteps, t_end, dt have to be larger than zero!")
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



def determinePhi_impl(gridsteps, t_end, dt, Da):

    """Determine the solution of two concentration matrices of both species at
    every gridpoint in space and time. This is an implicit method.
    The differential equation was dimensionalised with a second-order backward
    (BDF2) scheme for the derivative in time and with a second-order scheme for
    the derivative in space. The reaction term is linearised as
    R = Da*Phi_A_new*Phi_B_old and is the same in both equations. The first time
    step uses backward Euler, since BDF2 needs two old time levels.

    Every time step solves a tridiagonal system M*Phi_new = b for each species:
        subdiag  : -2F                      (F = dt/dl**2)
        diag     : 3 + 4F (+ 2*dt*Da*Phi_B[n] for A only)
        superdiag: -2F
        b        : 4*Phi[n] - Phi[n-1]      (for B also - 2*dt*R)
    with the first and last row replaced by the boundary conditions.
    A is solved first (R implicit in A), then B is solved with R as a known sink.

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
        print ("Arg gridsteps has to be an integer!")
        raise ValueError

    if (gridsteps <= 0) or (t_end <= 0) or (dt <= 0):
        print ("Args gridsteps, t_end, dt have to be larger than zero!")
        raise ValueError

    if (Da < 0):
        print("Arg Da cannot be negative")
        raise ValueError


    # Create the arrays for all space and time steps
    x = np.linspace(0, 1, gridsteps) #create coordinate mesh
    dl = x[1]-x[0] # define dl, just step between first and second gridpoint
    timesteps = int(t_end/dt) # number of time steps
    t = np.linspace(0, t_end, timesteps+1) #create time mesh
    F = dt/dl**2 # grid Fourier number, strength of the coupling between neighbours

    #Initialise concentraton arrays, rows are time, columns are space
    Phi_A = np.zeros((len(t),len(x)))
    Phi_B = np.zeros((len(t),len(x)))

    #Set initial boundary conditions
    Phi_A[0,0] = 1
    Phi_B[0,len(x)-1] = 1

    def solve_step(rhs, k_impl, c0, s, bc_left, bc_right):
        """Assemble the tridiagonal matrix of one species for one time step and solve it.
        c0 and s select the time discretisation scheme: backward Euler (c0=1, s=1) or BDF2 (c0=3, s=2).
        This will make comparing the different shcemes easier.
        k_impl is the array of implicit reaction coefficients (zeros if the reaction is put in rhs instead).
        """
        diag = c0 + 2*s*F + s*dt*k_impl   # new array, the reaction term differs per gridpoint
        sub  = -s*F*np.ones(len(x)-1)
        sup  = -s*F*np.ones(len(x)-1)

        # boundary rows read 1*Phi = bc, so remove their off-diagonal entries
        diag[0] = diag[-1] = 1.0
        sup[0]  = 0.0     # sup[0]  is matrix[0,1]
        sub[-1] = 0.0     # sub[-1] is matrix[N-1,N-2]
        matrix = sps.diags([sub, diag, sup], [-1, 0, 1], format="csc")

        b = rhs.copy()    # copy, so the stored history in Phi is not overwritten
        b[0]  = bc_left
        b[-1] = bc_right
        return spsolve(matrix, b)

    # time loop over desired number of timesteps, every pass fills one whole row n+1
    for n in range(timesteps):
        if n == 0:   # BDF2 needs Phi[n-1], so start with one backward Euler step
            c0, s = 1, 1
            rhs_A, rhs_B = Phi_A[0,:], Phi_B[0,:]
        else:
            c0, s = 3, 2
            rhs_A = 4*Phi_A[n,:] - Phi_A[n-1,:]
            rhs_B = 4*Phi_B[n,:] - Phi_B[n-1,:]

        # The reaction term R = Da*Phi_A_new*Phi_B_old must be identical in both equations,
        # otherwise A and B are not consumed in equal amounts.
        # A: R is linear in Phi_A_new, so it goes into the matrix (implicit).
        Phi_A[n+1,:] = solve_step(rhs_A, Da*Phi_B[n], c0, s, 1, 0)   # A = 1 at x = 0, A = 0 at x = L

        # B: the same R is now known (Phi_A_new is solved), so it moves to the right-hand side as a sink.
        R = Da*Phi_A[n+1,:]*Phi_B[n,:]
        rhs_B = rhs_B - s*dt*R
        Phi_B[n+1,:] = solve_step(rhs_B, np.zeros(len(x)), c0, s, 0, 1)   # B = 0 at x = 0, B = 1 at x = L

    return x, t, Phi_A, Phi_B

def plot(x, t, Phi_A, Phi_B,dt, N_snapshots = 10, ):
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
    #Define ValueError
    if not isinstance(N_snapshots, (int, np.integer)) or (N_snapshots <= 0):
        print("Arg N_snapshots has to be a positive integer larger than 0")
        raise ValueError

    idx = np.linspace(0, len(t)-1, N_snapshots, dtype=int) #snapshots at equally spaced times
    colors = plt.cm.rainbow(np.linspace(0, 1, N_snapshots)) #one color for each snapshot using a colormap

    plt.figure(figsize=(12, 8))
    for j, c in zip(idx, colors):
        plt.plot(x, Phi_A[j, :], color=c, label=f't={t[j]:.3f}')   # A: solid
        plt.plot(x, Phi_B[j, :], '--', color=c)                    # B: dashed

    plt.title("Dimensionless concentration of A (solid) and B (dashed) with dt = {:.6f}".format(dt))
    plt.xlabel('x')
    plt.ylabel('Phi')
    plt.legend(fontsize=8, ncol = 2)
    plt.show()

    return None

#STILL WORKING ON IT
def stability_crit(dt, Da):
    N_steps = 100
    dt_test = np.linspace((0.1*dt), (10*dt), N_steps)
    Da = 0
    for dt in dt_test:
        x, t, Phi_A, Phi_B = determinePhi_expl(gridsteps, tmax, dt, Da)
        if np.max(np.abs(Phi_A)) <= 1 and np.max(np.abs(Phi_B)) <= 1:
            last_stable = (x, t, Phi_A, Phi_B, dt)
            continue
        else:
            first_unstable = (x, t, Phi_A, Phi_B, dt)
            break

    plot(*last_stable[:4])
    plot(*first_unstable[:4])

    return None

#-------------------------------------RUN EXPLICIT SCHEME-----------------------------------------------
dt = 0.000025
tmax = 0.1
gridsteps = 100    # number of steps in the space direction
Da = 0 # Damkohler number (used to change k), basically no effect below 100 from my testing

#-------------------------------------Determine explicit stability in terms of delta_tau----------------
#Turns out tau plays the role of a Fourier number, so it defines stability, through the test shown below
#we found out the criteria is around dt > 1*10**(-4). Run below to see the error blow up.

# stability_crit(dt, Da)


#Showcase implicit function stablity by using bigger and bigger dt and plotting
for dt in [0.000025, 0.0001, 0.001, 0.01]: 
    x,t, Phi_A, Phi_B = determinePhi_impl(gridsteps, tmax, dt, Da)
    plot(x, t, Phi_A, Phi_B,dt)

# Error Checker function: compare each dt and gridsize to the smallest dt and largest gridsize, and see if the error is within a certain tolerance.
def error_checker(dt, gridsteps, tmax, Da, tol = 1e-3):
    """Compares solution with a given dt and gridsteps to a reference solution. The reference solution is the one computed with the \
    smallest dt and the largest gridsteps. Function works for both implicit and explicit methods. It computes the maximum absolute error \
    and graphs it as a function of dt or gridsteps. """

    #reference solution parameters:
    dt = 0.000001
    gridsteps = 1000
    