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

        # Ensure boundary condtions are enforced by removing off diagonal entries of first 
        # and last row and setting the diagonal to 1. This works in tandem with making the values
        # of b match the boundary conditions at the two walls. Then the equation becomes 1*Phi = bc
        # so Phi = bc.
        diag[0] = diag[-1] = 1.0
        sup[0]  = 0.0     # sup[0]  is matrix[0,1]
        sub[-1] = 0.0     # sub[-1] is matrix[N-1,N-2]
        matrix = sps.diags([sub, diag, sup], [-1, 0, 1], format="csc")

        #Allow for multiple discretisation schemes
        b = rhs.copy()    # copy, so the stored history in Phi is not overwritten
        b[0]  = bc_left   
        b[-1] = bc_right
        return spsolve(matrix, b) #returns the Phi vector at a single time step

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

def plot(x, t, Phi_A, Phi_B,dt, N_snapshots = 10):
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


def stability_crit(gridsteps, t_end, dt, Da, N_steps=50):
    """Finds the first and last timestep for which the explicit integration scheme blows up. 
    It finds this timestep in the range of the given timestep to 100 times the given timesteps, in N_steps steps.
    Increasing N_steps increases computational cost significantly.
    Value errors are already caught in Determin_Phi_expl.
    
    :param int gridsteps: the number of gridsteps for the 1D problem
    :param float t_end: the total span of time from the beginning
    :param float dt: represents the step in Fourier number - dimensionless timestep
    :param float Da: the Damköhler number - dimensionless diffusion coefficient
    :param int N_steps: the number of steps taken from dt to 100*dt, default is 50.
    :return: None

    """

    dt_test = np.linspace((dt), (100*dt), N_steps)
    for dt in dt_test:
        x, t, Phi_A, Phi_B = determinePhi_expl(gridsteps, t_end, dt, Da)
        if np.max(np.abs(Phi_A)) <= 1 and np.max(np.abs(Phi_B)) <= 1:
            last_stable = (x, t, Phi_A, Phi_B, dt)
            last_dt = dt
            continue
        elif np.max(np.abs(Phi_A)) >= 1 or np.max(np.abs(Phi_B)) >= 1:
            first_unstable = (x, t, Phi_A, Phi_B, dt)
            plot(*last_stable)
            plot(*first_unstable)
            print(f"The last stable timestep = {last_dt:.6f}")
            print(f"The first unstable timestep = {dt:.6f}")
            break
    else:
        print(f"Could not find a timestep for which the function breaks between {dt:.6f} and {100*dt:.6f}")
    return None

# Error Checker function: compare each dt and gridsize to the smallest dt and largest gridsize, and see if the error is within a certain tolerance.
def error_checker_dt( tmax, Da):
    """Compares solution with a given dt to a reference solution. The reference solution is the one computed with the \
    smallest dt and the largest gridsteps. Function works for both implicit and explicit methods. It computes the root mean square error \
    and graphs it as a function of dt """

    #reference solution parameters:
    dt = 0.00001
    gridsteps = 100

    dt_values = [0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001]
    #implicit reference 
    x_ref, t_ref, Phi_A_ref, Phi_B_ref = determinePhi_impl(gridsteps, tmax, dt, Da)

    #explicit reference
    x_ref_exp, t_ref_exp, Phi_A_ref_exp, Phi_B_ref_exp = determinePhi_expl(gridsteps, tmax, dt, Da)

    #implicit time step error check
    errors_A = []
    errors_B = []
    for dt in dt_values:
        x, t, Phi_A, Phi_B = determinePhi_impl(gridsteps, tmax, dt, Da)
        rmse_A = np.sqrt(np.mean((Phi_A[-1,:] - Phi_A_ref[-1,:])**2))
        rmse_B = np.sqrt(np.mean((Phi_B[-1,:] - Phi_B_ref[-1,:])**2))
        errors_A.append(rmse_A)
        errors_B.append(rmse_B)
        print(f"dt: {dt}, RMSE A: {rmse_A:.6e}, RMSE B: {rmse_B:.6e}")

    #explicit time step error check
    errors_A_exp = []  
    errors_B_exp = []
    for dt in dt_values:
        x, t, Phi_A, Phi_B = determinePhi_expl(gridsteps, tmax, dt, Da)
        rmse_A_exp = np.sqrt(np.mean((Phi_A[-1,:] - Phi_A_ref_exp[-1,:])**2))
        rmse_B_exp = np.sqrt(np.mean((Phi_B[-1,:] - Phi_B_ref_exp[-1,:])**2))
        errors_A_exp.append(rmse_A_exp)
        errors_B_exp.append(rmse_B_exp)
        print(f"dt: {dt}, RMSE A (explicit): {rmse_A_exp:.6e}, RMSE B (explicit): {rmse_B_exp:.6e}")

    plt.figure(figsize=(12, 8))
    plt.loglog(dt_values, errors_A, 'o-', label='RMSE A (implicit)')
    plt.loglog(dt_values, errors_B, 'o-', label='RMSE B (implicit)')
    plt.loglog(dt_values, errors_A_exp, 's--', label='RMSE A (explicit)')
    plt.loglog(dt_values, errors_B_exp, 's--', label='RMSE B (explicit)')
    plt.xlabel('dt') 
    plt.ylabel('RMSE')
    plt.title('Root Mean Square Error vs dt')  
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()
         
def error_checker_grid( tmax, Da):
    """Compares solution with a given grid size to a reference solution. The reference solution is the one computed with the \
    smallest dt and the largest gridsteps. Function works for both implicit and explicit methods. It computes the root mean square error \
    and graphs it as a function of grid steps """

    #reference solution parameters:
    dt = 0.00001
    gridsteps = 100

    grid_values = [80,60,40,20]
    #implicit reference 
    x_ref, t_ref, Phi_A_ref, Phi_B_ref = determinePhi_impl(gridsteps, tmax, dt, Da)

    #explicit reference
    x_ref_exp, t_ref_exp, Phi_A_ref_exp, Phi_B_ref_exp = determinePhi_expl(gridsteps, tmax, dt, Da)

    #implicit time step error check
    errors_A = []
    errors_B = []
    for grid in grid_values:
        x, t, Phi_A, Phi_B = determinePhi_impl(grid, tmax, dt, Da)
        Phi_A_interp = np.interp(x, x_ref, Phi_A_ref[-1, :])
        Phi_B_interp = np.interp(x, x_ref, Phi_B_ref[-1, :])
        rmse_A = np.sqrt(np.mean((Phi_A[-1,:] - Phi_A_interp)**2))
        rmse_B = np.sqrt(np.mean((Phi_B[-1,:] - Phi_B_interp)**2))
        errors_A.append(rmse_A)
        errors_B.append(rmse_B)
        print(f"Grid Steps: {grid}, RMSE A: {rmse_A:.6e}, RMSE B: {rmse_B:.6e}")

    #explicit time step error check
    errors_A_exp = []  
    errors_B_exp = []
    for grid in grid_values:
        x, t, Phi_A, Phi_B = determinePhi_expl(grid, tmax, dt, Da)
        Phi_A_interp = np.interp(x, x_ref_exp, Phi_A_ref_exp[-1, :])
        Phi_B_interp = np.interp(x, x_ref_exp, Phi_B_ref_exp[-1, :])
        rmse_A_exp = np.sqrt(np.mean((Phi_A[-1,:] - Phi_A_interp)**2))
        rmse_B_exp = np.sqrt(np.mean((Phi_B[-1,:] - Phi_B_interp)**2))
        errors_A_exp.append(rmse_A_exp)
        errors_B_exp.append(rmse_B_exp)
        print(f"Grid Steps: {grid}, RMSE A (explicit): {rmse_A_exp:.6e}, RMSE B (explicit): {rmse_B_exp:.6e}")

    plt.figure(figsize=(12, 8))
    plt.loglog(grid_values, errors_A, 'o-', label='RMSE A (implicit)')
    plt.loglog(grid_values, errors_B, 'o-', label='RMSE B (implicit)')
    plt.loglog(grid_values, errors_A_exp, 's--', label='RMSE A (explicit)')
    plt.loglog(grid_values, errors_B_exp, 's--', label='RMSE B (explicit)')
    plt.xlabel('Grid Steps') 
    plt.ylabel('RMSE')
    plt.title('Root Mean Square Error vs Grid Steps')  
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()


def main():
    """Entry point: sets the parameters and runs the selected parts of the analysis.
    Switch the flags to choose what is run, since the error checkers are computationally expensive."""

    #-------------------------------------Parameters-----------------------------------------------------
    dt = 0.000025
    tmax = 0.1
    gridsteps = 100    # number of steps in the space direction
    Da = 1000 # Damkohler number (used to change k), basically no effect below 100 from my testing

    #-------------------------------------Choose what to run---------------------------------------------
    run_stability      = True    # find the explicit stability limit
    run_impl_showcase  = False   # show implicit stability for increasing dt
    run_error_dt       = False   # convergence in dt
    run_error_grid     = False   # convergence in grid size

    #-------------------------------------Determine explicit stability in terms of delta_tau-------------
    #Turns out tau plays the role of a Fourier number, so it defines stability, through the test shown below
    #we found out the criteria is around dt > 1*10**(-4). Run below to see the error blow up.
    
    if run_stability:
        stability_crit(gridsteps, tmax, dt, Da)

    #Showcase implicit function stablity by using bigger and bigger dt and plotting
    if run_impl_showcase:
        for dt_i in [0.000025, 0.0001, 0.001, 0.01]:
            x, t, Phi_A, Phi_B = determinePhi_impl(gridsteps, tmax, dt_i, Da)
            plot(x, t, Phi_A, Phi_B, dt_i)

    #Compare each dt and gridsize to the reference solution
    if run_error_dt:
        error_checker_dt(tmax, Da)

    if run_error_grid:
        error_checker_grid(tmax, Da)


if __name__ == "__main__":
    main()