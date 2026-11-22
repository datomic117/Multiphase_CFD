import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from   scipy.sparse.linalg import spsolve, cg, bicgstab, spilu, LinearOperator
import math
from   scipy.linalg.lapack import dgtsv


def check_inputs(gridsteps, t_end, dt, Da):
    """Checks the input arguments shared by the solvers, the stability test and the error checkers.

    :param int gridsteps: the number of gridsteps for the 1D problem
    :param float t_end: the total span of time from the beginning
    :param float dt: dimensionless timestep
    :param float Da: the Damköhler number
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, t_end, dt is nonpositive
    :raises ValueError: if Da is negative
    """
    if not isinstance(gridsteps, (int, np.integer)):
        print ("Arg gridsteps has to be an integer!")
        raise ValueError

    if (gridsteps <= 0) or (t_end <= 0) or (dt <= 0):
        print ("Args gridsteps, t_end, dt have to be larger than zero!")
        raise ValueError

    if (Da < 0):
        print("Arg Da cannot be negative")
        raise ValueError


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
    check_inputs(gridsteps, t_end, dt, Da)


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



def determinePhi_impl(gridsteps, t_end, dt, Da, optimised_solving=True):

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
    :param bool optimised_solving: if True (default) the tridiagonal system is solved directly from its three
        diagonals with the LAPACK routine dgtsv, which is much faster. If False the sparse matrix is assembled
        and solved with spsolve.
    :return: the space mesh, time mesh, the concetration 2D arrays of A, B
    :rtype: tuple(np.array(1D), np.array(1D), np.array(2D), np.array(2D))
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, t_end, dt is nonpositive
    :raises ValueError: if Da is negative
    :raises ValueError: if optimised_solving is not a bool

    """

    #Define ValueErrors
    check_inputs(gridsteps, t_end, dt, Da)

    if not isinstance(optimised_solving, (bool, np.bool_)):
        print("Arg optimised_solving has to be a boolean!")
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

        #Allow for multiple discretisation schemes
        b = rhs.copy()    # copy, so the stored history in Phi is not overwritten
        b[0]  = bc_left
        b[-1] = bc_right

        if optimised_solving:
            # dgtsv solves the tridiagonal system directly from its three diagonals (no matrix is assembled).
            # It overwrites its inputs, which is fine since they are rebuilt every step. [3] is the solution.
            return dgtsv(sub, diag, sup, b)[3]

        matrix = sps.diags([sub, diag, sup], [-1, 0, 1], format="csc")
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


SCHEMES = {"implicit": determinePhi_impl, "explicit": determinePhi_expl}   # name -> solver, shared by the tests below


def stability_crit(gridsteps, t_end, dt, Da, N_steps=50, scheme="explicit"):
    """Finds the first and last timestep for which the chosen integration scheme blows up.
    It finds this timestep in the range of the given timestep to 100 times the given timesteps, in N_steps steps.
    The explicit scheme is only conditionally stable, the implicit scheme is expected to be stable for all timesteps.
    Increasing N_steps increases computational cost significantly.

    :param int gridsteps: the number of gridsteps for the 1D problem
    :param float t_end: the total span of time from the beginning
    :param float dt: represents the step in Fourier number - dimensionless timestep
    :param float Da: the Damköhler number - dimensionless diffusion coefficient
    :param int N_steps: the number of steps taken from dt to 100*dt, default is 50.
    :param str scheme: the scheme to test, "explicit" (default) or "implicit"
    :return: None
    :raises ValueError: if gridsteps or N_steps is not int
    :raises ValueError: if gridsteps, t_end, dt, N_steps is nonpositive
    :raises ValueError: if Da is negative
    :raises ValueError: if scheme is not "explicit" or "implicit"

    """

    #Define ValueErrors
    check_inputs(gridsteps, t_end, dt, Da)

    if not isinstance(N_steps, (int, np.integer)):
        print("Arg N_steps has to be an integer!")
        raise ValueError

    if N_steps <= 0:
        print("Arg N_steps has to be larger than zero!")
        raise ValueError

    if scheme not in SCHEMES:
        print('Arg scheme has to be "explicit" or "implicit"!')
        raise ValueError

    solver = SCHEMES[scheme]
    tol = 1e-9   # margin for rounding error in the bound |Phi| <= 1
    dt_test = np.linspace((dt), (100*dt), N_steps)
    for dt_i in dt_test:
        x, t, Phi_A, Phi_B = solver(gridsteps, t_end, dt_i, Da)
        # the concentrations are bounded by 1, with a margin for rounding error: the implicit solver
        # returns 1.0000000000000007 at the boundary, which is not an instability
        max_A, max_B = np.max(np.abs(Phi_A)), np.max(np.abs(Phi_B))
        if max_A <= 1 + tol and max_B <= 1 + tol:
            last_stable = (x, t, Phi_A, Phi_B, dt_i)
            last_dt = dt_i
            continue
        else:
            first_unstable = (x, t, Phi_A, Phi_B, dt_i)
            plot(*last_stable)
            plot(*first_unstable)
            print(f"The last stable timestep ({scheme}) = {last_dt:.6f}")
            print(f"The first unstable timestep ({scheme}) = {dt_i:.6f}")
            break
    else:
        print(f"Could not find a timestep for which the {scheme} function breaks between {dt:.6f} and {100*dt:.6f}")
    return None


def rmse(profile, profile_ref):
    """Root mean square error between two profiles in space. The reference can be a numerical solution
    or, later, an analytical one, as long as it is evaluated at the same gridpoints.

    :param np.array(1D) profile: the concentration profile to check
    :param np.array(1D) profile_ref: the reference concentration profile at the same gridpoints
    :return: the root mean square error
    :rtype: float
    """
    return np.sqrt(np.mean((profile - profile_ref)**2))


def plot_errors(values, errors, xlabel, title):
    """Plots the errors of the error checkers on a log-log scale, for both species and both schemes.

    :param np.array(1D) values: the values of the parameter that was varied
    :param dict errors: scheme name -> np.array of shape (2, len(values)), row 0 is species A and row 1 is species B
    :param str xlabel: the label of the x axis, the name of the varied parameter
    :param str title: the title of the plot
    :return: None
    """
    styles = {"implicit": 'o-', "explicit": 's--'}
    plt.figure(figsize=(12, 8))
    for scheme, err in errors.items():
        plt.loglog(values, err[0], styles[scheme], label=f'RMSE A ({scheme})')
        plt.loglog(values, err[1], styles[scheme], label=f'RMSE B ({scheme})')
    plt.xlabel(xlabel)
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()


# Error Checker function: compare each dt and gridsize to the smallest dt and largest gridsize, and see if the error is within a certain tolerance.
def error_checker_dt(tmax, Da, dt=0.00001, gridsteps=100):
    """Compares solution with a given dt to a reference solution. The reference solution is the one computed with the \
    smallest dt and the largest gridsteps. Function works for both implicit and explicit methods. It computes the root mean square error \
    and graphs it as a function of dt.

    :param float tmax: the total span of time from the beginning
    :param float Da: the Damköhler number
    :param float dt: timestep of the reference solution, default is 0.00001
    :param int gridsteps: number of gridsteps of the reference solution (also used for all compared solutions), default is 100
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt is nonpositive
    :raises ValueError: if Da is negative
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt, Da)

    dt_values = np.array([0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001])
    errors = {scheme: np.zeros((2, len(dt_values))) for scheme in SCHEMES}   # rows: species A, B

    for scheme, solver in SCHEMES.items():
        #reference solution of this scheme
        x_ref, t_ref, Phi_A_ref, Phi_B_ref = solver(gridsteps, tmax, dt, Da)

        #time step error check
        for i, dt_i in enumerate(dt_values):
            x, t, Phi_A, Phi_B = solver(gridsteps, tmax, dt_i, Da)
            errors[scheme][:, i] = rmse(Phi_A[-1,:], Phi_A_ref[-1,:]), rmse(Phi_B[-1,:], Phi_B_ref[-1,:])
            print(f"dt: {dt_i}, RMSE A ({scheme}): {errors[scheme][0, i]:.6e}, RMSE B ({scheme}): {errors[scheme][1, i]:.6e}")

    plot_errors(dt_values, errors, 'dt', 'Root Mean Square Error vs dt')


def error_checker_grid(tmax, Da, dt=0.00001, gridsteps=100):
    """Compares solution with a given grid size to a reference solution. The reference solution is the one computed with the \
    smallest dt and the largest gridsteps. Function works for both implicit and explicit methods. It computes the root mean square error \
    and graphs it as a function of grid steps.

    :param float tmax: the total span of time from the beginning
    :param float Da: the Damköhler number
    :param float dt: timestep of the reference solution (also used for all compared solutions), default is 0.00001
    :param int gridsteps: number of gridsteps of the reference solution, default is 100
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt is nonpositive
    :raises ValueError: if Da is negative
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt, Da)

    grid_values = np.array([80, 60, 40, 20])
    errors = {scheme: np.zeros((2, len(grid_values))) for scheme in SCHEMES}   # rows: species A, B

    for scheme, solver in SCHEMES.items():
        #reference solution of this scheme
        x_ref, t_ref, Phi_A_ref, Phi_B_ref = solver(gridsteps, tmax, dt, Da)

        #grid size error check, the reference is interpolated onto the coarser grid
        for i, grid in enumerate(grid_values):
            x, t, Phi_A, Phi_B = solver(grid, tmax, dt, Da)
            Phi_A_interp = np.interp(x, x_ref, Phi_A_ref[-1, :])
            Phi_B_interp = np.interp(x, x_ref, Phi_B_ref[-1, :])
            errors[scheme][:, i] = rmse(Phi_A[-1,:], Phi_A_interp), rmse(Phi_B[-1,:], Phi_B_interp)
            print(f"Grid Steps: {grid}, RMSE A ({scheme}): {errors[scheme][0, i]:.6e}, RMSE B ({scheme}): {errors[scheme][1, i]:.6e}")

    plot_errors(grid_values, errors, 'Grid Steps', 'Root Mean Square Error vs Grid Steps')


def error_checker_Da(tmax, dt=0.00001, gridsteps=100, Da_ref=0):
    """Compares solutions with different Damköhler numbers to a reference solution. The reference solution is the one \
    computed with Da_ref, which by default is zero, i.e. pure diffusion without any reaction. Function works for both implicit \
    and explicit methods. It computes the root mean square error, so it shows how strongly the reaction changes the \
    concentration profiles, and graphs it as a function of Da.

    :param float tmax: the total span of time from the beginning
    :param float dt: timestep of all solutions, default is 0.00001
    :param int gridsteps: number of gridsteps of all solutions, default is 100
    :param float Da_ref: the Damköhler number of the reference solution, default is 0 (no reaction)
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt is nonpositive
    :raises ValueError: if Da_ref is negative
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt, Da_ref)

    Da_values = np.array([1, 10, 100, 1000, 10000])
    errors = {scheme: np.zeros((2, len(Da_values))) for scheme in SCHEMES}   # rows: species A, B

    for scheme, solver in SCHEMES.items():
        #reference solution of this scheme
        x_ref, t_ref, Phi_A_ref, Phi_B_ref = solver(gridsteps, tmax, dt, Da_ref)

        #Da error check
        for i, Da_i in enumerate(Da_values):
            x, t, Phi_A, Phi_B = solver(gridsteps, tmax, dt, Da_i)
            errors[scheme][:, i] = rmse(Phi_A[-1,:], Phi_A_ref[-1,:]), rmse(Phi_B[-1,:], Phi_B_ref[-1,:])
            print(f"Da: {Da_i}, RMSE A ({scheme}): {errors[scheme][0, i]:.6e}, RMSE B ({scheme}): {errors[scheme][1, i]:.6e}")

    plot_errors(Da_values, errors, 'Da', f'Root Mean Square Error vs Da (reference Da = {Da_ref})')


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
    run_error_Da       = False   # effect of the Damkohler number

    #-------------------------------------Determine explicit stability in terms of delta_tau-------------
    #Turns out tau plays the role of a Fourier number, so it defines stability, through the test shown below
    #we found out the criteria is around dt > 1*10**(-4). Run below to see the error blow up.
    
    if run_stability:
        stability_crit(gridsteps, tmax, dt, Da, scheme="explicit")
        stability_crit(gridsteps, tmax, dt, Da, scheme="implicit")   # expected to be stable for all dt

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

    #Compare different Da to the reference solution without reaction
    if run_error_Da:
        error_checker_Da(tmax)


if __name__ == "__main__":
    main()