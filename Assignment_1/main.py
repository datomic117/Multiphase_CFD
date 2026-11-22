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


def determinePhi_expl(gridsteps, t_end, dt, Da, second_order=False):

    """Determine the solution of two concentration matrices of both species at
    every gridpoint in space and time. This is an explicit method.
    The derivative in time is discretised with the two step backward formula (BDF2) and the derivative in
    space with a second-order central scheme. The right-hand side f, the diffusion minus the reaction term
    R = Da*Phi_A*Phi_B (the same in both equations), is evaluated explicitly:

        3*Phi[j+1] - 4*Phi[j] + Phi[j-1] = 2*dt*f(Phi*)

    second_order=False: Phi* = Phi[j]. The formula is centred at time level j+1 but f is taken at level j, which makes the
        scheme only first order accurate in time. It is stable up to about dt/dl**2 = 1.
    second_order=True:  Phi* = 2*Phi[j] - Phi[j-1], f is extrapolated to level j+1. This is second order accurate in
        time, but only stable up to about dt/dl**2 = 1/3, so it needs a three times smaller timestep.
    BDF2 needs two old time levels, so the first step uses forward Euler.

    :param int gridsteps: the number of gridsteps for the 1D problem
    :param float t_end: the total span of time from the beginning
    :param float dt: represents the step in Fourier number - dimensionless timestep
    :param float Da: the Damköhler number - dimensionless diffusion coefficient/dimensionless reaction rate coefficient
    :param bool second_order: if True f is extrapolated to the new time level (second order in time),
        if False (default) it is taken at the old time level (first order in time)
    :return: the space mesh, time mesh, the concetration 2D arrays of A, B
    :rtype: tuple(np.array(1D), np.array(1D), np.array(2D), np.array(2D))
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, t_end, dt is nonpositive
    :raises ValueError: if Da is negative
    :raises ValueError: if second_order is not a bool

    """

    #Define ValueErrors
    check_inputs(gridsteps, t_end, dt, Da)
    check_flag("second_order", second_order)


    # Create the arrays for all space and time steps
    x = np.linspace(0, 1, gridsteps) #create coordinate mesh
    dl = x[1]-x[0] # define dl, just step between first and second gridpoint
    timesteps = int(t_end/dt) # number of time steps
    t = np.linspace(0, t_end, timesteps+1) #create time mesh
    F = dt/dl**2 # grid Fourier number, strength of the coupling between neighbours

    #Initialise concentraton arrays
    Phi_A = np.zeros((len(t),len(x)))
    Phi_B = np.zeros((len(t),len(x)))

    #Set the boundary conditions for all time levels before the time loop. The other wall stays zero.
    #Doing this inside the time loop would set the two species in a different order, which makes A and B unequal.
    Phi_A[:,0]  = 1    # boundary condition x = 0 : Phi_A = 1
    Phi_B[:,-1] = 1    # boundary condition x = L: Phi_B = 1

    def laplacian(P):
        """Second difference in space of the interior points of one time level, divided by dl**2 and times dt."""
        return F*(P[2:] - 2*P[1:-1] + P[:-2])

    # time loop over desired number of timesteps, every pass fills one whole row j+1 of the interior points
    for j in range(timesteps):
        if j == 0:   # first step with forward Euler, since BDF2 needs Phi[j-1]
            R = Da*Phi_A[0,1:-1]*Phi_B[0,1:-1]   # same R in both equations
            Phi_A[1,1:-1] = Phi_A[0,1:-1] + laplacian(Phi_A[0]) - dt*R
            Phi_B[1,1:-1] = Phi_B[0,1:-1] + laplacian(Phi_B[0]) - dt*R
        else:
            if second_order:   # extrapolate to the new time level
                A_star, B_star = 2*Phi_A[j] - Phi_A[j-1], 2*Phi_B[j] - Phi_B[j-1]
            else:              # old time level
                A_star, B_star = Phi_A[j], Phi_B[j]
            R = Da*A_star[1:-1]*B_star[1:-1]     # same R in both equations
            Phi_A[j+1,1:-1] = (4*Phi_A[j,1:-1] - Phi_A[j-1,1:-1] + 2*laplacian(A_star) - 2*dt*R)/3
            Phi_B[j+1,1:-1] = (4*Phi_B[j,1:-1] - Phi_B[j-1,1:-1] + 2*laplacian(B_star) - 2*dt*R)/3

    return x, t, Phi_A, Phi_B


def determinePhi_impl(gridsteps, t_end, dt, Da, optimised_solving=True, second_order=True, explicit_reaction=False):

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

    Two options change the discretisation, which makes comparing them easy:
        second_order=False   : backward Euler (first order in time) in every step, instead of BDF2.
        explicit_reaction=True: partially explicit reaction, R = Da*Phi_A_old*Phi_B_old, so both concentrations are
                                taken at the old time level. R is then known for both species and is a sink on the
                                right-hand side of both, the matrices contain no reaction term.

    :param int gridsteps: the number of gridsteps for the 1D problem
    :param float t_end: the total span of time from the beginning
    :param float dt: represents the step in Fourier number - dimensionless timestep
    :param float Da: the Damköhler number - dimensionless diffusion coefficient
    :param bool optimised_solving: if True (default) the tridiagonal system is solved directly from its three
        diagonals with the LAPACK routine dgtsv, which is much faster. If False the sparse matrix is assembled
        and solved with spsolve.
    :param bool second_order: if True (default) BDF2 is used for the derivative in time, if False backward Euler
    :param bool explicit_reaction: if False (default) R = Da*Phi_A_new*Phi_B_old, if True R = Da*Phi_A_old*Phi_B_old
    :return: the space mesh, time mesh, the concetration 2D arrays of A, B
    :rtype: tuple(np.array(1D), np.array(1D), np.array(2D), np.array(2D))
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, t_end, dt is nonpositive
    :raises ValueError: if Da is negative
    :raises ValueError: if optimised_solving, second_order or explicit_reaction is not a bool

    """

    #Define ValueErrors
    check_inputs(gridsteps, t_end, dt, Da)

    for name, flag in (("optimised_solving", optimised_solving), ("second_order", second_order),
                       ("explicit_reaction", explicit_reaction)):
        if not isinstance(flag, (bool, np.bool_)):
            print(f"Arg {name} has to be a boolean!")
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

    zeros = np.zeros(len(x))   # implicit reaction coefficient of a species whose reaction is on the right-hand side

    # time loop over desired number of timesteps, every pass fills one whole row n+1
    for n in range(timesteps):
        if n == 0 or not second_order:   # BDF2 needs Phi[n-1], so start with one backward Euler step
            c0, s = 1, 1
            rhs_A, rhs_B = Phi_A[n,:], Phi_B[n,:]
        else:
            c0, s = 3, 2
            rhs_A = 4*Phi_A[n,:] - Phi_A[n-1,:]
            rhs_B = 4*Phi_B[n,:] - Phi_B[n-1,:]

        # The reaction term R must be identical in both equations,
        # otherwise A and B are not consumed in equal amounts.
        if explicit_reaction:
            # Partially explicit: R = Da*Phi_A_old*Phi_B_old is known for both species, so it is a sink on
            # the right-hand side of both and the two species can be solved independently.
            R = Da*Phi_A[n,:]*Phi_B[n,:]
            Phi_A[n+1,:] = solve_step(rhs_A - s*dt*R, zeros, c0, s, 1, 0)   # A = 1 at x = 0, A = 0 at x = L
            Phi_B[n+1,:] = solve_step(rhs_B - s*dt*R, zeros, c0, s, 0, 1)   # B = 0 at x = 0, B = 1 at x = L
        else:
            # R = Da*Phi_A_new*Phi_B_old.
            # A: R is linear in Phi_A_new, so it goes into the matrix (implicit).
            Phi_A[n+1,:] = solve_step(rhs_A, Da*Phi_B[n], c0, s, 1, 0)   # A = 1 at x = 0, A = 0 at x = L

            # B: the same R is now known (Phi_A_new is solved), so it moves to the right-hand side as a sink.
            R = Da*Phi_A[n+1,:]*Phi_B[n,:]
            Phi_B[n+1,:] = solve_step(rhs_B - s*dt*R, zeros, c0, s, 0, 1)   # B = 0 at x = 0, B = 1 at x = L

    return x, t, Phi_A, Phi_B


def analytical_solution(x, tau, tol=1e-16):
    """Analytical solution of the problem WITHOUT a reaction term (Da = 0), so pure diffusion between the two walls.
    It is the sum of the steady state 1 - zeta and a transient Fourier sine series, where zeta = x/L is the
    dimensionless coordinate, which is the coordinate x of the numerical solvers (L = 1):

        Phi_A(zeta, tau) = 1 - zeta - 2/pi * SUM_{n=1}^{inf} sin(n*pi*zeta)/n * exp(-n^2*pi^2*tau)

    Species B has the mirrored boundary conditions (B = 1 at x = 1, B = 0 at x = 0), so Phi_B(x) = Phi_A(1 - x).
    The series is cut off when exp(-n^2*pi^2*tau) is smaller than tol, so the number of terms grows for small tau.
    At tau = 0 the initial condition is returned, since the series converges too slowly there.

    :param np.array(1D) x: the space coordinate, between 0 and 1
    :param float tau: the dimensionless time, a single value, equal to the time of the numerical solvers
    :param float tol: the size of the first neglected term of the series, default is 1e-16
    :return: the concentration profiles of A and B at time tau
    :rtype: tuple(np.array(1D), np.array(1D))
    :raises ValueError: if x is not between 0 and 1
    :raises ValueError: if tau is negative
    :raises ValueError: if tol is not between 0 and 1
    """

    #Define ValueErrors
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if np.any(x < 0) or np.any(x > 1):
        print("Arg x has to be between 0 and 1!")
        raise ValueError

    if tau < 0:
        print("Arg tau cannot be negative!")
        raise ValueError

    if not 0 < tol < 1:
        print("Arg tol has to be between 0 and 1!")
        raise ValueError

    if tau == 0:   # initial condition: only the walls are nonzero
        return np.where(x == 0, 1.0, 0.0), np.where(x == 1, 1.0, 0.0)

    def phi_A(xi):
        n_max = int(np.ceil(np.sqrt(-np.log(tol)/(np.pi**2*tau)))) + 1   # exp(-n^2 pi^2 tau) < tol beyond this
        series = np.zeros(len(xi))
        for start in range(1, n_max + 1, 2000):   # in blocks of terms, to limit the memory for very small tau
            n = np.arange(start, min(start + 2000, n_max + 1))
            series += (np.sin(np.outer(xi, n*np.pi))/n * np.exp(-n**2*np.pi**2*tau)).sum(axis=1)
        return 1 - xi - 2/np.pi*series

    return phi_A(x), phi_A(1 - x)


def plot(x, t, Phi_A, Phi_B,dt, N_snapshots = 10, analytical=False):
    """Plots the solution of the two concetration profiles
    by showing a number of snapshots of the system for different times. It uses
    the result of determinePhi_expl() or determinePhi_impl() and the number of snapshots.

    :param np.array(1D) x: the space coordinate
    :param np.array(1D) t: the time coordinate
    :param np.array(2D) Phi_A: the concetration profile of species A in space and time
    :param np.array(2D) Phi_B: the concetration profile of species B in space and time
    :param int N_snapshots: the number of snapshots taken from the system, default is 10
    :param bool analytical: if True the analytical solution is added at the same times as a black dotted line.
        It only exists without reaction, so use it for Da = 0 only. Default is False
    :return: None

    """
    #Define ValueError
    if not isinstance(N_snapshots, (int, np.integer)) or (N_snapshots <= 0):
        print("Arg N_snapshots has to be a positive integer larger than 0")
        raise ValueError

    if not isinstance(analytical, (bool, np.bool_)):
        print("Arg analytical has to be a boolean!")
        raise ValueError

    idx = np.linspace(0, len(t)-1, N_snapshots, dtype=int) #snapshots at equally spaced times
    colors = plt.cm.rainbow(np.linspace(0, 1, N_snapshots)) #one color for each snapshot using a colormap

    plt.figure(figsize=(12, 8))
    for j, c in zip(idx, colors):
        plt.plot(x, Phi_A[j, :], color=c, label=f't={t[j]:.3f}')   # A: solid
        plt.plot(x, Phi_B[j, :], '--', color=c)                    # B: dashed

    if analytical:   # same times, black dotted, drawn on top
        for j in idx:
            Phi_A_an, Phi_B_an = analytical_solution(x, t[j])
            plt.plot(x, Phi_A_an, ':', color='k', linewidth=1.5, label='analytical' if j == idx[0] else None)
            plt.plot(x, Phi_B_an, ':', color='k', linewidth=1.5)

    plt.title("Dimensionless concentration of A (solid) and B (dashed) with dt = {:.6f}".format(dt))
    plt.xlabel('x')
    plt.ylabel('Phi')
    plt.legend(fontsize=8, ncol = 2)
    plt.show()

    return None


SCHEMES = {"implicit": determinePhi_impl, "explicit": determinePhi_expl}   # name -> solver, shared by the tests below


def stability_crit(gridsteps, t_end, dt, Da, N_steps=50, scheme="explicit", second_order=None, explicit_reaction=False):
    """Finds the first and last timestep for which the chosen integration scheme blows up.
    It finds this timestep in the range of the given timestep to 100 times the given timesteps, in N_steps steps.
    The explicit scheme is only conditionally stable: the first order version up to about dt/dl**2 = 1 and the
    second order version (second_order=True) only up to about dt/dl**2 = 1/3. The diffusion term of the implicit scheme is stable for all
    timesteps, but its reaction term is lagged (or fully explicit if explicit_reaction=True), so the implicit scheme
    can still blow up when dt*Da is large. That limit depends on dt*Da, so choose dt and Da to make 100*dt*Da reach
    about 10 or more, and t_end long enough (many steps) for a blow up to grow.
    Increasing N_steps increases computational cost significantly.

    :param int gridsteps: the number of gridsteps for the 1D problem
    :param float t_end: the total span of time from the beginning
    :param float dt: represents the step in Fourier number - dimensionless timestep
    :param float Da: the Damköhler number - dimensionless diffusion coefficient
    :param int N_steps: the number of steps taken from dt to 100*dt, default is 50.
    :param str scheme: the scheme to test, "explicit" (default) or "implicit"
    :param bool second_order: the second_order argument of the solver, see determinePhi_expl() and determinePhi_impl().
        Default is None, which uses the default of the chosen solver (first order explicit, BDF2 implicit)
    :param bool explicit_reaction: R = Da*Phi_A_new*Phi_B_old (default) or the partially explicit
        R = Da*Phi_A_old*Phi_B_old, only for scheme="implicit", see determinePhi_impl()
    :return: None
    :raises ValueError: if gridsteps or N_steps is not int
    :raises ValueError: if gridsteps, t_end, dt, N_steps is nonpositive
    :raises ValueError: if Da is negative
    :raises ValueError: if scheme is not "explicit" or "implicit"
    :raises ValueError: if second_order is not None or a bool, or explicit_reaction is not a bool
    :raises ValueError: if explicit_reaction is True for scheme="explicit"

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

    if second_order is not None:
        check_flag("second_order", second_order)
    check_flag("explicit_reaction", explicit_reaction)

    if scheme == "explicit" and explicit_reaction:
        print('Arg explicit_reaction only applies to scheme "implicit"!')
        raise ValueError

    solver = SCHEMES[scheme]
    solver_kwargs = {} if second_order is None else {"second_order": second_order}
    if scheme == "implicit":
        solver_kwargs["explicit_reaction"] = explicit_reaction
    # name used in the printed output, only mentions the options that differ from the default
    if scheme == "implicit":
        scheme += (", 1st order" if second_order is False else "") + (", explicit R" if explicit_reaction else "")
    elif second_order:
        scheme += ", 2nd order"

    tol = 1e-9   # margin for rounding error in the bound |Phi| <= 1
    dt_test = np.linspace((dt), (100*dt), N_steps)
    for dt_i in dt_test:
        # an unstable run grows until it overflows to inf and nan, that is the thing we are looking for here,
        # so the overflow warnings are silenced for this call only
        with np.errstate(over='ignore', invalid='ignore'):
            x, t, Phi_A, Phi_B = solver(gridsteps, t_end, dt_i, Da, **solver_kwargs)
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
    or the analytical one, as long as it is evaluated at the same gridpoints.

    :param np.array(1D) profile: the concentration profile to check
    :param np.array(1D) profile_ref: the reference concentration profile at the same gridpoints
    :return: the root mean square error
    :rtype: float
    """
    return np.sqrt(np.mean((profile - profile_ref)**2))


def plot_errors(values, errors, xlabel, title, styles=None, floors=None):
    """Plots the errors of the error checkers on a log-log scale, for both species and every scheme or variant.

    :param np.array(1D) values: the values of the parameter that was varied
    :param dict errors: scheme name -> np.array of shape (2, len(values)), row 0 is species A and row 1 is species B
    :param str xlabel: the label of the x axis, the name of the varied parameter
    :param str title: the title of the plot
    :param dict styles: scheme name -> matplotlib format string, default is the one for "implicit" and "explicit"
    :param dict floors: name -> error of the numerical reference solution compared to the analytical solution, drawn as
        a horizontal gray line. Differences between solutions below this line are smaller than the error of the
        reference itself. Default is None, no lines.
    :return: None
    """
    if styles is None:
        styles = {"implicit": 'o-', "explicit": 's--'}
    plt.figure(figsize=(12, 8))
    for scheme, err in errors.items():
        plt.loglog(values, err[0], styles[scheme], label=f'RMSE A ({scheme})')
        plt.loglog(values, err[1], styles[scheme], label=f'RMSE B ({scheme})')
    for name, floor in (floors or {}).items():
        plt.axhline(floor, color='gray', ls=':', label=f'reference error vs analytical ({name})')
    plt.xlabel(xlabel)
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()


# Error Checker function: compare each dt and gridsize to the smallest dt and largest gridsize, and see if the error is within a certain tolerance.
def check_flag(name, flag):
    """Checks that an argument of the error checkers is a boolean.

    :param str name: the name of the argument, used in the message
    :param flag: the value to check
    :return: None
    :raises ValueError: if flag is not a bool
    """
    if not isinstance(flag, (bool, np.bool_)):
        print(f"Arg {name} has to be a boolean!")
        raise ValueError


def reference_error(Phi_A_ref, Phi_B_ref, t_ref):
    """Error of a numerical reference solution compared to the analytical solution at the final time.
    It is the part of the error that the comparisons to this reference can not see: the error of the reference
    itself, which for a fine timestep is dominated by the error of the grid. Only exists for Da = 0.

    :param np.array(2D) Phi_A_ref: the concentration profile of species A of the reference in space and time
    :param np.array(2D) Phi_B_ref: the concentration profile of species B of the reference in space and time
    :param np.array(1D) t_ref: the time coordinate of the reference
    :return: the largest RMSE of species A and B at the final time
    :rtype: float
    """
    x_ref = np.linspace(0, 1, Phi_A_ref.shape[1])
    Phi_A_an, Phi_B_an = analytical_solution(x_ref, t_ref[-1])
    return max(rmse(Phi_A_ref[-1,:], Phi_A_an), rmse(Phi_B_ref[-1,:], Phi_B_an))


def error_checker_dt(tmax, Da, dt=0.00001, gridsteps=100, use_analytical=True):
    """Compares solution with a given dt to a reference solution. The reference solution is the one computed with the \
    smallest dt and the largest gridsteps. Function works for both implicit and explicit methods. It computes the root mean square error \
    and graphs it as a function of dt.
    The reference is deliberately numerical, on the same grid as the compared solutions, so that only the error of the
    timestep is measured. Compared to the analytical solution the error would be limited by the grid, which is larger
    than the timestep error of the second order scheme. For Da = 0 the error of the numerical reference compared to the
    analytical solution is printed and drawn as a gray line, below it the differences are smaller than the reference error.

    :param float tmax: the total span of time from the beginning
    :param float Da: the Damköhler number
    :param float dt: timestep of the reference solution, default is 0.00001
    :param int gridsteps: number of gridsteps of the reference solution (also used for all compared solutions), default is 100
    :param bool use_analytical: if True (default) and Da = 0, the error of the reference compared to the analytical solution is added
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt is nonpositive
    :raises ValueError: if Da is negative
    :raises ValueError: if use_analytical is not a bool
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt, Da)
    check_flag("use_analytical", use_analytical)

    dt_values = np.array([0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001])
    errors = {scheme: np.zeros((2, len(dt_values))) for scheme in SCHEMES}   # rows: species A, B
    floors = {}

    for scheme, solver in SCHEMES.items():
        #reference solution of this scheme
        x_ref, t_ref, Phi_A_ref, Phi_B_ref = solver(gridsteps, tmax, dt, Da)
        if use_analytical and Da == 0:
            floors[scheme] = reference_error(Phi_A_ref, Phi_B_ref, t_ref)
            print(f"Error of the reference ({scheme}) compared to the analytical solution: {floors[scheme]:.6e}")

        #time step error check
        for i, dt_i in enumerate(dt_values):
            x, t, Phi_A, Phi_B = solver(gridsteps, tmax, dt_i, Da)
            errors[scheme][:, i] = rmse(Phi_A[-1,:], Phi_A_ref[-1,:]), rmse(Phi_B[-1,:], Phi_B_ref[-1,:])
            print(f"dt: {dt_i}, RMSE A ({scheme}): {errors[scheme][0, i]:.6e}, RMSE B ({scheme}): {errors[scheme][1, i]:.6e}")

    plot_errors(dt_values, errors, 'dt', 'Root Mean Square Error vs dt', floors=floors)


def error_checker_grid(tmax, Da, dt=0.00001, gridsteps=100, use_analytical=True):
    """Compares solution with a given grid size to a reference solution. The reference solution is the one computed with the \
    smallest dt and the largest gridsteps. Function works for both implicit and explicit methods. It computes the root mean square error \
    and graphs it as a function of grid steps. It prints the observed order of convergence in space, which is the slope
    of the graph.
    For Da = 0 the reference is the analytical solution (if use_analytical is True), which is exact, so no reference
    solution is computed and it is evaluated on each grid directly. For Da > 0 there is no analytical solution and the
    numerical reference is used, interpolated onto the coarser grid.

    :param float tmax: the total span of time from the beginning
    :param float Da: the Damköhler number
    :param float dt: timestep of all solutions (and of the numerical reference), default is 0.00001
    :param int gridsteps: number of gridsteps of the numerical reference solution, default is 100. Not used for the analytical reference
    :param bool use_analytical: if True (default) and Da = 0, the analytical solution is the reference
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt is nonpositive
    :raises ValueError: if Da is negative
    :raises ValueError: if use_analytical is not a bool
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt, Da)
    check_flag("use_analytical", use_analytical)

    analytical = use_analytical and Da == 0
    print("Reference: analytical solution" if analytical else "Reference: numerical solution")

    grid_values = np.array([80, 60, 40, 20])
    errors = {scheme: np.zeros((2, len(grid_values))) for scheme in SCHEMES}   # rows: species A, B

    for scheme, solver in SCHEMES.items():
        if not analytical:
            #reference solution of this scheme
            x_ref, t_ref, Phi_A_ref, Phi_B_ref = solver(gridsteps, tmax, dt, Da)

        #grid size error check
        for i, grid in enumerate(grid_values):
            x, t, Phi_A, Phi_B = solver(grid, tmax, dt, Da)
            if analytical:
                Phi_A_ref_i, Phi_B_ref_i = analytical_solution(x, t[-1])
            else:   # the reference is interpolated onto the coarser grid
                Phi_A_ref_i = np.interp(x, x_ref, Phi_A_ref[-1, :])
                Phi_B_ref_i = np.interp(x, x_ref, Phi_B_ref[-1, :])
            errors[scheme][:, i] = rmse(Phi_A[-1,:], Phi_A_ref_i), rmse(Phi_B[-1,:], Phi_B_ref_i)
            print(f"Grid Steps: {grid}, RMSE A ({scheme}): {errors[scheme][0, i]:.6e}, RMSE B ({scheme}): {errors[scheme][1, i]:.6e}")

    #observed order of convergence in space: slope of log(error) against log(grid spacing), taken for species A
    print()
    h = 1/(grid_values - 1)
    for scheme, err in errors.items():
        if np.all(np.isfinite(err[0])) and np.all(err[0] > 0):
            print(f"Observed order in space ({scheme}): {np.polyfit(np.log(h), np.log(err[0]), 1)[0]:.2f}")
        else:
            print(f"Observed order in space ({scheme}): not defined, the error is zero or not finite")

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


def error_checker_order(tmax, Da, dt=0.00001, gridsteps=100, use_analytical=True):
    """Compares the discretisation of the implicit method: first order (backward Euler) against second order (BDF2) in
    time, and the reaction term R = Da*Phi_A_new*Phi_B_old against the partially explicit R = Da*Phi_A_old*Phi_B_old.
    All four combinations are compared to one common reference solution, the best variant (BDF2 with
    R = Da*Phi_A_new*Phi_B_old) computed with the given dt. It computes the root mean square error, graphs it as a
    function of dt and prints the observed order of convergence, which is the slope of the graph.
    Note that the reaction term is only first order accurate in time, so for large Da BDF2 will not reach second order.
    As in error_checker_dt() the reference is numerical on the same grid, and for Da = 0 its error compared to the
    analytical solution is printed and drawn as a gray line.

    :param float tmax: the total span of time from the beginning
    :param float Da: the Damköhler number
    :param float dt: timestep of the reference solution, default is 0.00001
    :param int gridsteps: number of gridsteps of all solutions, default is 100
    :param bool use_analytical: if True (default) and Da = 0, the error of the reference compared to the analytical solution is added
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt is nonpositive
    :raises ValueError: if Da is negative
    :raises ValueError: if use_analytical is not a bool
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt, Da)
    check_flag("use_analytical", use_analytical)

    dt_values = np.array([0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001])
    # name: (second_order, explicit_reaction)
    variants = {"2nd order, semi-implicit R": (True, False),
                "2nd order, explicit R":      (True, True),
                "1st order, semi-implicit R": (False, False),
                "1st order, explicit R":      (False, True)}
    styles = {"2nd order, semi-implicit R": 'o-',  "2nd order, explicit R": 's-',
              "1st order, semi-implicit R": '^--', "1st order, explicit R": 'D--'}
    errors = {name: np.zeros((2, len(dt_values))) for name in variants}   # rows: species A, B

    #common reference solution
    x_ref, t_ref, Phi_A_ref, Phi_B_ref = determinePhi_impl(gridsteps, tmax, dt, Da)
    floors = {}
    if use_analytical and Da == 0:
        floors["reference"] = reference_error(Phi_A_ref, Phi_B_ref, t_ref)
        print(f"Error of the reference compared to the analytical solution: {floors['reference']:.6e}")

    for name, (second_order, explicit_reaction) in variants.items():
        for i, dt_i in enumerate(dt_values):
            # an unstable variant overflows, this shows up as an infinite error, so the warnings are silenced
            with np.errstate(over='ignore', invalid='ignore'):
                x, t, Phi_A, Phi_B = determinePhi_impl(gridsteps, tmax, dt_i, Da,
                                                       second_order=second_order, explicit_reaction=explicit_reaction)
                errors[name][:, i] = rmse(Phi_A[-1,:], Phi_A_ref[-1,:]), rmse(Phi_B[-1,:], Phi_B_ref[-1,:])
            print(f"dt: {dt_i}, RMSE A ({name}): {errors[name][0, i]:.6e}, RMSE B ({name}): {errors[name][1, i]:.6e}")

    #observed order of convergence: slope of log(error) against log(dt), taken for species A
    print()
    for name, err in errors.items():
        if np.all(np.isfinite(err[0])) and np.all(err[0] > 0):
            order = np.polyfit(np.log(dt_values), np.log(err[0]), 1)[0]
            print(f"Observed order ({name}): {order:.2f}")
        else:
            print(f"Observed order ({name}): not defined, the error is zero or not finite (unstable?)")

    plot_errors(dt_values, errors, 'dt', f'Root Mean Square Error vs dt for different discretisations (Da = {Da})', styles, floors)


def validate_analytical(tmax, Da=0, dt=0.00001, gridsteps=100, show_plot=True):
    """Validates the numerical solvers with the analytical solution, for both the implicit and explicit method.

    The analytical solution has no reaction term, so for Da = 0 the concentrations Phi_A and Phi_B are compared to it
    directly. For Da > 0 only the difference Phi_A - Phi_B can be compared: the reaction consumes A and B equally, so it
    cancels in the difference and Phi_A - Phi_B obeys pure diffusion for any Da. The numerical difference then has to
    stay equal to the analytical one, whatever the value of Da, as long as the solution is stable. This checks the
    diffusion, the boundary conditions and that both species lose the same amount by reaction, but not the size of
    the reaction rate itself.
    The error at the final time is printed as a root mean square error, and for Da = 0 the snapshots of the implicit
    solution are plotted together with the analytical solution.

    :param float tmax: the total span of time from the beginning
    :param float Da: the Damköhler number, default is 0
    :param float dt: timestep of the numerical solutions, default is 0.00001
    :param int gridsteps: number of gridsteps of the numerical solutions, default is 100
    :param bool show_plot: if True (default) and Da = 0, plot the numerical and analytical snapshots
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt is nonpositive
    :raises ValueError: if Da is negative
    :raises ValueError: if show_plot is not a bool
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt, Da)
    check_flag("show_plot", show_plot)

    print(f"Validation against the analytical solution, Da = {Da}, dt = {dt}, gridsteps = {gridsteps}")
    for scheme, solver in SCHEMES.items():
        x, t, Phi_A, Phi_B = solver(gridsteps, tmax, dt, Da)
        Phi_A_an, Phi_B_an = analytical_solution(x, t[-1])
        if Da == 0:
            print(f"  {scheme}: RMSE A: {rmse(Phi_A[-1,:], Phi_A_an):.6e}, RMSE B: {rmse(Phi_B[-1,:], Phi_B_an):.6e}")
        print(f"  {scheme}: RMSE of A - B: {rmse(Phi_A[-1,:] - Phi_B[-1,:], Phi_A_an - Phi_B_an):.6e}")

        if show_plot and Da == 0 and scheme == "implicit":
            plot(x, t, Phi_A, Phi_B, dt, analytical=True)


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
    run_stability_reaction = True  # find the stability limit of the implicit method for large dt*Da, all four variants
    run_impl_showcase  = True   # show implicit stability for increasing dt
    run_error_dt       = True   # convergence in dt
    run_error_grid     = True   # convergence in grid size
    run_error_Da       = True   # effect of the Damkohler number
    run_error_order    = True   # first vs second order in time, and semi-implicit vs explicit reaction term
    run_validation     = True   # compare to the analytical solution: Da = 0 directly, and A - B for the given Da

    #-------------------------------------Validate against the analytical solution------------------------
    #The analytical solution has no reaction term, so the solvers are compared to it directly for Da = 0. For the given
    #Da only the difference A - B can be compared, since the reaction cancels in it.
    if run_validation:
        validate_analytical(tmax, 0, dt, gridsteps)
        validate_analytical(tmax, Da, dt, gridsteps, show_plot=False)

    #-------------------------------------Determine explicit stability in terms of delta_tau-------------
    #Turns out tau plays the role of a Fourier number, so it defines stability, through the test shown below
    #we found out the criteria is around dt > 1*10**(-4). Run below to see the error blow up.
    
    if run_stability:
        stability_crit(gridsteps, tmax, dt, Da, scheme="explicit")
        stability_crit(gridsteps, tmax, dt, Da, scheme="explicit", second_order=True)   # about three times smaller limit
        stability_crit(gridsteps, tmax, dt, Da, scheme="implicit")   # stable in this dt range, but see run_stability_reaction for larger dt*Da

    #The reaction term of the implicit method is lagged, so it can still blow up for large dt*Da. This limit needs a wider
    #dt range and a longer time than above, since a blow up needs enough timesteps to grow.
    if run_stability_reaction:
        for second_order in (True, False):
            for explicit_reaction in (False, True):
                stability_crit(gridsteps, 1.0, 0.0001, Da, scheme="implicit",
                               second_order=second_order, explicit_reaction=explicit_reaction)

    #Showcase implicit function stablity by using bigger and bigger dt and plotting
    if run_impl_showcase:
        for dt_i in [0.000025, 0.0001, 0.001, 0.01]:
            x, t, Phi_A, Phi_B = determinePhi_impl(gridsteps, tmax, dt_i, Da)
            plot(x, t, Phi_A, Phi_B, dt_i)

    #Compare each dt and gridsize to the reference solution. Da = 0 first: there the analytical solution exists, the
    #grid study is compared to it and the reference of the dt studies is checked against it. Then the given Da, where
    #the reference is numerical.
    if run_error_dt:
        for Da_i in (0, Da):
            error_checker_dt(tmax, Da_i)

    if run_error_grid:
        for Da_i in (0, Da):
            error_checker_grid(tmax, Da_i)

    #Compare different Da to the reference solution without reaction
    if run_error_Da:
        error_checker_Da(tmax)

    #Compare the discretisations of the implicit method, Da = 0 shows pure second order
    if run_error_order:
        for Da_i in (0, Da):
            error_checker_order(tmax, Da_i)


if __name__ == "__main__":
    main()