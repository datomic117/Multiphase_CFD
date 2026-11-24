import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from   scipy.sparse.linalg import spsolve, cg, bicgstab, spilu, LinearOperator
import math
import os
import csv
from   cycler import cycler
from   matplotlib.cm import ScalarMappable
from   matplotlib.colors import LinearSegmentedColormap, Normalize
from   matplotlib.lines import Line2D
from   matplotlib.ticker import NullFormatter, ScalarFormatter
from   scipy.linalg.lapack import dgtsv

# Colours of the graphs: the first three slots of a validated categorical palette, always in this order (blue, orange, aqua),
# a sequential blue ramp for time, and ink and hairline grid colours that stay quiet next to the data.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, AXIS, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"
TIME_CMAP = LinearSegmentedColormap.from_list("time", ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=[BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]),
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False, "axes.titlesize": 12,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
})

GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")   # folder where the graphs are saved
SHOW_PLOTS = True   # if False the graphs are only saved, not shown in a window, see main()


def finish_figure(filename=None):
    """Ends a figure: saves it in the graphs folder when a filename is given, then shows or closes it.

    :param str filename: name of the png file inside the graphs folder, default is None, so the figure is not saved
    :return: None
    """
    if filename is not None:
        os.makedirs(GRAPH_DIR, exist_ok=True)
        plt.savefig(os.path.join(GRAPH_DIR, filename), dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


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


def plot(x, t, Phi_A, Phi_B, dt, N_snapshots = 10, analytical=False, filename=None, title=None):
    """Plots the solution of the two concetration profiles
    by showing a number of snapshots of the system for different times. It uses
    the result of determinePhi_expl() or determinePhi_impl() and the number of snapshots.
    The time is shown by the colour (light blue early, dark blue late), species A is solid and species B is dashed.

    :param np.array(1D) x: the space coordinate
    :param np.array(1D) t: the time coordinate
    :param np.array(2D) Phi_A: the concetration profile of species A in space and time
    :param np.array(2D) Phi_B: the concetration profile of species B in space and time
    :param float dt: the timestep, only used in the default title
    :param int N_snapshots: the number of snapshots taken from the system, default is 10
    :param bool analytical: if True the analytical solution is added at the same times as a black dotted line.
        It only exists without reaction, so use it for Da = 0 only. Default is False
    :param str filename: if given, the graph is saved as this png file in the graphs folder, default is None
    :param str title: the title of the graph, default is a title with dt
    :return: None
    :raises ValueError: if N_snapshots is not a positive integer
    :raises ValueError: if analytical is not a bool

    """
    #Define ValueError
    if not isinstance(N_snapshots, (int, np.integer)) or (N_snapshots <= 0):
        print("Arg N_snapshots has to be a positive integer larger than 0")
        raise ValueError

    if not isinstance(analytical, (bool, np.bool_)):
        print("Arg analytical has to be a boolean!")
        raise ValueError

    idx = np.linspace(0, len(t)-1, N_snapshots, dtype=int) #snapshots at equally spaced times
    norm = Normalize(t[0], t[-1])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    width = 2.6 if analytical else 1.8   # thicker with the analytical solution on top, so both stay visible
    for j in idx:
        c = TIME_CMAP(norm(t[j]))   # one colour for each snapshot, time is a magnitude so one hue light to dark
        ax.plot(x, Phi_A[j, :], color=c, linewidth=width)          # A: solid
        ax.plot(x, Phi_B[j, :], '--', color=c, linewidth=width)    # B: dashed

    handles = [Line2D([0], [0], color=INK2, linewidth=1.8, label="species A"),
               Line2D([0], [0], color=INK2, linewidth=1.8, linestyle='--', label="species B")]
    if analytical:   # same times, black dotted, drawn on top
        for j in idx:
            Phi_A_an, Phi_B_an = analytical_solution(x, t[j])
            ax.plot(x, Phi_A_an, ':', color=INK, linewidth=1.2)
            ax.plot(x, Phi_B_an, ':', color=INK, linewidth=1.2)
        handles.append(Line2D([0], [0], color=INK, linewidth=1.2, linestyle=':', label="analytical solution"))

    ax.set_title(title if title is not None else "Dimensionless concentration of A (solid) and B (dashed) with dt = {:.6f}".format(dt))
    ax.set_xlabel('x')
    ax.set_ylabel('Phi')
    ax.legend(handles=handles, loc='upper center')
    fig.colorbar(ScalarMappable(norm=norm, cmap=TIME_CMAP), ax=ax, label="t", pad=0.02)
    finish_figure(filename)

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
            plot(*last_stable, title=f"Last stable timestep, {scheme}, dt = {last_dt:.6f}")
            plot(*first_unstable, title=f"First unstable timestep, {scheme}, dt = {dt_i:.6f}")
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


def plot_errors(values, errors, xlabel, title, styles=None, floors=None, filename=None):
    """Plots the errors of the error checkers on a log-log scale, for both species and every scheme or variant.

    :param np.array(1D) values: the values of the parameter that was varied
    :param dict errors: scheme name -> np.array of shape (2, len(values)), row 0 is species A and row 1 is species B
    :param str xlabel: the label of the x axis, the name of the varied parameter
    :param str title: the title of the plot
    :param dict styles: scheme name -> matplotlib format string, default is the one for "implicit" and "explicit"
    :param dict floors: name -> error of the numerical reference solution compared to the analytical solution, drawn as
        a horizontal gray line. Differences between solutions below this line are smaller than the error of the
        reference itself. Default is None, no lines.
    :param str filename: if given, the graph is saved as this png file in the graphs folder, default is None.
        Points with an RMSE above 1 (runs that blew up) are not plotted, and the x label says so.
    :return: None
    """
    if styles is None:
        styles = {"implicit": 'o-', "explicit": 's--'}
    plt.figure(figsize=(12, 8))
    blown_up = False
    for scheme, err in errors.items():
        # The concentrations are bounded by 1, so an RMSE above 1 can only come from a run that blew up (unstable).
        # Those points are left out, otherwise they stretch the y axis and flatten all the other points.
        blown_up = blown_up or bool(np.any(err > 1))
        err = np.where(err > 1, np.nan, err)
        plt.loglog(values, err[0], styles[scheme], label=f'RMSE A ({scheme})')
        plt.loglog(values, err[1], styles[scheme], label=f'RMSE B ({scheme})')
    for name, floor in (floors or {}).items():
        plt.axhline(floor, color='gray', ls=':', label=f'reference error vs analytical ({name})')
    plt.xlabel(xlabel + ("   (unstable runs with RMSE > 1 are not shown)" if blown_up else ""))
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    finish_figure(filename)


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

    plot_errors(dt_values, errors, 'dt', f'Root Mean Square Error vs dt (Da = {Da})', floors=floors)


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

    plot_errors(grid_values, errors, 'Grid Steps', f'Root Mean Square Error vs Grid Steps (Da = {Da})')


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


#--------------------------------------------------------------------------------------------------------------
# FINAL GRAPHS: the figures that are saved in the graphs folder (see make_graphs() and main())
#--------------------------------------------------------------------------------------------------------------

# One entry per discretisation. The colour is the scheme (and how the reaction is treated), solid or dashed is the
# order in time and every colour has its own marker, so nothing is shown by colour alone.
VARIANTS = {
    "implicit, 2nd order":             dict(scheme="implicit", second_order=True,  explicit_reaction=False, color=BLUE,   marker="o", ls="-"),
    "implicit, 1st order":             dict(scheme="implicit", second_order=False, explicit_reaction=False, color=BLUE,   marker="o", ls="--"),
    "explicit, 2nd order":             dict(scheme="explicit", second_order=True,  explicit_reaction=False, color=ORANGE, marker="s", ls="-"),
    "explicit, 1st order":             dict(scheme="explicit", second_order=False, explicit_reaction=False, color=ORANGE, marker="s", ls="--"),
    "implicit, 2nd order, explicit R": dict(scheme="implicit", second_order=True,  explicit_reaction=True,  color=AQUA,   marker="D", ls="-"),
    "implicit, 1st order, explicit R": dict(scheme="implicit", second_order=False, explicit_reaction=True,  color=AQUA,   marker="D", ls="--"),
}


def solve_variant(variant, gridsteps, t_end, dt, Da):
    """Runs the solver of one of the VARIANTS.

    :param str variant: a key of VARIANTS
    :param int gridsteps: the number of gridsteps for the 1D problem
    :param float t_end: the total span of time from the beginning
    :param float dt: the timestep
    :param float Da: the Damköhler number
    :return: the space mesh, time mesh, the concetration 2D arrays of A, B
    """
    v = VARIANTS[variant]
    if v["scheme"] == "implicit":
        return determinePhi_impl(gridsteps, t_end, dt, Da, second_order=v["second_order"], explicit_reaction=v["explicit_reaction"])
    return determinePhi_expl(gridsteps, t_end, dt, Da, second_order=v["second_order"])


def check_steps(t_end, dts):
    """Checks that every timestep fits an integer number of times in t_end, so that the solution really ends at t_end.

    :param float t_end: the total span of time from the beginning
    :param dts: one timestep or an array of them
    :return: None
    :raises ValueError: if a timestep does not fit
    """
    for dt in np.atleast_1d(dts):
        if int(t_end/dt) != round(t_end/dt):
            print(f"Timestep dt = {dt} does not fit an integer number of times in t_end = {t_end}!")
            raise ValueError


def combined_rmse(Phi_A, Phi_B, Phi_A_ref, Phi_B_ref):
    """Root mean square error over both species together, so one number per run. At Da = 0 both species
    are mirror images, so they have the same error anyway.

    :param np.array(1D) Phi_A: the profile of species A
    :param np.array(1D) Phi_B: the profile of species B
    :param np.array(1D) Phi_A_ref: the reference profile of species A
    :param np.array(1D) Phi_B_ref: the reference profile of species B
    :return: the combined root mean square error
    :rtype: float
    """
    return np.sqrt((np.mean((Phi_A - Phi_A_ref)**2) + np.mean((Phi_B - Phi_B_ref)**2))/2)


def bounded(Phi_A, Phi_B, tol=1e-9):
    """True if the concentrations stay between -1 and 1 (with a margin for rounding error), which they must do
    for a stable solution. A solution with inf or nan is not bounded.

    :param np.array(2D) Phi_A: the concentration profile of species A in space and time
    :param np.array(2D) Phi_B: the concentration profile of species B in space and time
    :param float tol: margin for rounding error, default is 1e-9
    :return: whether the solution is bounded
    :rtype: bool
    """
    return bool(np.max(np.abs(Phi_A)) <= 1 + tol and np.max(np.abs(Phi_B)) <= 1 + tol)


def fit_slope(values, errors, floor=None, n_points=None):
    """Slope of log(error) against log(values), which is the observed order of convergence.

    :param np.array(1D) values: the varied parameter (dt or the grid spacing)
    :param np.array(1D) errors: the errors
    :param float floor: if given, only errors larger than 3 times this floor are used, since below it the error is
        dominated by something else (the grid error, in the dt study). Default is None
    :param int n_points: if given, only the first n_points usable points are used. Default is None
    :return: the slope, nan if fewer than two points can be used
    :rtype: float
    """
    values, errors = np.asarray(values, float), np.asarray(errors, float)
    use = np.isfinite(errors) & (errors > 0)
    if floor is not None:
        use &= errors > 3*floor
    idx = np.flatnonzero(use)[:n_points]
    if len(idx) < 2:
        return np.nan
    return np.polyfit(np.log(values[idx]), np.log(errors[idx]), 1)[0]


def save_table(filename, header, rows):
    """Saves the plotted numbers of a graph as a csv file in graphs/data, as a table version of the graph.

    :param str filename: name of the csv file
    :param list header: the column names
    :param list rows: the rows, each a list of values
    :return: None
    """
    folder = os.path.join(GRAPH_DIR, "data")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def add_guide(ax, x0, y0, x1, slope, label):
    """Draws a thin gray guide line with a given slope on a log-log graph, to compare the order of convergence.

    :param ax: the axes
    :param float x0: x of the start of the line
    :param float y0: y of the start of the line
    :param float x1: x of the end of the line
    :param float slope: the slope on the log-log graph
    :param str label: text at the end of the line
    :return: None
    """
    y1 = y0*(x1/x0)**slope
    ax.plot([x0, x1], [y0, y1], color=MUTED, linewidth=1, zorder=1)
    ax.annotate(label, (x1, y1), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")


def mark_chosen_Da(ax, Da_chosen):
    """Marks the Da that is used in the other graphs with a thin vertical line.

    :param ax: the axes
    :param float Da_chosen: the Damköhler number to mark
    :return: None
    """
    ax.axvline(Da_chosen, color=INK2, linewidth=1, zorder=1)
    ax.annotate(f"Da = {Da_chosen:g}", (Da_chosen, 0), xycoords=("data", "axes fraction"), xytext=(4, 5),
                textcoords="offset points", color=INK2, fontsize=9)


def graph_error_vs_analytical(tmax=0.1, gridsteps=100, dt_grid=0.0000025, grid_values=(20, 40, 80, 160, 320),
                              filename="01_error_vs_analytical.png"):
    """Graph 1: the error of all four schemes compared to the analytical solution, without reaction (Da = 0).
    Left the error against the timestep (on a grid of gridsteps points), right against the number of gridsteps
    (with a small timestep dt_grid).
    Each scheme is run in its own stable range of dt: the implicit schemes are stable for any dt, so they are run up to
    dt = 0.01, which brings their time error above the grid error. The error of the grid, which is what remains
    when dt goes to zero, is drawn as a gray line. Below it the time error can not be seen in a comparison to the
    analytical solution. The slope in the legend is the observed order of convergence, taken from the points that
    are more than 3 times above that line (dt) or from the coarsest three grids (grid).

    :param float tmax: the total span of time from the beginning, default is 0.1
    :param int gridsteps: the number of gridsteps in the dt study, default is 100
    :param float dt_grid: the timestep in the grid study, default is 0.0000025. It has to be small enough for the
        explicit schemes to be stable on the finest grid
    :param tuple grid_values: the numbers of gridsteps in the grid study
    :param str filename: name of the png file in the graphs folder
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt_grid is nonpositive
    :raises ValueError: if a timestep does not fit an integer number of times in tmax
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt_grid, 0)

    # every scheme in its own stable range of dt: explicit 1st order is stable up to about dt/dl**2 = 1,
    # explicit 2nd order up to about 1/3, the implicit schemes always
    implicit_dts = [1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 2.5e-5]
    dt_values = {"implicit, 2nd order": implicit_dts, "implicit, 1st order": implicit_dts,
                 "explicit, 2nd order": [2.5e-5, 1e-5, 5e-6, 2.5e-6],
                 "explicit, 1st order": [1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6, 2.5e-6]}
    grids = np.array(grid_values)
    h = 1/(grids - 1)
    check_steps(tmax, [dt for dts in dt_values.values() for dt in dts] + [dt_grid])

    # error of the grid: the second order implicit scheme with a very small dt, so almost no time error is left
    x, t, A, B = determinePhi_impl(gridsteps, tmax, 1e-5, 0)
    floor = combined_rmse(A[-1], B[-1], *analytical_solution(x, t[-1]))
    print(f"Graph 1: error of the grid with {gridsteps} points (no time error) = {floor:.3e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    rows = []
    for name, dts in dt_values.items():
        dts = np.array(dts)
        err = np.zeros(len(dts))
        for i, dt in enumerate(dts):
            x, t, A, B = solve_variant(name, gridsteps, tmax, dt, 0)
            err[i] = combined_rmse(A[-1], B[-1], *analytical_solution(x, t[-1]))
            rows.append(["dt study", name, dt, err[i]])
        slope = fit_slope(dts, err, floor)
        note = f"slope {slope:.1f}" if np.isfinite(slope) else "at the grid error"
        print(f"  dt study, {name}: observed order {slope:.2f}")
        v = VARIANTS[name]
        ax1.loglog(dts, err, color=v["color"], marker=v["marker"], linestyle=v["ls"], label=f"{name} ({note})")

        err = np.zeros(len(grids))
        for i, N in enumerate(grids):
            x, t, A, B = solve_variant(name, N, tmax, dt_grid, 0)
            err[i] = combined_rmse(A[-1], B[-1], *analytical_solution(x, t[-1]))
            rows.append(["grid study", name, N, err[i]])
        slope = fit_slope(h, err, n_points=3)
        print(f"  grid study, {name}: observed order {slope:.2f}")
        ax2.loglog(grids, err, color=v["color"], marker=v["marker"], linestyle=v["ls"], label=f"{name} (slope {slope:.1f})")

    ax1.axhline(floor, color=MUTED, linewidth=1.2, linestyle=":", zorder=1)
    ax1.annotate(f"error of the grid ({gridsteps} points)", (1e-5, floor), xytext=(0, -10), textcoords="offset points",
                 color=INK2, fontsize=9, ha="left", va="top")
    add_guide(ax1, 1.5e-4, 2e-7, 6e-4, 1, "dt")
    add_guide(ax1, 2.5e-4, 2e-8, 1e-3, 2, "dt²")
    ax1.set_xlabel("dt")
    ax1.set_ylabel("RMSE compared to the analytical solution")
    ax1.set_title(f"(a) Time step, {gridsteps} gridpoints, Da = 0")
    ax1.legend(fontsize=9, loc="upper left")

    add_guide(ax2, 160, 3e-7, 320, -2, "N⁻²")
    ax2.set_xticks(grids)
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.annotate("the first order schemes level off\nat their time error", (0.97, 0.55), xycoords="axes fraction",
                 ha="right", color=INK2, fontsize=9)
    ax2.set_xlabel("number of gridpoints N")
    ax2.set_ylabel("RMSE compared to the analytical solution")
    ax2.set_title(f"(b) Grid, dt = {dt_grid:g}, Da = 0")
    ax2.legend(fontsize=9, loc="lower left")
    fig.suptitle(f"Error compared to the analytical solution at t = {tmax:g}, no reaction", fontsize=13)
    fig.tight_layout()
    save_table("01_error_vs_analytical.csv", ["study", "scheme", "dt or gridsteps", "RMSE"], rows)
    finish_figure(filename)


def graph_error_vs_Da(tmax=0.1, gridsteps=101, dt_coarse=0.0001, dt_fine=0.00001, fine_factor=4, Da_chosen=1000,
                      filename="02_error_vs_Da.png"):
    """Graph 2: the numerical error when Da increases. There is no analytical solution with reaction, so the error is
    the difference with a very fine numerical solution of the same scheme: (a) a coarse timestep against a fine one
    on the same grid, (b) a coarse grid against a grid with fine_factor times smaller spacing, with the same timestep,
    so the grid error is separated from the time error. The four implicit schemes are compared: first and second order in
    time, and the reaction term R = Da*Phi_A_new*Phi_B_old (semi-implicit) or Da*Phi_A_old*Phi_B_old (explicit R).

    :param float tmax: the total span of time from the beginning, default is 0.1
    :param int gridsteps: the number of gridsteps of the coarse grid, default is 101 so that every
        fine_factor-th point of the fine grid is a point of the coarse grid
    :param float dt_coarse: the coarse timestep, default is 0.0001
    :param float dt_fine: the fine timestep of the reference, default is 0.00001
    :param int fine_factor: the grid of the reference has fine_factor times smaller spacing, default is 4
    :param float Da_chosen: the Da that is used in the other graphs, marked in the graph, default is 1000
    :param str filename: name of the png file in the graphs folder
    :return: None
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt_coarse, dt_fine is nonpositive
    :raises ValueError: if a timestep does not fit an integer number of times in tmax
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt_coarse, 0)
    check_inputs(gridsteps, tmax, dt_fine, 0)
    check_steps(tmax, [dt_coarse, dt_fine])

    Da_values = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
    names = ["implicit, 2nd order", "implicit, 1st order", "implicit, 2nd order, explicit R", "implicit, 1st order, explicit R"]
    gridsteps_fine = (gridsteps - 1)*fine_factor + 1
    err_dt = np.zeros((len(names), len(Da_values)))
    err_grid = np.zeros((len(names), len(Da_values)))

    rows = []
    print("Graph 2: error against Da")
    for i, name in enumerate(names):
        for j, Da in enumerate(Da_values):
            x, t, A, B = solve_variant(name, gridsteps, tmax, dt_coarse, Da)
            x_f, t_f, A_f, B_f = solve_variant(name, gridsteps, tmax, dt_fine, Da)
            err_dt[i, j] = combined_rmse(A[-1], B[-1], A_f[-1], B_f[-1])
            x_g, t_g, A_g, B_g = solve_variant(name, gridsteps_fine, tmax, dt_coarse, Da)
            err_grid[i, j] = combined_rmse(A[-1], B[-1], A_g[-1, ::fine_factor], B_g[-1, ::fine_factor])
            rows.append([name, Da, err_dt[i, j], err_grid[i, j]])
        print(f"  {name}: time error {err_dt[i, 0]:.2e} (Da = {Da_values[0]}) to {err_dt[i, -1]:.2e} (Da = {Da_values[-1]})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, err in ((ax1, err_dt), (ax2, err_grid)):
        for name in ["implicit, 2nd order, explicit R", "implicit, 1st order, explicit R", "implicit, 2nd order", "implicit, 1st order"]:
            v = VARIANTS[name]   # the explicit R curves first, so the semi-implicit curves are on top where they overlap
            ax.loglog(Da_values, err[names.index(name)], color=v["color"], marker=v["marker"], linestyle=v["ls"],
                      markersize=9 if v["color"] == AQUA else 6, label=name)
        mark_chosen_Da(ax, Da_chosen)
        ax.set_xlabel("Da")
        ax.set_ylabel("RMSE against the fine solution")
    ax1.set_title(f"(a) Time step: dt = {dt_coarse:g} against dt = {dt_fine:g}")
    ax2.set_title(f"(b) Grid: {gridsteps} against {gridsteps_fine} points, dt = {dt_coarse:g}")
    ax2.set_ylim(1e-6, 3e-5)
    ax2.annotate("all four schemes coincide", (0.03, 0.92), xycoords="axes fraction", color=INK2, fontsize=9)
    ax1.legend(fontsize=9, loc="lower left")
    fig.suptitle(f"Numerical error when Da increases, at t = {tmax:g}", fontsize=13)
    fig.tight_layout()
    save_table("02_error_vs_Da.csv", ["scheme", "Da", "time step error", "grid error"], rows)
    finish_figure(filename)


def critical_dt(variant, gridsteps, Da, t_end, dt_min, dt_max, n_scan=16, rel_width=0.05, t_long=None):
    """Finds the largest timestep for which a scheme is stable (the concentrations stay bounded by 1). The timesteps
    are scanned upwards on a logarithmic scale until the first unstable one, then the interval between the last
    stable and that unstable one is halved (on a logarithmic scale) until it is narrower than rel_width.
    A slowly growing instability needs time to exceed 1, so a timestep that is stable in a short run can still be
    unstable in a longer run. That is the main uncertainty, so if t_long is given the last stable timestep is checked
    with a run of that length and lowered in steps of rel_width until it is stable there too.
    The true limit lies between the two timesteps that are returned: it is unstable in the short run, so it is
    unstable in a longer one, and the lower one is stable in the long run.

    :param str variant: a key of VARIANTS
    :param int gridsteps: the number of gridsteps
    :param float Da: the Damköhler number
    :param float t_end: the total span of time of the runs of the scan
    :param float dt_min: the smallest timestep of the scan
    :param float dt_max: the largest timestep of the scan
    :param int n_scan: the number of timesteps in the scan, default is 16
    :param float rel_width: stop when the interval is this fraction of the lower end, also the step of the
        lowering, default is 0.05
    :param float t_long: the span of time of the long run that checks the last stable timestep, default is None, no check
    :return: the last stable and the first unstable timestep. The first is 0 if dt_min is already unstable and the
        second is inf if no timestep up to dt_max is unstable
    :rtype: tuple(float, float)
    """
    def stable(dt, t_run):
        with np.errstate(over='ignore', invalid='ignore'):   # an unstable run overflows, that is what we look for
            x, t, Phi_A, Phi_B = solve_variant(variant, gridsteps, t_run, dt, Da)
        return bounded(Phi_A, Phi_B)

    lo, hi = 0.0, np.inf
    for dt in np.geomspace(dt_min, dt_max, n_scan):
        if stable(dt, t_end):
            lo = dt
        else:
            hi = dt
            break
    if 0.0 < lo and np.isfinite(hi):
        while hi/lo > 1 + rel_width:
            mid = np.sqrt(lo*hi)
            if stable(mid, t_end):
                lo = mid
            else:
                hi = mid
    if t_long is not None:
        while lo >= dt_min and not stable(lo, t_long):
            hi = min(hi, lo)   # unstable in the long run, so the limit is below it
            lo /= 1 + rel_width
        if lo < dt_min:
            lo = 0.0
    return lo, hi


def graph_stability(gridsteps=100, Da_values=(100, 300, 1000, 3000, 10000, 30000, 100000), Da_chosen=1000,
                    filename="03_stability_critical_dt.png"):
    """Graph 3: the stability of all six schemes: the largest stable timestep against Da, for the explicit scheme and
    the implicit scheme, both first and second order in time, and for the implicit scheme with the reaction term
    R = Da*Phi_A_old*Phi_B_old (explicit R) instead of Da*Phi_A_new*Phi_B_old (semi-implicit).
    The explicit schemes are limited by the diffusion (dt/dl**2), which does not depend on Da, until the reaction
    takes over at very large Da. The implicit schemes are only limited by the reaction (dt*Da), so their limit
    falls like 1/Da and there is none for small Da.
    The bars are the uncertainty: the true limit lies between the largest timestep that is stable in a long run and the
    smallest one that was found unstable in a shorter run (see critical_dt()). The bar is mostly the effect of the length
    of the run, since a slowly growing instability needs time to exceed 1. An open triangle pointing up means that nothing was
    unstable up to the largest timestep that was tried (the implicit runs are 0.3 long, so the largest is 0.03: a blow up needs
    about 10 steps to show), and an open triangle pointing down that even the smallest was unstable.

    :param int gridsteps: the number of gridsteps, default is 100
    :param tuple Da_values: the Damköhler numbers
    :param float Da_chosen: the Da that is used in the other graphs, marked in the graph, default is 1000
    :param str filename: name of the png file in the graphs folder
    :return: scheme name -> {Da: (last stable dt, first unstable dt)}
    :rtype: dict
    :raises ValueError: if gridsteps is not int or nonpositive
    """

    #Define ValueErrors
    check_inputs(gridsteps, 1, 1, 0)

    # A blow up has to grow, so the runs are long enough for it: the explicit schemes blow up from the first steps (diffusion),
    # but the reaction only starts when A and B have met, so t_end has to be long enough for that
    def window(scheme, Da):
        if scheme == "explicit":
            return dict(t_end=0.1, dt_min=4e-6, dt_max=4e-4, t_long=0.3)
        # The implicit limit is about a few times 1/Da, so the scan starts below that. It stops at 0.03, since a run of
        # 0.3 then has 10 steps, which is the least for a blow up to show.
        return dict(t_end=0.3, dt_min=min(0.5/Da, 0.003), dt_max=0.03, t_long=1.0)

    Da_values = np.array(Da_values)
    names = list(VARIANTS)
    lo = np.zeros((len(names), len(Da_values)))
    hi = np.zeros((len(names), len(Da_values)))

    print("Graph 3: critical timestep, between the last stable and the first unstable timestep")
    rows = []
    for i, name in enumerate(names):
        for j, Da in enumerate(Da_values):
            lo[i, j], hi[i, j] = critical_dt(name, gridsteps, Da, **window(VARIANTS[name]["scheme"], Da))
            rows.append([name, Da, lo[i, j], hi[i, j]])
            print(f"  {name}, Da = {Da:g}: {lo[i, j]:.3e} to {hi[i, j]:.3e}")

    # (a) the largest stable dt, (b) for the explicit schemes the same in units of dl**2, which is the Fourier number dt/dl**2,
    # (c) for the implicit schemes in units of 1/Da, which is dt*Da. In those units the limits are of order one and the
    # bars (a few percent) are big enough to see.
    fig, (ax, ax_f, ax_k) = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    dl = 1/(gridsteps - 1)
    order = ["implicit, 2nd order, explicit R", "implicit, 1st order, explicit R", "implicit, 2nd order", "implicit, 1st order",
             "explicit, 2nd order", "explicit, 1st order"]   # aqua first: it coincides with blue, so blue is drawn on top
    for name in order:
        v = VARIANTS[name]
        i = names.index(name)
        size = 9 if v["color"] == AQUA else 6
        found = (lo[i] > 0) & np.isfinite(hi[i])
        up = (lo[i] > 0) & ~np.isfinite(hi[i])   # nothing unstable up to the largest timestep
        down = lo[i] == 0                        # already unstable at the smallest timestep
        with np.errstate(invalid="ignore"):   # the entries without a bracket contain inf, they are not used
            mid = np.sqrt(lo[i]*hi[i])
            err_lo, err_hi = mid - lo[i], hi[i] - mid

        def draw(a, factor, label=None):
            a.errorbar(Da_values[found], mid[found]*factor[found], yerr=[err_lo[found]*factor[found], err_hi[found]*factor[found]],
                       color=v["color"], marker=v["marker"], linestyle=v["ls"], markersize=size, capsize=3, label=label)
            a.plot(Da_values[up], lo[i][up]*factor[up], linestyle="none", marker="^", markersize=size + 1,
                   markerfacecolor="none", color=v["color"])
            a.plot(Da_values[down], hi[i][down]*factor[down], linestyle="none", marker="v", markersize=size + 1,
                   markerfacecolor="none", color=v["color"])

        draw(ax, np.ones(len(Da_values)), name)
        if v["scheme"] == "explicit":
            draw(ax_f, np.ones(len(Da_values))/dl**2)
        else:
            draw(ax_k, Da_values)

    ax.loglog()
    ax_f.semilogx()
    ax_k.semilogx()
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="^", markerfacecolor="none", color=INK2))
    labels.append("no instability up to the largest dt tried")
    ax.legend(handles, labels, fontsize=9, loc="upper right")
    ax.set_ylabel("largest stable timestep dt")
    ax.set_title("(a) Largest stable timestep")
    for level, text in ((1, "dt / dl² = 1"), (1/3, "dt / dl² = 1/3")):
        ax_f.axhline(level, color=MUTED, linewidth=1, zorder=1)
        ax_f.annotate(text, (Da_values[0], level), xytext=(0, 10), textcoords="offset points", color=INK2, fontsize=9)
    ax_f.set_ylim(0, 1.2)
    ax_f.set_ylabel("dt / dl²  (dl = grid spacing)")
    ax_f.set_title("(b) Explicit schemes: limit set by diffusion")
    ax_k.set_ylim(0, 10)
    ax_k.set_ylabel("dt · Da")
    ax_k.set_title("(c) Implicit schemes: limit set by the reaction")
    for a in (ax, ax_f, ax_k):
        mark_chosen_Da(a, Da_chosen)
        a.set_xlabel("Da")
    fig.suptitle(f"Stability of the six schemes ({gridsteps} gridpoints). Bars: uncertainty of the limit", fontsize=13)
    fig.tight_layout()
    save_table("03_stability_critical_dt.csv", ["scheme", "Da", "last stable dt", "first unstable dt"], rows)
    finish_figure(filename)
    return {name: {float(Da): (lo[i, j], hi[i, j]) for j, Da in enumerate(Da_values)} for i, name in enumerate(names)}


def graph_profiles(Da, tmax=0.1, gridsteps=100, dt=0.0001, analytical=False, filename=None):
    """Graphs 4 and 5: snapshots of the concentration profiles over time, from the second order implicit scheme.
    Without reaction (Da = 0) the analytical solution can be added to show that it agrees with the numerical solution.

    :param float Da: the Damköhler number
    :param float tmax: the total span of time from the beginning, default is 0.1
    :param int gridsteps: the number of gridsteps, default is 100
    :param float dt: the timestep, default is 0.0001
    :param bool analytical: if True the analytical solution is added, only for Da = 0
    :param str filename: name of the png file in the graphs folder
    :return: None
    :raises ValueError: if analytical is True and Da is not zero
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt, Da)
    if analytical and Da != 0:
        print("The analytical solution only exists for Da = 0!")
        raise ValueError

    x, t, Phi_A, Phi_B = determinePhi_impl(gridsteps, tmax, dt, Da)
    if analytical:
        title = f"Numerical (implicit, 2nd order) and analytical solution without reaction, dt = {dt:g}"
    else:
        title = f"Concentration profiles with reaction, Da = {Da:g}, dt = {dt:g}"
    plot(x, t, Phi_A, Phi_B, dt, analytical=analytical, filename=filename, title=title)


def fast_reaction_limit(x, tau):
    """Analytical solution for Da going to infinity, where the reaction is infinitely fast, so A and B can not be present at the
    same place. With equal diffusion coefficients and mirrored boundary conditions the front between them stays at x = 0.5,
    where A = B = 0. A is then pure diffusion on the half 0 <= x <= 0.5 with Phi_A(0) = 1 and Phi_A(0.5) = 0, which is the
    solution without reaction on a domain half as long, and zero on the other half. B is the mirror image.

    :param np.array(1D) x: the space coordinate, between 0 and 1
    :param float tau: the dimensionless time
    :return: the concentration profiles of A and B
    :rtype: tuple(np.array(1D), np.array(1D))
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))

    def phi_A(xi):
        A = np.zeros(len(xi))
        left = xi <= 0.5
        A[left] = analytical_solution(2*xi[left], 4*tau)[0]   # half the length, so x/L doubles and tau = t/L**2 is 4 times larger
        return A

    return phi_A(x), phi_A(1 - x)


def fast_reaction_limit_integrals(tau):
    """The wall flux of A, the total reaction rate and the amount of A for Da going to infinity, from the series of
    fast_reaction_limit(): flux = -dPhi_A/dx at x = 0, amount = integral of Phi_A over x, and
    total reaction rate = flux - d(amount)/dt, since A can only leave through the wall at x = 0 or react.

    :param float tau: the dimensionless time
    :return: the wall flux, the total reaction rate and the amount of A
    :rtype: tuple(float, float, float)
    """
    n = np.arange(1, int(np.sqrt(40/(4*np.pi**2*tau))) + 3)
    e = np.exp(-4*n**2*np.pi**2*tau)
    odd = n % 2 == 1
    flux = 2*(1 + 2*e.sum())
    d_amount = 8*e[odd].sum()
    amount = 0.25 - 2/np.pi**2*(e[odd]/n[odd]**2).sum()
    return flux, flux - d_amount, amount


def graph_why_Da(tmax=0.1, Da_chosen=1000, gridsteps=400, dt=0.00001, plot_gridsteps=100, filename="06_why_Da1000.png"):
    """Graph 6: the argument for Da = 1000. With reaction the concentrations of A and B cannot be present at the same place for
    large Da: the reaction is limited by how fast diffusion brings them together. There is an analytical solution for that
    limit of Da going to infinity (fast_reaction_limit()). The graph shows (a) that the wall flux of A and the total reaction rate
    have reached that limit at Da = 1000 (the reaction is diffusion limited, a larger Da changes them by less than 0.5%),
    (b) that the reaction zone at Da = 1000 is still resolved by the grid that is used in the other graphs, which
    is not true for a much larger Da, and (c) that the profile itself is closer to the limit than to the solution
    without reaction. Below Da = 10 the reaction does nothing, so the chosen Da has to be well above that.
    The stability limit on dt falls like 1/Da (graph 3), so a larger Da is also more expensive.

    :param float tmax: the total span of time from the beginning, default is 0.1
    :param float Da_chosen: the Da to argue for, default is 1000
    :param int gridsteps: the number of gridsteps of the runs, fine enough to resolve a thin reaction zone, default is 400
    :param float dt: the timestep of the runs, default is 0.00001
    :param int plot_gridsteps: the number of gridsteps that is used in the other graphs, to compare the width of the
        reaction zone with the grid spacing, default is 100
    :param str filename: name of the png file in the graphs folder
    :return: the numbers used in the argument: the ratios of the wall flux and the reaction rate to their limit at Da_chosen
        and the width of the reaction zone
    :rtype: dict
    :raises ValueError: if gridsteps is not int
    :raises ValueError: if gridsteps, tmax, dt is nonpositive
    :raises ValueError: if a timestep does not fit an integer number of times in tmax
    """

    #Define ValueErrors
    check_inputs(gridsteps, tmax, dt, 0)
    check_steps(tmax, dt)

    Da_values = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000])
    flux_limit, rate_limit, amount_limit = fast_reaction_limit_integrals(tmax)

    x, t, Phi_A, Phi_B = determinePhi_impl(gridsteps, tmax, dt, 0)   # no reaction: the analytical solution
    A_no_reaction = analytical_solution(x, t[-1])[0]
    dl = x[1] - x[0]

    flux = np.zeros(len(Da_values))
    rate = np.zeros(len(Da_values))
    width = np.zeros(len(Da_values))
    dev_limit = np.zeros(len(Da_values))
    dev_no_reaction = np.zeros(len(Da_values))
    A_limit = fast_reaction_limit(x, t[-1])[0]
    integrate = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed trapz
    rows = []
    for j, Da in enumerate(Da_values):
        x, t, Phi_A, Phi_B = determinePhi_impl(gridsteps, tmax, dt, Da)
        flux[j] = -(-3*Phi_A[-1, 0] + 4*Phi_A[-1, 1] - Phi_A[-1, 2])/(2*dl)   # second order one sided difference
        R = Da*Phi_A[-1]*Phi_B[-1]
        rate[j] = integrate(R, x)
        width[j] = rate[j]/R.max()   # the width of a block with the same height and area as the reaction zone
        dev_limit[j] = rmse(Phi_A[-1], A_limit)
        dev_no_reaction[j] = rmse(Phi_A[-1], A_no_reaction)
        rows.append([Da, flux[j], rate[j], width[j], dev_limit[j], dev_no_reaction[j]])

    k = list(Da_values).index(Da_chosen)
    numbers = {"flux_ratio": flux[k]/flux_limit, "rate_ratio": rate[k]/rate_limit, "width": width[k],
               "points_in_zone": width[k]/(1/(plot_gridsteps - 1)), "flux_limit": flux_limit, "rate_limit": rate_limit,
               "dev_limit": dev_limit[k], "dev_no_reaction": dev_no_reaction[k],
               "dev_no_reaction_Da10": dev_no_reaction[list(Da_values).index(10)],
               "width_slope": np.polyfit(np.log(Da_values[-4:]), np.log(width[-4:]), 1)[0],   # the last four Da
               "Da_rate_within_5_percent": Da_values[np.argmax(rate/rate_limit > 0.95)]}
    print(f"Graph 6: at Da = {Da_chosen:g}: wall flux {numbers['flux_ratio']:.4f} and reaction rate {numbers['rate_ratio']:.4f} "
          f"of the Da -> infinity limit, reaction zone width {width[k]:.3f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.4))
    ax1.semilogx(Da_values, flux/flux_limit, color=BLUE, marker="o", label="wall flux of A")
    ax1.semilogx(Da_values, rate/rate_limit, color=ORANGE, marker="s", label="total reaction rate")
    ax1.axhline(1, color=MUTED, linewidth=1, zorder=1)
    ax1.annotate("limit Da → ∞", (Da_values[0], 1), xytext=(2, 4), textcoords="offset points", color=INK2, fontsize=9)
    ax1.set_ylabel("value divided by its limit for Da → ∞")
    ax1.set_title("(a) Flux and reaction rate reach the limit")
    ax1.legend(fontsize=9, loc="lower right")

    ax2.loglog(Da_values, width, color=BLUE, marker="o", label="width of the reaction zone")
    ax2.axhline(1/(plot_gridsteps - 1), color=MUTED, linewidth=1.2, linestyle=":")
    ax2.annotate(f"grid spacing, {plot_gridsteps} points", (Da_values[-1], 1/(plot_gridsteps - 1)), xytext=(0, 4),
                 textcoords="offset points", ha="right", color=INK2, fontsize=9)
    ax2.set_ylabel("width of the reaction zone")
    ax2.set_title("(b) The reaction zone gets thinner")

    ax3.loglog(Da_values, dev_limit, color=BLUE, marker="o", label="compared to Da → ∞ limit")
    ax3.loglog(Da_values, dev_no_reaction, color=ORANGE, marker="s", label="compared to no reaction (Da = 0)")
    ax3.set_ylabel("RMSE of the profile of A")
    ax3.set_title("(c) The profile moves from one limit to the other")
    ax3.set_ylim(1e-3, 0.6)
    ax3.legend(fontsize=9, loc="upper center")

    for ax in (ax1, ax2, ax3):
        mark_chosen_Da(ax, Da_chosen)
        ax.set_xlabel("Da")
    fig.suptitle(f"Why Da = {Da_chosen:g}: the reaction is diffusion limited at t = {tmax:g} ({gridsteps} gridpoints)", fontsize=13)
    fig.tight_layout()
    save_table("06_why_Da1000.csv", ["Da", "wall flux of A", "total reaction rate", "reaction zone width",
                                    "RMSE vs Da to infinity", "RMSE vs no reaction"], rows)
    finish_figure(filename)
    return numbers


def write_graph_index(numbers=None, limits=None):
    """Writes graphs/README.txt: which graph shows what, with the argument for the chosen Da.

    :param dict numbers: the numbers returned by graph_why_Da(), default is None, then the argument is left out
    :param dict limits: the stability limits returned by graph_stability(), default is None, then they are left out
    :return: None
    """
    n = numbers
    cost = ""
    if limits is not None and 1000.0 in limits["implicit, 2nd order"] and 100000.0 in limits["implicit, 2nd order"]:
        lo3, hi3 = limits["implicit, 2nd order"][1000.0]
        lo5, hi5 = limits["implicit, 2nd order"][100000.0]
        cost = (f"  - The stability limit of the implicit schemes falls like 1/Da (graph 3): the largest stable dt of the second order\n"
                f"    scheme is between {lo3:.1e} and {hi3:.1e} at Da = 1000 and between {lo5:.1e} and {hi5:.1e} at Da = 100000, so a\n"
                f"    larger Da is much more expensive for integral results that do not change.\n")
    why = ("Run graph 6 to get the numbers.\n" if n is None else
           f"  - Up to Da = 10 the reaction hardly changes the profile (RMSE against Da = 0 is {n['dev_no_reaction_Da10']:.3f}, graph 6c), so the\n"
           f"    chosen Da has to be well above that.\n"
           f"  - For large Da the reaction is limited by how fast diffusion brings A and B together. At Da = 1000 the wall flux of A is\n"
           f"    {100*n['flux_ratio']:.2f}% and the total reaction rate {100*n['rate_ratio']:.1f}% of the analytical limit for Da -> infinity\n"
           f"    ({n['flux_limit']:.4f} and {n['rate_limit']:.4f}), so a larger Da changes these by less than 0.5%. The reaction rate is\n"
           f"    within 5% of the limit from Da = {n['Da_rate_within_5_percent']:g} (graph 6a).\n"
           f"  - What still changes is the thickness of the reaction zone, which falls like Da^({n['width_slope']:.2f}) at large Da. At Da = 1000 it is\n"
           f"    {n['width']:.2f}, so about {n['points_in_zone']:.0f} gridpoints of the 100 point grid lie inside it: it is resolved (graph 6b).\n"
           f"{cost}"
           f"  - The profile is closer to the Da -> infinity limit than to the solution without reaction (graph 6c), so Da = 1000\n"
           f"    tests the diffusion limited regime and not the trivial no reaction regime.\n"
           f"  Limits of this argument: the numbers are at t = 0.1; the limit solution assumes equal diffusion coefficients and mirrored\n"
           f"  boundary conditions (so the front stays at x = 0.5), as in this problem.\n")
    text = f"""GRAPHS - what each file shows. The numbers behind every graph are in the folder data (csv).

01_error_vs_analytical.png
    Error compared to the analytical solution, without reaction (Da = 0). (a) against the timestep, (b) against the number of
    gridpoints, for the implicit and explicit scheme, both first and second order in time. Combined RMSE of A and B at t = 0.1.
    The gray line in (a) is the error of the grid: below it the time error can not be seen in a comparison to the analytical
    solution. The implicit schemes are run up to dt = 0.01 so that their time error rises above it; the explicit schemes can
    only be run for small dt (stability). The slopes in the legends are the observed orders of convergence.

02_error_vs_Da.png
    Numerical error when Da increases (Da = 1 to 10000), for the implicit scheme with first and second order in time and with
    the reaction term semi-implicit or explicit. (a) a coarse timestep against a fine one, (b) a coarse grid against a fine grid.
    There is no analytical solution with reaction, so the reference is a fine numerical solution of the same scheme.

03_stability_critical_dt.png
    Largest stable timestep against Da for six schemes: explicit and implicit, each first and second order in time, and the
    implicit schemes with the reaction term explicit (R explicit) or semi-implicit. The bars are the uncertainty: the limit lies
    between the largest dt that is stable in a long run and the smallest dt found unstable in a shorter run (a slowly growing
    instability needs time to show, so the length of the run is the main uncertainty). The explicit schemes are limited by
    diffusion (flat in Da, at dl^2 and dl^2/3 for first and second order, with dl the grid spacing) until the reaction takes over
    at very large Da; the implicit schemes only by the reaction (about 1/Da). The semi-implicit and explicit reaction limits
    coincide for the same order.

04_profiles_vs_analytical_Da0.png
    Numerical (implicit, second order) concentration profiles over time without reaction, with the analytical solution as
    black dotted lines. Species A is solid, species B is dashed, time is the colour.

05_profiles_Da1000.png
    The same profiles with reaction, Da = 1000: A and B are consumed where they meet, in the middle.

06_why_Da1000.png
    The argument for choosing Da = 1000:
{why}"""
    os.makedirs(GRAPH_DIR, exist_ok=True)
    with open(os.path.join(GRAPH_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def make_graphs(tmax, Da, gridsteps, dt, run=None):
    """Makes the graphs in the graphs folder: the error against the analytical solution, the error against Da, the
    stability, the concentration profiles and the argument for the chosen Da, and writes graphs/README.txt.

    :param float tmax: the total span of time from the beginning
    :param float Da: the Damköhler number that is used for the profiles and that is argued for (graph 6)
    :param int gridsteps: the number of gridsteps for the profiles
    :param float dt: the timestep for the profiles
    :param dict run: graph name -> bool, which graphs to make: "error_analytical", "error_Da", "stability", "profiles",
        "why_Da". Default is None, all of them
    :return: None
    """
    run = {"error_analytical": True, "error_Da": True, "stability": True, "profiles": True, "why_Da": True, **(run or {})}

    if run["error_analytical"]:
        graph_error_vs_analytical(tmax, gridsteps)
    if run["error_Da"]:
        graph_error_vs_Da(tmax, Da_chosen=Da)
    limits = graph_stability(gridsteps, Da_chosen=Da) if run["stability"] else None
    if run["profiles"]:
        graph_profiles(0, tmax, gridsteps, 0.0001, analytical=True, filename="04_profiles_vs_analytical_Da0.png")
        graph_profiles(Da, tmax, gridsteps, 0.0001, filename=f"05_profiles_Da{Da}.png")
    numbers = graph_why_Da(tmax, Da_chosen=Da) if run["why_Da"] else None
    write_graph_index(numbers, limits)


def main():
    """Entry point: sets the parameters and runs the selected parts of the analysis.
    Switch the flags to choose what is run, since the error checkers are computationally expensive.
    The final graphs are saved as png files in the graphs folder next to this file, together with a README.txt that
    says what each graph shows and why Da = 1000 is used. The checks at the bottom print their results in the console."""

    global SHOW_PLOTS
    show_plots = False   # True also opens every graph in a window, False only saves the graphs
    SHOW_PLOTS = show_plots

    #-------------------------------------Parameters-----------------------------------------------------
    dt = 0.000025
    tmax = 0.1
    gridsteps = 100    # number of steps in the space direction
    Da = 1000 # Damkohler number. Why 1000: the reaction is then diffusion limited, see graph 6 and graphs/README.txt

    #-------------------------------------Final graphs, saved in the graphs folder-------------------------
    graphs = {"error_analytical": True,   # 01: error against the analytical solution (Da = 0), for dt and gridsteps
              "error_Da":         True,   # 02: numerical error when Da increases
              "stability":        True,   # 03: largest stable dt against Da, six schemes (takes a few minutes)
              "profiles":         True,   # 04, 05: concentration profiles without (with analytical solution) and with reaction
              "why_Da":           True}   # 06: the argument for Da = 1000
    make_graphs(tmax, Da, gridsteps, dt, graphs)

    #-------------------------------------Other checks, printed in the console------------------------------
    run_validation     = False   # compare to the analytical solution: Da = 0 directly, and A - B for the given Da
    run_stability      = False   # find the stability limit of the explicit scheme, first and second order
    run_stability_reaction = False  # find the stability limit of the implicit method for large dt*Da, all four variants
    run_impl_showcase  = False   # show implicit stability for increasing dt
    run_error_dt       = False   # convergence in dt
    run_error_grid     = False   # convergence in grid size
    run_error_Da       = False   # effect of the Damkohler number
    run_error_order    = False   # first vs second order in time, and semi-implicit vs explicit reaction term

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
