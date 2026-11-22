GRAPHS - what each file shows. The numbers behind every graph are in the folder data (csv).

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
  - Up to Da = 10 the reaction hardly changes the profile (RMSE against Da = 0 is 0.011, graph 6c), so the
    chosen Da has to be well above that.
  - For large Da the reaction is limited by how fast diffusion brings A and B together. At Da = 1000 the wall flux of A is
    100.00% and the total reaction rate 99.5% of the analytical limit for Da -> infinity
    (2.0772 and 1.9228), so a larger Da changes these by less than 0.5%. The reaction rate is
    within 5% of the limit from Da = 300 (graph 6a).
  - What still changes is the thickness of the reaction zone, which falls like Da^(-0.33) at large Da. At Da = 1000 it is
    0.27, so about 27 gridpoints of the 100 point grid lie inside it: it is resolved (graph 6b).
  - The stability limit of the implicit schemes falls like 1/Da (graph 3): the largest stable dt of the second order
    scheme is between 6.6e-03 and 7.0e-03 at Da = 1000 and between 4.2e-05 and 4.4e-05 at Da = 100000, so a
    larger Da is much more expensive for integral results that do not change.
  - The profile is closer to the Da -> infinity limit than to the solution without reaction (graph 6c), so Da = 1000
    tests the diffusion limited regime and not the trivial no reaction regime.
  Limits of this argument: the numbers are at t = 0.1; the limit solution assumes equal diffusion coefficients and mirrored
  boundary conditions (so the front stays at x = 0.5), as in this problem.
