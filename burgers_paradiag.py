from firedrake import *  # noqa: F403
from firedrake.petsc import PETSc
import asQ
from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter

parser = ArgumentParser(
    description='ParaDiag timestepping for the heat equation.',
    formatter_class=DefaultsAndRawTextFormatter
)
parser.add_argument('--nx', type=int, default=100, help='Number of cells.')
parser.add_argument('--re', type=int, default=100, help='Reynolds number.')
parser.add_argument('--cfl', type=float, default=1.5, help='Courant number.')
parser.add_argument('--ubar', type=float, default=1.0, help='Reference velocity.')
parser.add_argument('--degree', type=int, default=2, help='Degree of the solution space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')

parser.add_argument('--aaos_pc', type=str, default='circulant', choices=('circulant', 'composite'), help='All-at-once preconditioner.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows to solve.')
parser.add_argument('--slice_length', type=int, default=8, help='Number of timesteps per time-slice. Total number of timesteps in the all-at-once system is nslices*slice_length.')

args = parser.parse_known_args()
args = args[0]
PETSc.Sys.Print(args)

# The time partition describes how many timesteps
# are included on each time-slice of the ensemble

time_partition = tuple(args.slice_length for _ in range(COMM_WORLD.size))
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

# The Ensemble with the spatial and time communicators
ensemble = asQ.create_ensemble(time_partition)

# # # === --- domain --- === # # #

# 1D periodic mesh
mesh = PeriodicUnitIntervalMesh(
    args.nx, comm=ensemble.comm)
x, = SpatialCoordinate(mesh)

# We use a continuous Galerkin space for the solution
V = FunctionSpace(mesh, "CG", args.degree)

# # # === --- initial conditions --- === # # #

x, = SpatialCoordinate(mesh)

ic_expr = Constant(args.ubar)*(1 + 0.5*sin(2*pi*x) + 0.25*cos(4*pi*x+0.3))
ics = Function(V).project(ic_expr)

# # # === --- finite element forms --- === # # #

x, = SpatialCoordinate(mesh)

ic_expr = Constant(args.ubar)*(1 + 0.5*sin(2*pi*x) + 0.25*cos(4*pi*x+0.3))
ics = Function(V).project(ic_expr)

# # # === --- finite element forms --- === # # #

# Calculate the viscosity and the timestep
# from the Reynolds and Courant numbers.
nu = Constant(args.ubar/args.re)
dt = args.cfl/(args.nx*args.ubar)


# We add some forcing so that the solution
# has some interesting features and doesn't
# eventually diffuse away.
def g(t):
    return Constant(0.5)*sin(2*pi*x)*(
        cos(2*pi*t+0) + Constant(0.5)*cos(4*pi*t-1)*sin(4*pi*x - 0.1))


# The time-derivative mass form for the burgers equation.
# asQ assumes that the mass form is linear so here
# u is a TrialFunction and v is a TestFunction
def form_mass(u, v):
    return u*v*dx


# The CG form for Burgers' equation.
# asQ assumes that the function form is nonlinear so here
# u is a Function and v is a TestFunction
def form_function(u, v, t):
    return (inner(dot(as_vector([u]), nabla_grad(u)), v)*dx
            + inner(nu*grad(u), grad(v))*dx
            - inner(g(t), v)*dx)


# # # === --- PETSc solver parameters --- === # # #

# Direct solver for the blocks
lu_params = {
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

# all-at-once preconditioners
circulant_params = {
    "pc_type": "python",
    "pc_python_type": "asQ.CirculantPC",
    "circulant_alpha": 1e-4,
    "circulant_block": lu_params,
}

composite_params = {
    "pc_composite_type": "multiplicative",
    "pc_composite_pcs": "python,python",
    "sub_0": {
        "pc_python_type": "asQ.CirculantPC",
        "circulant_alpha": 1e-4,
        "circulant_block": lu_params,
    },
    "sub_1": {
        "pc_python_type": "asQ.JacobiPC",
        "aaojacobi_block": lu_params,
    },
}

# The PETSc solver parameters for solving the all-at-once system.
# The python preconditioner 'asQ.CirculantPC' applies the ParaDiag matrix.

paradiag_parameters = {
    # nonlinear options
    "snes_type": "newtonls",
    "snes": {
        "linesearch_type": "none",
        "converged_reason": None,
        "rtol": 1e-8,
        "stol": 0,
    },
    # linear options
    "ksp_rtol": 1e-3,
    "ksp_type": "gmres",
}

if args.aaos_pc == 'circulant':
    aaos_params = circulant_params
elif args.aaos_pc == 'composite':
    aaos_params = composite_params
paradiag_parameters.update(aaos_params)

# # # === --- Setup all-at-once-system --- === # # #

# Give everything to the Paradiag object which will build the all-at-once system.
paradiag = asQ.Paradiag(ensemble=ensemble,
                        form_function=form_function,
                        form_mass=form_mass,
                        ics=ics, dt=dt, theta=args.theta,
                        time_partition=time_partition,
                        solver_parameters=paradiag_parameters)


# This function will be called before paradiag solves each time-window.
def window_preproc(paradiag, wndw, rhs):
    return
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')


PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')

# Solve nwindows of the all-at-once system
paradiag.solve(args.nwindows, preproc=window_preproc)

# # # === --- Solver diagnostics --- === # # #

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

# paradiag collects a few iteration counts for us
nonlinear_its = paradiag.nonlinear_iterations
linear_its = paradiag.linear_iterations

# Number of linear iterations of the all-at-once system, total and per window.
nT = args.nwindows*window_length
PETSc.Sys.Print(f'Number of windows: {args.nwindows}')
PETSc.Sys.Print(f'Number of timesteps: {nT}')
PETSc.Sys.Print(f"Linear iterations:    {paradiag.linear_iterations:>3d}")
PETSc.Sys.Print(f"Nonlinear iterations: {paradiag.nonlinear_iterations:>3d}")
PETSc.Sys.Print(f"Block solves per timestep: {paradiag.linear_iterations/nT:.2e}")
