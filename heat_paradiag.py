from firedrake import *  # noqa: F403
from firedrake.petsc import PETSc
import asQ
from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter

parser = ArgumentParser(
    description='ParaDiag timestepping for the heat equation.',
    formatter_class=DefaultsAndRawTextFormatter
)
parser.add_argument('--nx', type=int, default=32, help='Number of cells along each side of the square.')
parser.add_argument('--nlvls', type=int, default=3, help='Number of multigrid levels.')
parser.add_argument('--dt', type=float, default=0.02, help='Timestep size.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar space.')
parser.add_argument('--theta', type=float, default=1.0, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--nt', type=int, default=4, help='Number of timesteps to solve.')
parser.add_argument('--block_pc', type=str, default='lu', choices=('lu', 'jacobi', 'vanka', 'mg'), help='Block preconditioner.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows to solve.')
parser.add_argument('--slice_length', type=int, default=4, help='Number of timesteps per time-slice. Total number of timesteps in the all-at-once system is nslices*slice_length.')

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

# Multigrid square mesh hierarchy
base_nx = args.nx//(2**(args.nlvls-1))
base_mesh = UnitSquareMesh(nx=base_nx, ny=base_nx,
                           quadrilateral=True,
                           comm=ensemble.comm)
mesh_hierarchy = MeshHierarchy(base_mesh, args.nlvls-1)
mesh = mesh_hierarchy[-1]

# We use a continuous Galerkin space for the solution
V = FunctionSpace(mesh, "CG", args.degree)

# # # === --- initial conditions --- === # # #

x, y = SpatialCoordinate(mesh)

ics = Function(V).interpolate(sin(0.25*pi*x)*cos(2*pi*y))

# # # === --- finite element forms --- === # # #


# We add some forcing so that the solution
# doesn't just diffuse away.
def g(t):
    return sin(2*pi*x)*(cos(6*pi*t) + cos(3*pi*t-1)*sin(4*pi*y - 0.1))


# The time-derivative mass form for the heat equation.
# asQ assumes that the mass form is linear so here
# u is a TrialFunction and v is a TestFunction
def form_mass(u, v):
    return u*v*dx


# The CG form for the heat equation.
# asQ assumes that the function form is nonlinear so here
# u is a Function and v is a TestFunction
def form_function(u, v, t):
    return inner(grad(u), grad(v))*dx - inner(g(t), v)*dx


# Dirichlet condition on the left boundary,
# Neumann conditions assumed everywhere else.
bcs = [DirichletBC(V, 0, sub_domain=1)]


# # # === --- PETSc solver parameters --- === # # #

# The PETSc solver parameters used to solve the
# blocks in step (b) of inverting the ParaDiag matrix.

lu_params = {
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

jacobi_params = {
    'mat_type': 'aij',
    'ksp_type': 'richardson',
    'ksp_richardson_scale': 2/3,
    'pc_type': 'pbjacobi',
}

vanka_params = {
    'mat_type': 'aij',
    'ksp_type': 'gmres',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.ASMVankaPC',
    'pc_vanka': {
        'construct_dim': 0,
        'sub_sub_ksp_type': 'preonly',
        'sub_sub_pc_type': 'lu',
    },
}

mg_params = {
    'mat_type': 'aij',
    'ksp_type': 'richardson',
    'pc_type': 'mg',
    'pc_mg_type': 'multiplicative',
    'pc_mg_cycle_type': 'v',
    'mg_levels': vanka_params,
    'mg_levels_ksp_max_it': 3,
    'mg_coarse': lu_params,
}

if args.block_pc == 'lu':
    block_params = lu_params
elif args.block_pc == 'jacobi':
    block_params = jacobi_params
elif args.block_pc == 'vanka':
    block_params = vanka_params
elif args.block_pc == 'mg':
    block_params = mg_params

# The PETSc solver parameters for solving the all-at-once system.
# The python preconditioner 'asQ.CirculantPC' applies the ParaDiag matrix.
#
# The equation is linear so we can either:
# a) Solve it using a preconditioned Krylov method:
#    P^{-1}Au = P^{-1}b
#    The solver option for this is:
#    -ksp_type gmres
# b) Solve it with stationary iterations:
#    Pu_{k+1} = (P - A)u_{k} + b
#    The solver option for this is:
#    -ksp_type richardson

paradiag_parameters = {
    # all-at-once options
    'snes_type': 'ksponly',
    'ksp_rtol': 1e-10,
    'ksp_type': 'richardson',
    'ksp_converged_rate': None,
    # paradiag preconditioner options
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant': {
        'state': 'linear',
        'alpha': 1e-4,
        'block_ksp_rtol': 1e-5,
        'block': block_params,
    },
}

# # # === --- Setup all-at-once-system --- === # # #

# Give everything to the Paradiag object which will build the all-at-once system.
paradiag = asQ.Paradiag(ensemble=ensemble,
                        form_function=form_function,
                        form_mass=form_mass,
                        ics=ics, dt=args.dt, theta=args.theta,
                        time_partition=time_partition,
                        solver_parameters=paradiag_parameters)


# This function will be called before paradiag solves each time-window.
def window_preproc(paradiag, wndw, rhs):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')


PETSc.Sys.Print('### === --- Solving timeseries --- === ###')

# Solve nwindows of the all-at-once system
paradiag.solve(args.nwindows, preproc=window_preproc)

# # # === --- Solver diagnostics --- === # # #

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

# paradiag collects a few iteration counts for us
linear_its = paradiag.linear_iterations
block_its = paradiag.block_iterations.data()

# Number of linear iterations of the all-at-once system, total and per window.
PETSc.Sys.Print(f'Number of windows: {args.nwindows}')
PETSc.Sys.Print(f'Number of timesteps: {args.nwindows*window_length}')
PETSc.Sys.Print(f'All-at-once iterations:            {linear_its}')
PETSc.Sys.Print(f'All-at-once iterations per window: {linear_its/args.nwindows}')

# Number of iterations needed for each block in step-(b), total and per block solve
PETSc.Sys.Print(f'Total block iterations: {block_its}')
PETSc.Sys.Print(f'Maximum block linear iterations per solve: {np.max(block_its)/linear_its}')
PETSc.Sys.Print(f'Minimum block linear iterations per solve: {np.min(block_its)/linear_its}')
PETSc.Sys.Print('')
