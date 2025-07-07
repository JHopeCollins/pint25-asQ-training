from firedrake import *  # noqa: F403
from firedrake.petsc import PETSc
from utils.serial import SerialMiniApp
from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter

parser = ArgumentParser(
    description='Serial timestepping for the heat equation.',
    formatter_class=DefaultsAndRawTextFormatter
)
parser.add_argument('--nx', type=int, default=32, help='Number of cells along each side of the square.')
parser.add_argument('--nlvls', type=int, default=2, help='Number of multigrid levels.')
parser.add_argument('--dt', type=float, default=0.02, help='Timestep size.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar space.')
parser.add_argument('--theta', type=float, default=1.0, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--nt', type=int, default=4, help='Number of timesteps to solve.')
parser.add_argument('--block_pc', type=str, default='lu', choices=('lu', 'jacobi', 'vanka', 'mg'), help='Block preconditioner.')

args = parser.parse_known_args()
args = args[0]
PETSc.Sys.Print(args)

# # # === --- domain --- === # # #

# Multigrid square mesh hierarchy
base_nx = args.nx//(2**(args.nlvls-1))
base_mesh = UnitSquareMesh(nx=base_nx, ny=base_nx,
                           quadrilateral=True)
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

# Different block solve options

# Direct solve
lu_params = {
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

# Diagonal Jacobi preconditioning
jacobi_params = {
    'mat_type': 'aij',
    'ksp_type': 'richardson',
    'ksp_richardson_scale': 2/3,
    'pc_type': 'jacobi',
}

# Vertex Vanka patch smoother
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

# Multigrid with Vanka patch smoother
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

# The PETSc solver parameters used to
# solve the serial-in-time method.
serial_parameters = {
    'snes_type': 'ksponly',
    'ksp': {
        'view': ':ksp_view.log',
        'converged_rate': None,  # show the contraction rate for the linear solve
        'rtol': 1e-6,
    },
}
if args.block_pc == 'lu':
    serial_parameters.update(lu_params)
elif args.block_pc == 'jacobi':
    serial_parameters.update(jacobi_params)
elif args.block_pc == 'vanka':
    serial_parameters.update(vanka_params)
elif args.block_pc == 'mg':
    serial_parameters.update(mg_params)


# # # === --- Setup the solver --- === # # #


# The SerialMiniApp class will set up the implicit-theta
# system for the serial-in-time method.
miniapp = SerialMiniApp(args.dt, args.theta, ics,
                        form_mass, form_function,
                        serial_parameters)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
PETSc.Sys.Print('')
linear_its = 0
nonlinear_its = 0


# This function will be called before solving each timestep.
def preproc(app, step, t):
    PETSc.Sys.Print(f'## === --- Calculating timestep {step} --- === ##')


# This function will be called after solving each timestep.
# We can use to record the number of iterations.
def postproc(app, step, t):
    global linear_its
    global nonlinear_its
    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()


PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')

# Solve nt timesteps
miniapp.solve(args.nt,
              preproc=preproc,
              postproc=postproc)

# # # === --- Solver diagnostics --- === # # #

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

# Number of linear iterations.
PETSc.Sys.Print(f'Block iterations:              {linear_its}')
PETSc.Sys.Print(f'Block iterations per timestep: {linear_its/args.nt}')
