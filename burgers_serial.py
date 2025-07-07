from firedrake import *  # noqa: F403
from firedrake.petsc import PETSc
from utils.serial import SerialMiniApp
from argparse import ArgumentParser
from argparse_formatter import DefaultsAndRawTextFormatter

parser = ArgumentParser(
    description='Serial timestepping for Burgers equation.',
    formatter_class=DefaultsAndRawTextFormatter
)
parser.add_argument('--nx', type=int, default=100, help='Number of cells.')
parser.add_argument('--re', type=int, default=100, help='Reynolds number.')
parser.add_argument('--cfl', type=float, default=1.5, help='Courant number.')
parser.add_argument('--ubar', type=float, default=1.0, help='Reference velocity.')
parser.add_argument('--degree', type=int, default=2, help='Degree of the solution space.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--nt', type=int, default=8, help='Number of timesteps to solve.')

args = parser.parse_known_args()
args = args[0]
PETSc.Sys.Print(args)

# # # === --- domain --- === # # #

# 1D periodic mesh
mesh = PeriodicUnitIntervalMesh(args.nx)
x, = SpatialCoordinate(mesh)

# We use a continuous Galerkin space for the solution
V = FunctionSpace(mesh, "CG", args.degree)

# # # === --- initial conditions --- === # # #

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

# The PETSc solver parameters used to
# solve the serial-in-time method.

# The nonlinear solver is Newton iterations
# with full weighting (i.e. no line search).
# We just do a direct solve for the blocks
# because the problem is 1D and we're interested
# in experimenting with the nonlinear options.

serial_parameters = {
    # nonlinear options
    'snes_type': 'newtonls',
    'snes': {
        'linesearch_type': 'none',
        'converged_reason': None,
        'rtol': 1e-8,
        'stol': 0,
    },
    # linear options
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}


# # # === --- Setup the solver --- === # # #


# The SerialMiniApp class will set up the implicit-theta
# system for the serial-in-time method.
miniapp = SerialMiniApp(dt, args.theta, ics,
                        form_mass, form_function,
                        serial_parameters)

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
PETSc.Sys.Print('')
linear_its = 0
nonlinear_its = 0

from firedrake.output import VTKFile
uout = Function(V, name='u')
vtkfile = VTKFile("vtk/burgers.pvd")
uout.assign(ics)
vtkfile.write(uout, t=0)


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
    uout.assign(app.w1)
    vtkfile.write(uout, t=float(t))


PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')

# Solve nt timesteps
PETSc.Sys.Print(f"{norm(miniapp.w0) = :.3e}")
miniapp.solve(args.nt,
              preproc=preproc,
              postproc=postproc)
PETSc.Sys.Print(f"{norm(miniapp.w0) = :.3e}")

# # # === --- Solver diagnostics --- === # # #

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Iteration counts --- === ###')
PETSc.Sys.Print('')

# Number of linear iterations.
PETSc.Sys.Print(f'Nonlinear iterations:              {nonlinear_its}')
PETSc.Sys.Print(f'Nonlinear iterations per timestep: {nonlinear_its/args.nt}')
