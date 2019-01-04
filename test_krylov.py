from krylov_exp import *
from firedrake import *

from firedrake import *

n = 50
R = 6371220.
H = 5960.

mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=3)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

x, y, z = SpatialCoordinate(mesh)

outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)
    
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))

u, h = TrialFunctions(W)
v, phi = TestFunctions(W)

Omega = Constant(7.292e-5)  # rotation rate
f = 2*Omega*z/R  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
t = 60*60 #1 hour
gamma = Constant(t/10)

a = (
    inner(u,v)*dx - inner(f*gamma*perp(u),v)*dx
    +gamma*g*h*div(v)*dx
    +h*phi*dx - H*gamma*div(u)*phi*dx
)

solver_in = Function(W)
u_in, h_in = split(solver_in)

F = inner(v,u_in)*dx + phi*h_in*dx

solver_out = Function(W)

params = {
    'mat_type': 'matfree',
    #'ksp_converged_reason':True,
    #'ksp_monitor_true_residual': True,
    'ksp_type': 'fgmres',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.HybridizationPC',
    'hybridization': {'ksp_type': 'gmres',
                      #'ksp_monitor':True,
                      #'ksp_converged_reason':True,
                      'pc_type': 'gamg'}}

Prob = LinearVariationalProblem(a, F, solver_out)
Solver = LinearVariationalSolver(Prob, solver_parameters=params)

operator_in = Function(W)
u_in, h_in = split(solver_in)

F = (
    inner(u_in,v)*dx - inner(f*gamma*perp(u_in),v)*dx
    +gamma*g*h_in*div(v)*dx
    +h_in*phi*dx - H*gamma*div(u_in)*phi*dx
)

a = inner(v,u)*dx + phi*h*dx

operator_out = Function(W)

params = {
    'ksp_type': 'preonly',
    'pc_type': 'fieldsplit',
    'fieldsplit_0_ksp_type':'cg',
    'fieldsplit_1_pc_type':'ilu',
    'fieldsplit_0_ksp_type':'cg',
    'fieldsplit_1_pc_type':'ilu'
}

Prob = LinearVariationalProblem(a, F, operator_out)
OperatorSolver = LinearVariationalSolver(Prob, solver_parameters=params)

exp_op = krylov_exp(Solver, solver_in, solver_out, gamma,
                    40, OperatorSolver, operator_in, operator_out)

x0 = Function(W)
ux, hx = x0.split()
hx.interpolate(exp((x+y+z)/R)*x*y*z/R**3)
y0 = Function(W)

exp_op.apply(x0, y0, t=t)

uy, hy = y0.split()

File('exp.pvd').write(uy,hy)
