from krylov_exp import *
from cheby_exp import *
from firedrake import *

n = 50
R = 6371220.
H = Constant(5960.)

mesh = IcosahedralSphereMesh(radius=R, refinement_level=5, degree=3)
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
hours = 2.
hour = 60*60
t = hours*hour

c = sqrt(g*H)

operator_in = Function(W)
u_in, h_in = split(operator_in)

F = (
    - inner(f*perp(u_in),v)*dx
    +c*h_in*div(v)*dx
    - c*div(u_in)*phi*dx
)

a = inner(v,u)*dx + phi*h*dx

operator_out = Function(W)

params = {
    'ksp_type': 'preonly',
    'pc_type': 'fieldsplit',
    'fieldsplit_0_ksp_type':'preonly',
    'fieldsplit_1_pc_type':'lu',
    'fieldsplit_0_ksp_type':'preonly',
    'fieldsplit_1_pc_type':'lu'
}

Prob = LinearVariationalProblem(a, F, operator_out)
OperatorSolver = LinearVariationalSolver(Prob, solver_parameters=params)

max_sv(OperatorSolver, operator_in, operator_out)
