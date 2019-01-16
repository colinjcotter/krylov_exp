from cheby_exp import *
from firedrake import *

R = 6371220.
H = Constant(5960.)

ref_level = 3

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=3)
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
f = 2*Omega*z/Constant(R)  # Coriolis parameter
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

sparams = {
    'ksp_converged_reason':True
}

Prob = LinearVariationalProblem(a, F, operator_out)
OperatorSolver = LinearVariationalSolver(Prob, solver_parameters=sparams)

eigs = [0.003465, 0.007274, 0.014955]
days = 1
t = 60*60*days
L = eigs[ref_level]*t


cheby = cheby_exp(OperatorSolver, operator_in, operator_out,
                  ncheb, tol=1.0e-14, L)

x0 = Function(W)
ux, hx = x0.split()
hx.interpolate(exp((x+y+z)/R)*x*y*z/R**3)
y0 = Function(W)

cheby.apply(x0, y0, t)
