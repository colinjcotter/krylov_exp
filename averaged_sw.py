from cheby_exp import *
from firedrake import *

#some domain, parameters and FS setup
R = 6371220.
H = Constant(5960.)

ref_level = 4

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=3)
cx = SpatialCoordinate(mesh)
mesh.init_cell_orientations(cx)

cx, cy, cz = SpatialCoordinate(mesh)

outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)
    
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))

u, h = TrialFunctions(W)
v, phi = TestFunctions(W)

Omega = Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/Constant(R)  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant

c = sqrt(g*H)

#Set up the exponential operator
operator_in = Function(W)
u_in, h_in = split(operator_in)

u, h = TrialFunctions(W)
v, phi = TestFunctions(W)

F = (
    - inner(f*perp(u_in),v)*dx
    +c*h_in*div(v)*dx
    - c*div(u_in)*phi*dx
)

a = inner(v,u)*dx + phi*h*dx

assemble(a)
assemble(F)

operator_out = Function(W)

params = {
    'ksp_type': 'preonly',
    'pc_type': 'fieldsplit',
    'fieldsplit_0_ksp_type':'cg',
    'fieldsplit_0_pc_type':'sor',
    'fieldsplit_1_ksp_type':'preonly',
    'fieldsplit_1_pc_type':'ilu'
}

Prob = LinearVariationalProblem(a, F, operator_out)
OperatorSolver = LinearVariationalSolver(Prob, solver_parameters=params)

eigs = [0.003465, 0.007274, 0.014955]
min_time_period = 2*pi/eigs(ref_level-3)
hours = 6
dT = 60*60*hours
L = eigs[ref_level-3]*dT
Mbar = int(3*dT/min_time_period)
ncheb = 10000

cheby = cheby_exp(OperatorSolver, operator_in, operator_out,
                  ncheb, tol=1.0e-6, L=L)

#solvers for slow part
USlow_in = Function(W) #value at previous timestep
USlow_out = Function(W) #value at RK stage

u0, h0 = split(USlow_in)
u1, h1 = split(USlow_in)

#RHS for Forward Euler step
gradperp = lambda f: perp(grad(f))
n = FacetNormal(mesh)
L = (
    inner(v, u0)*dx + phi*h0*dx
    + dT*inner(gradperp(inner(v,u)))*dx
    - dT*inner(
)
