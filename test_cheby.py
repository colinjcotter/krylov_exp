from cheby_exp import *
from firedrake import *

R = 6371220.
H = Constant(5960.)

ref_level = 3

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
    'fieldsplit_0_ksp_type':'preonly',
    'fieldsplit_0_pc_type':'sor',
    'fieldsplit_1_ksp_type':'preonly',
    'fieldsplit_1_pc_type':'ilu'
}

Prob = LinearVariationalProblem(a, F, operator_out)
OperatorSolver = LinearVariationalSolver(Prob, solver_parameters=params)

eigs = [0.003465, 0.007274, 0.014955]
hours = 0.1
t = 60*60*hours
L = eigs[ref_level-3]*t
ncheb = 1000

cheby = cheby_exp(OperatorSolver, operator_in, operator_out,
                  ncheb, tol=1.0e-5, L=L)

x0 = Function(W)
ux, hx = x0.split()
hx.interpolate(exp((cx+cy+cz)/R)*cx*cy*cz/R**3)
y0 = Function(W)

y0.assign(x0)
uy, hy = y0.split()
file0 = File('cheb.pvd')
file0.write(uy, hy)

print("doing cheby")
cheby.apply(x0, y0, t)

file0.write(uy, hy)

y0.assign(x0)

solver_in = Function(W)
u_in, h_in = split(solver_in)
solver_out = Function(W)

dt = t/1000
Dt = Constant(dt)

L = (
    - inner(f*perp(u),v)*dx
    +c*h*div(v)*dx
    - c*div(u)*phi*dx
)

a = inner(v,u)*dx + phi*h*dx - 0.5*Dt*L
F = action(inner(v,u)*dx + phi*h*dx + 0.5*Dt*L, solver_in)

assemble(F)

params = {
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.HybridizationPC',
    'hybridization': {'ksp_type': 'preonly',
                      'pc_type': 'lu'}}

Prob = LinearVariationalProblem(a, F, solver_out)
Solver = LinearVariationalSolver(Prob, solver_parameters=params)

#t0 = 0.
#solver_in.assign(x0)
#print("time integration")
#while(t0 < t + 0.5*dt):
#    print(t0)
#    t0 += dt
#    Solver.solve()
#    solver_in.assign(solver_out)
#y0.assign(solver_out)
#file0.write(uy, hy)
