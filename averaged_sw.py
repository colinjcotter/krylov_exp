from cheby_exp import *
from firedrake import *
import numpy as np

from firedrake.petsc import PETSc
print = PETSc.Sys.Print

#picking cheby parameters based on ref_level
ref_level = 3
eigs = [0.003465, 0.007274, 0.014955]
min_time_period = 2*pi/eigs[ref_level-3]
hours = 6
dt = 60*60*hours
rho = 1.0 #averaging window is rho*dt

L = eigs[ref_level-3]*dt*rho
Mbar = int(3*rho*dt/min_time_period)
assert Mbar == COMM_WORLD.size, "Mbar = "+str(Mbar)+" "+str(COMM_WORLD.size)

print("We are off")

#ensemble communicator
ensemble = Ensemble(COMM_WORLD, 1)

#some domain, parameters and FS setup
R0 = 6371220.
H = Constant(5960.)

print("building mesh")
mesh = IcosahedralSphereMesh(radius=R0,
                             refinement_level=ref_level, degree=3,
                             comm = ensemble.comm)
cx = SpatialCoordinate(mesh)
mesh.init_cell_orientations(cx)

cx, cy, cz = SpatialCoordinate(mesh)

outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)
    
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))

u, eta = TrialFunctions(W)
v, phi = TestFunctions(W)

Omega = Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/Constant(R0)  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
b = Function(V2, name="Topography")
c = sqrt(g*H)

#Set up the exponential operator
operator_in = Function(W)
u_in, eta_in = split(operator_in)

u, eta = TrialFunctions(W)
v, phi = TestFunctions(W)

F = (
    - inner(f*perp(u_in),v)*dx
    +c*eta_in*div(v)*dx
    - c*div(u_in)*phi*dx
)

a = inner(v,u)*dx + phi*eta*dx

print("Forcing some assembly")
assemble(a)
assemble(F)

operator_out = Function(W)

params = {
    'ksp_type': 'preonly',
    'pc_type': 'fieldsplit',
    'fieldsplit_0_ksp_type':'cg',
    'fieldsplit_0_pc_type':'bjacobi',
    'fieldsplit_0_sub_pc_type':'ilu',
    'fieldsplit_1_ksp_type':'preonly',
    'fieldsplit_1_pc_type':'bjacobi',
    'fieldsplit_1_sub_pc_type':'ilu'
}

print("building solver for Cheby")
Prob = LinearVariationalProblem(a, F, operator_out)
OperatorSolver = LinearVariationalSolver(Prob, solver_parameters=params)

ncheb = 10000

cheby = cheby_exp(OperatorSolver, operator_in, operator_out,
                  ncheb, tol=1.0e-6, L=L)

#solvers for slow part
USlow_in = Function(W) #value at previous timestep
USlow_out = Function(W) #value at RK stage

u0, eta0 = split(USlow_in)
u1, eta1 = split(USlow_in)

#RHS for Forward Euler step
gradperp = lambda f: perp(grad(f))
n = FacetNormal(mesh)
Upwind = 0.5 * (sign(dot(u0, n)) + 1)
both = lambda u: 2*avg(u)
K = 0.5*inner(u0, u0)
uup = 0.5 * (dot(u0, n) + abs(dot(u0, n)))

ncycles = 2
dT = Constant(dt/ncycles)

L = (
    inner(v, u0)*dx + phi*eta0*dx
    + dT*inner(perp(grad(inner(v, perp(u0)))), u0)*dx
    - dT*inner(both(perp(n)*inner(v, perp(u0))),
               both(Upwind*u0))*dS
    + dT*div(v)*K*dx
    + dT*inner(grad(phi), u0*(eta0-b))*dx
    - dT*jump(phi)*(uup('+')*(eta0('+')-b('+'))
                    - uup('-')*(eta0('-') - b('-')))*dS
)
#with topography, D = H + eta - b

print("building solver for slow problem")
SlowProb = LinearVariationalProblem(a, L, USlow_in)
SlowSolver = LinearVariationalSolver(SlowProb,
                                     solver_parameters = params)

t = 0.
tmax = 60.*60.*24.*15

tvals = rho*(np.arange(0,Mbar*1.0)/(Mbar-1)-0.5)*dt
rank = ensemble.ensemble_comm.rank
expt = tvals[rank]

x = SpatialCoordinate(mesh)

u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = Function(V1).project(u_expr)
etan = Function(V2).interpolate(eta_expr)

# Topography
rl = pi/9.0
lambda_x = atan_2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = Min(pow(rl, 2), pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b.interpolate(bexpr)

un1 = Function(V1)
etan1 = Function(V1)

U = Function(W)
U_u, U_eta = U.split()
V = Function(W)
V_u, V_eta = V.split()

U_u.assign(un)
U_eta.assign(etan)

print("going for output")
if rank==0:
    file_sw = File('averaged_sw.pvd', comm=ensemble.comm)
    file_sw.write(un, etan)

while t < tmax + 0.5*dt:
    print(t)
    t += dt

    #apply forward transformation and put result in V
    print("cheby")
    cheby.apply(U, V, expt)
    
    #apply forward slow step to V
    for i in range(ncycles):
        print("forward "+str(i))
        USlow_in.assign(V)
        SlowSolver.solve()
        USlow_in.assign(USlow_out)
        SlowSolver.solve()
        V.assign(USlow_in)

    #apply backwards transformation, put result in U
    print("cheby backwards")
    cheby.apply(U, V, -expt)

    #average into V
    print("reduction")
    ensemble.allreduce(U, V)
    V /= Mbar

    #transform forwards to next timestep
    print("cheby forwards")
    cheby.apply(V, U, dt)

    un.assign(U_u)
    etan.assign(U_eta)

    if rank == 0:
        file_sw.write(un, etan)
