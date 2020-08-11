#get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 3.')
parser.add_argument('--tmax', type=float, default=360, help='Final time in hours. Default 24x15=360.')
parser.add_argument('--dumpt', type=float, default=6, help='Dump time in hours. Default 6.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
parser.add_argument('--rho', type=float, default=1, help='Averaging window width as a multiple of dt. Default 1.')
parser.add_argument('--linear', action='store_false', dest='nonlinear', help='Run linear model if present, otherwise run nonlinear model')
parser.add_argument('--Mbar', action='store_true', dest='get_Mbar', help='Compute suitable Mbar, print it and exit.')
parser.add_argument('--filter', type=bool, default=True, help='Use a filter in the averaging exponential')
parser.add_argument('--filter2', type=bool, default=True, help='Use a filter for cheby2')
parser.add_argument('--filter_val', type=float, default=0.75, help='Cut-off for filter')
parser.add_argument('--ppp', type=float, default=3, help='Points per time-period for averaging.')
parser.add_argument('--filename', type=str, default='w2hw')
args = parser.parse_known_args()
args = args[0]

filter = args.filter
filter2 = args.filter2
filter_val = args.filter_val

#checking cheby parameters based on ref_level
ref_level = args.ref_level
eigs = [0.003465, 0.007274, 0.014955] #maximum frequency
from math import pi
min_time_period = 2*pi/eigs[ref_level-3]
hours = args.dt
dt = 60*60*hours
rho = args.rho #averaging window is rho*dt

L = eigs[ref_level-3]*dt*rho
ppp = args.ppp #points per (minimum) time period

# rho*dt/min_time_period = number of min_time_periods that fit in rho*dt
# we want at least ppp times this number of sample points
from math import ceil
Mbar = ceil(ppp*rho*dt*eigs[ref_level-3]/2/pi)
print(args)

if args.get_Mbar:
    print("Mbar="+str(Mbar))
    import sys; sys.exit()

from cheby_exp import *
from firedrake import *
import numpy as np

from firedrake.petsc import PETSc
print = PETSc.Sys.Print
assert Mbar==COMM_WORLD.size, str(Mbar)+' '+str(COMM_WORLD.size)
print('averaging window', rho*dt, 'sample width', rho*dt/Mbar)
print('Mbar', Mbar, 'samples per min time period', min_time_period/(rho*dt/Mbar))

#ensemble communicator
ensemble = Ensemble(COMM_WORLD, 1)

#some domain, parameters and FS setup
R0 = 6371220.
H = Constant(5960.)

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

#D = eta + b

u, eta = TrialFunctions(W)
v, phi = TestFunctions(W)

F = (
    - inner(f*perp(u_in),v)*dx
    +g*eta_in*div(v)*dx
    - H*div(u_in)*phi*dx
)

a = inner(v,u)*dx + phi*eta*dx

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

Prob = LinearVariationalProblem(a, F, operator_out)
OperatorSolver = LinearVariationalSolver(Prob, solver_parameters=params)

ncheb = 10000

cheby = cheby_exp(OperatorSolver, operator_in, operator_out,
                  ncheb, tol=1.0e-8, L=L, filter=filter, filter_val=filter_val)

cheby2 = cheby_exp(OperatorSolver, operator_in, operator_out,
                   ncheb, tol=1.0e-8, L=L, filter=filter2, filter_val=filter_val)

#solvers for slow part
USlow_in = Function(W) #value at previous timestep
USlow_out = Function(W) #value at RK stage

u0, eta0 = split(USlow_in)

#RHS for Forward Euler step
gradperp = lambda f: perp(grad(f))
n = FacetNormal(mesh)
Upwind = 0.5 * (sign(dot(u0, n)) + 1)
both = lambda u: 2*avg(u)
K = 0.5*inner(u0, u0)
uup = 0.5 * (dot(u0, n) + abs(dot(u0, n)))

dT = Constant(dt)

vector_invariant = True
if vector_invariant:
    L = (
        dT*inner(perp(grad(inner(v, perp(u0)))), u0)*dx
        - dT*inner(both(perp(n)*inner(v, perp(u0))),
                   both(Upwind*u0))*dS
        + dT*div(v)*K*dx
        + dT*inner(grad(phi), u0*(eta0-b))*dx
        - dT*jump(phi)*(uup('+')*(eta0('+')-b('+'))
                        - uup('-')*(eta0('-') - b('-')))*dS
        )
else:
    L = (
        dT*inner(div(outer(u0, v)), u0)*dx
        - dT*inner(both(inner(n, u0)*v), both(Upwind*u0))*dS
        + dT*inner(grad(phi), u0*(eta0-b))*dx
        - dT*jump(phi)*(uup('+')*(eta0('+')-b('+'))
                        - uup('-')*(eta0('-') - b('-')))*dS
        )

#with topography, D = H + eta - b

SlowProb = LinearVariationalProblem(a, L, USlow_out)
SlowSolver = LinearVariationalSolver(SlowProb,
                                     solver_parameters = params)

t = 0.
tmax = 60.*60.*args.tmax
dumpt = args.dumpt*60.*60.
tdump = 0.

svals = np.arange(0.5, Mbar)/Mbar #tvals goes from -rho*dt/2 to rho*dt/2
weights = np.exp(-1.0/svals/(1.0-svals))
weights = weights/np.sum(weights)
print(weights)
svals -= 0.5

rank = ensemble.ensemble_comm.rank
expt = rho*dt*svals[rank]
wt = weights[rank]
print(wt,"weight",expt)

x = SpatialCoordinate(mesh)

u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = Function(V1, name="Velocity").project(u_expr)
etan = Function(V2, name="Elevation").project(eta_expr)

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
DU = Function(W)
V = Function(W)

U_u, U_eta = U.split()
U_u.assign(un)
U_eta.assign(etan)

name = args.filename
if rank==0:
    file_sw = File(name+'.pvd', comm=ensemble.comm)
    file_sw.write(un, etan, b)

nonlinear = args.nonlinear

print ('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Step forward V by half a step
    V.assign(U + 0.5*V)

    #transform forwards to U^{n+1/2}
    cheby2.apply(V, DU, dt/2)

    #Average the nonlinearity
    cheby.apply(DU, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Advance U by half a step
    cheby2.apply(U, DU, dt/2)
    V.assign(DU + V)

    #transform forwards to next timestep
    cheby2.apply(V, U, dt/2)

    if rank == 0:
        if tdump > dumpt - dt*0.5:
            un.assign(U_u)
            etan.assign(U_eta)
            file_sw.write(un, etan, b)
            tdump -= dumpt
