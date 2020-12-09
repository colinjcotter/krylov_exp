from cheby_exp import *
from firedrake import *
from firedrake.petsc import PETSc
from math import pi
from math import ceil
from timestepping_methods import *
import numpy as np
import argparse

#get command arguments
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 3.')
parser.add_argument('--tmax', type=float, default=360, help='Final time in hours. Default 24x15=360.')
parser.add_argument('--dumpt', type=float, default=6, help='Dump time in hours. Default 6.')
parser.add_argument('--dt', type=float, default=2, help='Timestep in hours. Default 2.')
parser.add_argument('--rho', type=float, default=1, help='Averaging window width as a multiple of dt. Default 1.')
parser.add_argument('--linear', action='store_false', dest='nonlinear', help='Run linear model if present, otherwise run nonlinear model')
parser.add_argument('--Mbar', action='store_true', dest='get_Mbar', help='Compute suitable Mbar, print it and exit.')
parser.add_argument('--filter', type=bool, default=True, help='Use a filter in the averaging exponential')
parser.add_argument('--filter2', type=bool, default=False, help='Use a filter for cheby2')
parser.add_argument('--filter_val', type=float, default=0.75, help='Cut-off for filter')
parser.add_argument('--ppp', type=float, default=3, help='Points per time-period for averaging.')
parser.add_argument('--timestepping', type=str, default='ssprk3', choices=['rk2', 'rk4', 'heuns', 'ssprk3', 'leapfrog'], help='Choose a time steeping method. Default SSPRK3.')
parser.add_argument('--asselin', type=float, default=0.3, help='Asselin Filter coefficient. Default 0.3.')
parser.add_argument('--filename', type=str, default='explicit')
args = parser.parse_known_args()
args = args[0]
filter = args.filter
filter2 = args.filter2
filter_val = args.filter_val
timestepping = args.timestepping
asselin = args.asselin
ref_level = args.ref_level

#ensemble communicator
ensemble = Ensemble(COMM_WORLD, 1)

#parameters
R0 = 6371220.
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
g = Constant(9.8)  # Gravitational constant
mesh = IcosahedralSphereMesh(radius=R0,
                             refinement_level=ref_level, degree=3,
                             comm = ensemble.comm)
x = SpatialCoordinate(mesh)
global_normal = as_vector([x[0], x[1], x[2]])
mesh.init_cell_orientations(global_normal)
outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))
f = 2*Omega*x[2]/Constant(R0)  # Coriolis parameter
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = Function(V1, name="Velocity").project(u_expr)
etan = Function(V2, name="Elevation").project(eta_expr)

#topography (D = H + eta - b)
rl = pi/9.0
lambda_x = atan_2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = Min(pow(rl, 2), pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b = Function(V2, name="Topography")
b.interpolate(bexpr)

#set Mbar
eigs = [0.003465, 0.007274, 0.014955] #maximum frequency
min_time_period = 2*pi/eigs[ref_level-3]
hours = args.dt
dt = 60*60*hours
rho = args.rho #averaging window is rho*dt
L = eigs[ref_level-3]*dt*rho
ppp = args.ppp #points per (minimum) time period
               #rho*dt/min_time_period = number of min_time_periods that fit in rho*dt
               #we want at least ppp times this number of sample points
Mbar = ceil(ppp*rho*dt*eigs[ref_level-3]/2/pi)
print(args)
if args.get_Mbar:
    print("Mbar="+str(Mbar))
    import sys; sys.exit()

#set svals
svals = np.arange(0.5, Mbar)/Mbar #tvals goes from -rho*dt/2 to rho*dt/2
weights = np.exp(-1.0/svals/(1.0-svals))
weights = weights/np.sum(weights)
print(weights)
svals -= 0.5

#set weights
rank = ensemble.ensemble_comm.rank
expt = rho*dt*svals[rank]
wt = weights[rank]
print(wt,"weight",expt)

#parameters for timestepping
t = 0.
tmax = 60.*60.*args.tmax
dumpt = args.dumpt*60.*60.
tdump = 0.

#dump settings
print = PETSc.Sys.Print
assert Mbar==COMM_WORLD.size, str(Mbar)+' '+str(COMM_WORLD.size)
print('averaging window', rho*dt, 'sample width', rho*dt/Mbar)
print('Mbar', Mbar, 'samples per min time period', min_time_period/(rho*dt/Mbar))


##############################################################################
# Set up the exponential operator
##############################################################################

#apply the exponential of an operator using chebyshev approximation
operator_in = Function(W)
u_in, eta_in = split(operator_in)
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


##############################################################################
# Set up solvers for the slow part
##############################################################################
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

SlowProb = LinearVariationalProblem(a, L, USlow_out)
SlowSolver = LinearVariationalSolver(SlowProb,
                                     solver_parameters = params)


##############################################################################
# Time loop
##############################################################################
U = Function(W)
DU = Function(W)
U1 = Function(W)
U2 = Function(W)
U3 = Function(W)
W1 = Function(W)
W2 = Function(W)
W3 = Function(W)
V = Function(W)

U_u, U_eta = U.split()
U_u.assign(un)
U_eta.assign(etan)

#write out initial fields
name = args.filename
if rank==0:
    file_sw = File(name+'.pvd', comm=ensemble.comm)
    file_sw.write(un, etan, b)

#start time loop
print ('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    if t < dt*1.5 and timestepping == 'leapfrog':
        U_old = Function(W)
        U_new = Function(W)
        U_old.assign(U)
        rk2(U, USlow_in, USlow_out, DU, V, W,
            expt, ensemble, cheby, cheby2, SlowSolver, wt, dt)
    else:
        if timestepping == 'leapfrog':
            leapfrog(U, USlow_in, USlow_out, U_old, U_new, DU, U1, U2, V, W,
                     expt, ensemble, cheby, cheby2, SlowSolver, wt, dt, asselin)
        elif timestepping == 'ssprk3':
            ssprk3(U, USlow_in, USlow_out, DU, U1, U2, W,
                   expt, ensemble, cheby, cheby2, SlowSolver, wt, dt)
        elif timestepping == 'rk2':
            rk2(U, USlow_in, USlow_out, DU, V, W,
                expt, ensemble, cheby, cheby2, SlowSolver, wt, dt)
        elif timestepping == 'rk4':
            rk4(U, USlow_in, USlow_out, DU, U1, U2, U3, W1, W2, W3, V, W,
                expt, ensemble, cheby, cheby2, SlowSolver, wt, dt)
        elif timestepping == 'heuns':
            heuns(U, USlow_in, USlow_out, DU, U1, U2, W,
                  expt, ensemble, cheby, cheby2, SlowSolver, wt, dt)

    #dump
    if rank == 0:
        if tdump > dumpt - dt*0.5:
            un.assign(U_u)
            etan.assign(U_eta)
            file_sw.write(un, etan, b)
            print("dumped at t =", t)
            tdump -= dumpt

#check if dumbcheckpoint is working
valf = assemble(etan*dx)

print("create checkpointing file at rank =", rank)
chk = DumbCheckpoint("dump_explicit", mode=FILE_CREATE)
chk.store(un)
chk.store(etan)
chk.store(b)
chk.close()

und = Function(V1)
etand = Function(V2)
bd = Function(V2)

chk = DumbCheckpoint("dump_explicit", mode=FILE_READ)
chk.load(und, name="Velocity")
chk.load(etand, name="Elevation")
chk.load(bd, name="Topography")

valg = assemble(etand*dx)

print("valf = ", valf)
print("valg = ", valg)
assert(valf == valg)

chk.close()
