from cheby_exp import *
from firedrake import *
import numpy as np

from firedrake.petsc import PETSc
print = PETSc.Sys.Print

#checking cheby parameters based on ref_level
ref_level = 3
eigs = [0.003465, 0.007274, 0.014955] #maximum frequency
min_time_period = 2*pi/eigs[ref_level-3] 
hours = 3
dt = 60*60*hours
rho = 1.0 #averaging window is rho*dt

L = eigs[ref_level-3]*dt*rho
ppp = 3 #points per (minimum) time period

# rho*dt/min_time_period = number of min_time_periods that fit in rho*dt
# we want at least ppp times this number of sample points
Mbar = COMM_WORLD.size
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
                  ncheb, tol=1.0e-6, L=L)

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

ncycles = 2
dT = Constant(dt/ncycles)

L = (
    inner(v, u0)*dx + phi*eta0*dx
    #+ dT*inner(perp(grad(inner(v, perp(u0)))), u0)*dx
    #- dT*inner(both(perp(n)*inner(v, perp(u0))),
    #           both(Upwind*u0))*dS
    #+ dT*div(v)*K*dx
    #+ dT*inner(grad(phi), u0*(eta0-b))*dx
    #- dT*jump(phi)*(uup('+')*(eta0('+')-b('+'))
    #                - uup('-')*(eta0('-') - b('-')))*dS
)
#with topography, D = H + eta - b

SlowProb = LinearVariationalProblem(a, L, USlow_in)
SlowSolver = LinearVariationalSolver(SlowProb,
                                     solver_parameters = params)

t = 0.
tmax = 60.*60.*24.*15
dumpt = 60.*60.*6
tdump = 0.

svals = np.arange(0.5, Mbar)/Mbar #tvals goes from -rho*dt/2 to rho*dt/2
weights = np.exp(-1.0/svals/(1.0-svals))
weights = weights/np.sum(weights)
svals -= 0.5

rank = ensemble.ensemble_comm.rank
expt = rho*dt*svals[rank]
wt = weights[rank]

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
#b.interpolate(bexpr)

un1 = Function(V1)
etan1 = Function(V1)

U = Function(W)
DU = Function(W)
U_u, U_eta = U.split()
V = Function(W)
V_u, V_eta = V.split()

U_u.assign(un)
U_eta.assign(etan)

name = 'w2'
if rank==0:
    file_sw = File(name+'.pvd', comm=ensemble.comm)
    file_sw.write(un, etan, b)

nonlinear = True

print ('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += t

    if nonlinear:

        #first order splitting
        # U_{n+1} = \Phi(\exp(tL)U_n)
        #         = \exp(tL)(U_n + \exp(-tL)\Delta\Phi(\exp(tL)U_n))
        #averaged version
        # U_{n+1} = \exp(tL)(U_n + \int \rho\exp(-sL)\Delta\Phi(\exp(sL)U_n))ds

        #apply forward transformation and put result in V
        cheby.apply(U, V, expt)
        
        #apply forward slow step to V
        #using sub-cycled SSPRK2
        for i in range(ncycles):
            USlow_in.assign(V)
            SlowSolver.solve()
            USlow_in.assign(USlow_out)
            SlowSolver.solve()
            V.assign(0.5*(V + USlow_out))
        #compute difference from initial value
        V.assign(V-U)

        #apply backwards transformation, put result in DU
        cheby.apply(V, DU, -expt)
        DU *= wt

        #average into V
        ensemble.allreduce(DU, V)
        U += V

    V.assign(U)

    #transform forwards to next timestep
    cheby.apply(V, U, dt)

    if rank == 0:
        if tdump > dumpt - dt*0.5:
            un.assign(U_u)
            etan.assign(U_eta)
            file_sw.write(un, etan, b)
            tdump -= dumpt
