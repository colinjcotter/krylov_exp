from firedrake import *
from math import pi
from math import ceil
from latlon import *
import numpy as np
import argparse

#get command arguments
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator.')
parser.add_argument('--ref_level', type=int, default=4, help='Refinement level of icosahedral grid. Default 4.')
parser.add_argument('--tmax', type=float, default=120, help='Final time in hours. Default 24x5=120.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--checkt', type=float, default=6, help='Create checkpointing file every checkt hours. Default 6.')
parser.add_argument('--dt', type=float, default=45, help='Timestep for the standard model in seconds. Default 45.')
parser.add_argument('--filename', type=str, default='standard')
parser.add_argument('--pickup', action='store_true', help='Pickup the result from the checkpoint.')
parser.add_argument('--pickup_from', type=str, default='standard')
args = parser.parse_known_args()
args = args[0]
ref_level = args.ref_level
filename = args.filename
dt = args.dt
print(args)

#parameters
R0 = 6371220.
R = Constant(R0)
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
g = Constant(9.8)  # Gravitational constant
mesh = IcosahedralSphereMesh(radius=R0,
                             refinement_level=ref_level, degree=3)
x = SpatialCoordinate(mesh)
global_normal = as_vector([x[0], x[1], x[2]])
mesh.init_cell_orientations(global_normal)
outward_normals = CellNormal(mesh)
perp = lambda u: cross(outward_normals, u)
V1 = FunctionSpace(mesh, "BDM", 2)
V2 = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace((V1, V2))
f_expr = 2 * Omega * x[2] / R
Vf = FunctionSpace(mesh, "CG", 3)
f = Function(Vf).interpolate(f_expr)    # Coriolis frequency
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
h_expr = H-((R*Omega*u_max+u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g

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

#parameters for timestepping
t = 0.
tmax = 60.*60.*args.tmax
dumpt = args.dumpt*60.*60.
checkt = args.checkt*60.*60.
tdump = 0.
tcheck = 0.
k_max = 4        # Maximum number of Picard iterations

#pickup the result
if args.pickup:
    chkfile = DumbCheckpoint(args.pickup_from, mode=FILE_READ)
    un = Function(V1, name="Velocity")
    hn = Function(V2, name="Depth")
    chkfile.load(un, name="Velocity")
    chkfile.load(hn, name="Depth")
    etan = Function(V2, name="Elevation").assign(hn + b - H)
    t = chkfile.read_attribute("/", "time")
    tdump = chkfile.read_attribute("/", "tdump")
    tcheck = chkfile.read_attribute("/", "tcheck")
    chkfile.close()
else:
    un = Function(V1, name="Velocity").project(u_expr)
    hn = Function(V2, name="Depth").interpolate(h_expr)
    hn -= b
    etan = Function(V2, name="Elevation").assign(hn + b - H)

#setup PV solver
PV = Function(Vf, name="PotentialVorticity")
gamma = TestFunction(Vf)
q = TrialFunction(Vf)
D = hn
a = q*gamma*D*dx
L = (- inner(perp(grad(gamma)), un))*dx + gamma*f*dx
PVproblem = LinearVariationalProblem(a, L, PV)
PVsolver = LinearVariationalSolver(PVproblem, solver_parameters={"ksp_type": "cg"})
PVsolver.solve()

#write out initial fields
mesh_ll = get_latlon_mesh(mesh)
file_sw = File(filename+'_sw.pvd', mode="a")
field_un = Function(
    functionspaceimpl.WithGeometry(un.function_space(), mesh_ll),
    val=un.topological)
field_etan = Function(
    functionspaceimpl.WithGeometry(etan.function_space(), mesh_ll),
    val=etan.topological)
field_PV = Function(
    functionspaceimpl.WithGeometry(PV.function_space(), mesh_ll),
    val=PV.topological)
field_b = Function(
    functionspaceimpl.WithGeometry(b.function_space(), mesh_ll),
    val=b.topological)
if not args.pickup:
    file_sw.write(field_un, field_etan, field_PV, field_b)

##############################################################################
# Set up solvers
##############################################################################
# Set up depth advection solver (DG upwinded scheme)
up = Function(V1)
hp = Function(V2)
hpt = Function(V2)
h = TrialFunction(V2)
phi = TestFunction(V2)
hh = 0.5 * (hn + h)
uh = 0.5 * (un + up)
n = FacetNormal(mesh)
uup = 0.5 * (dot(uh, n) + abs(dot(uh, n)))
Heqn = ((h - hn)*phi*dx - dt*inner(grad(phi), uh*hh)*dx
        + dt*jump(phi)*(uup('+')*hh('+')-uup('-')*hh('-'))*dS)
Hproblem = LinearVariationalProblem(lhs(Heqn), rhs(Heqn), hpt)
lu_params = {'ksp_type': 'preonly',
             'pc_type': 'lu',
             'pc_factor_mat_solver_type': 'mumps'}
Hsolver = LinearVariationalSolver(Hproblem,
                                  solver_parameters=lu_params,
                                  options_prefix="H-advection")

# Velocity advection (Natale et. al (2016) extended to SWE)
upt = Function(V1)
u = TrialFunction(V1)
v = TestFunction(V1)
hh = 0.5 * (hn + hp)
ubar = 0.5 * (un + up)
uup = 0.5 * (dot(ubar, n) + abs(dot(ubar, n)))
uh = 0.5 * (un + u)
Upwind = 0.5 * (sign(dot(ubar, n)) + 1)
K = 0.5 * (inner(0.5 * (un + up), 0.5 * (un + up)))
both = lambda u: 2*avg(u)
outward_normals = CellNormal(mesh)
perp = lambda arg: cross(outward_normals, arg)
Ueqn = (inner(u - un, v)*dx + dt*inner(perp(uh)*f, v)*dx
        - dt*inner(perp(grad(inner(v, perp(ubar)))), uh)*dx
        + dt*inner(both(perp(n)*inner(v, perp(ubar))),
                   both(Upwind*uh))*dS
        - dt*div(v)*(g*(hh + b) + K)*dx)
Uproblem = LinearVariationalProblem(lhs(Ueqn), rhs(Ueqn), upt)
Usolver = LinearVariationalSolver(Uproblem,
                                  solver_parameters=lu_params,
                                  options_prefix="U-advection")

# Linear solver for incremental updates
HU = Function(W)
deltaU, deltaH = HU.split()
w, phi = TestFunctions(W)
du, dh = TrialFunctions(W)
alpha = 0.5
HUlhs = (inner(w, du + alpha*dt*f*perp(du))*dx
         - alpha*dt*div(w)*g*dh*dx
         + phi*(dh + alpha*dt*H*div(du))*dx)
HUrhs = -inner(w, up - upt)*dx - phi*(hp - hpt)*dx
HUproblem = LinearVariationalProblem(HUlhs, HUrhs, HU)
params = {'ksp_type': 'preonly',
          'mat_type': 'aij',
          'pc_type': 'lu',
          'pc_factor_mat_solver_type': 'mumps'}
HUsolver = LinearVariationalSolver(HUproblem,
                                   solver_parameters=params,
                                   options_prefix="impl-solve")

##############################################################################
# Time loop
##############################################################################
#start time loop
print('tmax', tmax, 'dt', dt)
while t < tmax - 0.5*dt:
    print(t)
    t += dt
    tdump += dt
    tcheck += dt

    up.assign(un)
    hp.assign(hn)

    #start picard cycle
    for i in range(k_max):
        #advect to get candidates
        Hsolver.solve()
        Usolver.solve()

        #linear solve for updates
        HUsolver.solve()

        #increment updates
        up += deltaU
        hp += deltaH

    #update fields for next time step
    un.assign(up)
    hn.assign(hp)

    #dumping results
    if tdump > dumpt - dt*0.5:
        #dump results
        PVsolver.solve()
        file_sw.write(field_un, field_etan, field_PV, field_b)
        #update dumpt
        print("dumped at t =", t)
        tdump -= dumpt

    #create checkpointing file every tcheck hours
    if tcheck > checkt - dt*0.5:
        print("checkpointing at t =", t)
        thours = int(t/3600)
        chk = DumbCheckpoint(filename+"_"+str(thours)+"h", mode=FILE_CREATE)
        chk.store(un)
        chk.store(hn)
        chk.write_attribute("/", "time", t)
        chk.write_attribute("/", "tdump", tdump)
        chk.write_attribute("/", "tcheck", tcheck)
        chk.close()
        tcheck -= checkt

print("Completed calculation at t =", t)
