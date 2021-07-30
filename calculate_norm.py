from firedrake import *
import numpy as np
import argparse
from firedrake.petsc import PETSc
print = PETSc.Sys.Print

#get command arguments
parser = argparse.ArgumentParser(description='Williamson 5 testcase for averaged propagator.')
parser.add_argument('--file0', type=str, default='standard0')
parser.add_argument('--file1', type=str, default='standard1')
args = parser.parse_known_args()
args = args[0]
print(args)

# parameters
REF_LEVEL = 5
HOURS = [6, 12, 18, 24]

#parameters
R0 = 6371220.
R = Constant(R0)
H = Constant(5960.)
Omega = Constant(7.292e-5)  # rotation rate
g = Constant(9.8)  # Gravitational constant
mesh = IcosahedralSphereMesh(radius=R0,
                             refinement_level=REF_LEVEL, degree=3)
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
eta_expr = -((R*Omega*u_max+u_max*u_max/2.0)*(x[2]*x[2]/(R*R)))/g
h_expr = eta_expr + H

#calculate norms for debug
uini = Function(V1, name="Velocity0").project(u_expr)
etaini = Function(V2, name="Elevation0").interpolate(eta_expr)
etaavg = 0
uavg = 0
etastd = 0
ustd = 0

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

print('calculate normalised norms in '+str(args.file0)+" with respect to "+str(args.file1)+" at hours", HOURS)
eta_norm = []
u_norm_Hdiv = []
u_norm_L2 = []
for hour in HOURS:
    t = int(hour)

    #read data from file0
    chkfile0 = DumbCheckpoint(args.file0+"_"+str(t)+"h", mode=FILE_READ)
    u0 = Function(V1, name="Velocity")
    eta0 = Function(V2, name="Elevation")
    chkfile0.load(u0, name="Velocity")
    chkfile0.load(eta0, name="Elevation")
    chkfile0.close()
    uavg = errornorm(u0, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
    etaavg = errornorm(eta0, etaini)/norm(etaini)
    print('etaavg', etaavg, 'uavg', uavg)

    #read data from file1
    chkfile1 = DumbCheckpoint(args.file1+"_"+str(t)+"h", mode=FILE_READ)
    u1 = Function(V1, name="VelocityR")
    h1 = Function(V2, name="DepthR")
    chkfile1.load(u1, name="Velocity")
    chkfile1.load(h1, name="Depth")
    eta1 = Function(V2, name="ElevationR").assign(h1 + b - H)
    chkfile1.close()
    ustd = errornorm(u1, uini, norm_type="Hdiv")/norm(uini, norm_type="Hdiv")
    etastd = errornorm(eta1, etaini)/norm(etaini)
    print('etastd', etastd, 'ustd', ustd)

    #calculate norms
    etanorm = errornorm(eta0, eta1)/norm(eta1)
    unorm_Hdiv = errornorm(u0, u1, norm_type="Hdiv")/norm(u1, norm_type="Hdiv")
    unorm_L2 = errornorm(u0, u1)/norm(u1)

    #append norms in array
    eta_norm.append(etanorm)
    u_norm_Hdiv.append(unorm_Hdiv)
    u_norm_L2.append(unorm_L2)

print('etanorm =', eta_norm)
print('unorm_Hdiv =', u_norm_Hdiv)
print('unorm_L2 =', u_norm_L2)
