from firedrake.petsc import PETSc
from firedrake import *
from slepc4py import SLEPc

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
f = 2*Omega*z/Constant(R)  # Coriolis parameter
g = Constant(9.8)  # Gravitational constant
hours = 2.
hour = 60*60
t = hours*hour

c = sqrt(g*H)

operator_in = Function(W)
u_in, h_in = split(operator_in)

a = (
    - inner(f*perp(u),v)*dx
    +c*h*div(v)*dx
    - c*div(u)*phi*dx
)

m = inner(v,u)*dx + phi*h*dx

petsc_a = assemble(a, mat_type='aij').M.handle
petsc_m = assemble(m, mat_type='aij').M.handle

num_eigenvalues = 1

opts = PETSc.Options()
opts.setValue("eps_gen_non_hermitian", None)
opts.setValue("st_pc_factor_shift_type", "NONZERO")
opts.setValue("eps_type", "krylovschur")
opts.setValue("eps_largest_imaginary", None)
opts.setValue("eps_tol", 1e-10)

es = SLEPc.EPS().create(comm=COMM_WORLD)
es.setDimensions(num_eigenvalues)
es.setOperators(petsc_a, petsc_m)
es.setFromOptions()
es.solve()

nconv = es.getConverged()
print("Converged n",nconv)

vr, vi = petsc_a.getVecs()

lam = es.getEigenpair(0, vr, vi)

print("Lambda", lam)
