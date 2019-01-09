from krylov_exp import *
from firedrake import *

n = 20
mesh = UnitSquareMesh(n,n)
V = FunctionSpace(mesh, "DG", 1)

x, y = SpatialCoordinate(mesh)

q0 = Function(V).interpolate(exp(-((x-0.5)**2 + (y-0.25)**2)/0.25**2))

Vv = VectorFunctionSpace(mesh, "CG", 1)

from numpy import pi

U = Function(Vv).interpolate(as_vector([-sin(pi*x)*cos(pi*y),
                                        cos(pi*x)*sin(pi*y)]))

q = TrialFunction(V)
phi = TestFunction(V)

n = FacetNormal(mesh)
un = 0.5*(dot(U, n) + abs(dot(U, n)))

t = 1.0
gamma = Constant(t/10)

L = -inner(grad(phi), U*q)*dx + jump(phi)*(un('+')*q('+')
                                           - un('-')*q('-'))*dS

a = phi*q*dx + gamma*L

q_in = Function(V)
RHS = phi*q_in*dx

q_out = Function(V)

solver_parameters = {'ksp_type':'preonly',
                     'pc_type':'lu'}

Prob = LinearVariationalProblem(a, RHS, q_out)
Solver = LinearVariationalSolver(Prob, solver_parameters=solver_parameters)

qop_in = Function(V)
L = action(a, qop_in)
a = phi*q*dx
qop_out = Function(V)

Prob = LinearVariationalProblem(a, L, qop_out)
opSolver = LinearVariationalSolver(Prob, solver_parameters=solver_parameters)

exp_op = krylov_exp(Solver, q_in, q_out, gamma,
                    100, opSolver, qop_in, qop_out)

q1 = Function(V).assign(q0)

fileq = File('q.pvd')
fileq.write(q1)

exp_op.apply(q0, q1, t=t, its=2)

fileq.write(q1)

exp_op.apply(q0, q1, t=t, its=10)

fileq.write(q1)

exp_op.apply(q0, q1, t=t, its=20)

fileq.write(q1)

exp_op.apply(q0, q1, t=t, its=50)

fileq.write(q1)
