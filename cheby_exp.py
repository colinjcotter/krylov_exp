from firedrake import *
from numpy.random import randn

def max_sv(operator_solver, operator_in, operator_out):

    n = operator_in.dat.data[0][:].size
    operator_in.dat.data[0][:] = randn(n)
    n = operator_in.dat.data[1][:].size
    operator_in.dat.data[1][:] = randn(n)

    #power method on -L^2
 
    i = -1
    while True:
        i += 1
        norm = assemble(inner(operator_in, operator_in)*dx)**0.5
        print("norm", norm, "value", norm**0.5, i)
        operator_in /= -norm

        operator_solver.solve()
        operator_in.assign(operator_out)
        operator_solver.solve()
        operator_in.assign(operator_out)
