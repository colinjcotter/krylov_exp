from firedrake import *
import numpy as np

class krylov_exp(object):
    def __init__(self, solver, solver_in, solver_out, gamma,
                 kdim):
        """
        Class to apply the exponential of an operator
        using Shift-and-Invert (SAI) Krylov subspace scheme

        solver: The VariationalLinearSolver implementing the 
        Shift-and-Invert
        solver_in: The Function taken as input to solver
        solver_out: The Function taken as ouput to solver
        gamma: A Constant that specifies the shift parameter
        """

        self.solver = solver
        self.uin = solver_in
        self.uout = solver_out
        self.gamma = gamma

        self.kdim = kdim

        #allocate functions to store the Krylov subspace
        FS = solver_in.function_space()
        self.krylov_subspace = []
        for i in range(kdim):
            self.krylov_subspace.append(Function(FS))

        self.H = numpy.array((kdim,kdim))
            
    def apply(self, x, y):
        """
        takes x, applies exp, places result in y
        """

        for i in range(kdim):
            if i == 0:
                self.solver_in.assign(x)
            else:
                self.solver_in.assign(self.krylov_subspace[i-1])

            self.solver.solve()

            if i > 0:
                
