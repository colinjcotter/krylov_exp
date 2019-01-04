from firedrake import *
import numpy as np
import scipy as sp

class krylov_exp(object):
    def __init__(self, solver, solver_in, solver_out, gamma,
                 kdim, operator_solver, operator_in, operator_out)
        """
        Class to apply the exponential of an operator
        using Shift-and-Invert (SAI) Krylov subspace scheme

        solver: The VariationalLinearSolver implementing the 
        Shift-and-Invert
        solver_in: The Function taken as input to solver
        solver_out: The Function taken as ouput to solver
        gamma: A Constant that specifies the shift parameter
        
        kdim: the maximum dimension of the krylov subspace

        operator_solver: a VariationalLinearSolver implementing
        the forward operator for Shift-and-Invert (used for residual
        calculation)
        operator_in: the input to operator_solver
        operator_out: the output to operator_solver
        """

        self.solver = solver
        self.uin = solver_in
        self.uout = solver_out
        self.gamma = gamma

        self.kdim = kdim

        self.operator_solver = operator_solver
        self.operator_in = operator_in
        self.operator_out = operator_out
        
        #allocate functions to store the Krylov subspace
        FS = solver_in.function_space()
        self.krylov_subspace = []
        for i in range(kdim+1):
            self.krylov_subspace.append(Function(FS))

        #Array for hessenberg matrix
        self.H = numpy.array((kdim+1,kdim))

    def apply(self, x, t, y):
        """
        takes x, applies exp(At), places result in y
        """

        def norm(x):
            return assemble(inner(x,x)*dx)**0.5

        beta = norm(x)
        t = self.t
        gamma = self.gamma.evaluate(0,0,0,0)
        w = self.solver_out
        V = self.krylov_subspace
        V[0].assign(x/beta)
        for j in range(kdim):
            self.solver_in.assign(self.krylov_subspace[j])
            self.solver.solve()
            
            for i in range(j):
                self.H[i,j] = assemble(inner(w, V[i])*dx)
                w -= self.H[i,j]*V[i]

            self.H[j+1,j] = norm(w)

            V[j+1].assign(w/self.H[j+1,j])

            #exponential and residual calculation
            e1 = np.zeros(j); e1[0] = 1
            ej = np.zeros(j); ej[j] = 1
            Hinv = sp.linalg.inv(self.H[:j+1,:j+1])
            H1 = (Hinv - np.eye(j))/gamma
            s0   = [1./3, 2./3, 1.];
            res = 0.*s0
            for q, s in enumerate(s0):
                u = np.multiply(sp.linalg.sparse.expm(-H1*s*t),e1)*beta
                self.operator_in.assign(V[j+1])
                self.operator_solver.solve()
                factor = np.dot(ej, np.multiply(Hinv, u))
                res[q] = self.H[j+1,j]/gamma*self.operator_out*factor
            residual = np.norm(res, order=numpy.inf)
            print("Residual: ", residual)
        y.assign(0.)
        for j in range(self.kdim):
            y += u[j]*V[j]
