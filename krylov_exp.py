from firedrake import *
import numpy as np
import scipy as sp

class krylov_exp(object):
    def __init__(self, solver, solver_in, solver_out, gamma,
                 kdim, operator_solver, operator_in, operator_out):
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
        self.solver_in = solver_in
        self.solver_out = solver_out
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
        self.H = np.zeros((kdim+1,kdim))

        #Working memory to avoid issues with mixed fields
        self.uw = Function(FS)
        
    def apply(self, x, y, t, its=-999):
        """
        takes x, applies exp(At), places result in y
        """

        def norm(x):
            return assemble(inner(x,x)*dx)**0.5

        beta = norm(x)
        gamma = self.gamma.evaluate(0,0,0,0)
        kdim = self.kdim
        w = self.solver_out
        V = self.krylov_subspace
        V[0].assign(x)
        V[0] /= beta

        if its < 0:
            its = self.kdim
        assert(its <= self.kdim)
        for j in range(its):
            self.solver_in.assign(self.krylov_subspace[j])
            self.solver.solve()

            for i in range(j+1):
                self.H[i,j] = assemble(inner(w, V[i])*dx)
                self.uw.assign(V[i])
                self.uw *= self.H[i,j]
                w -= self.uw
            self.H[j+1,j] = norm(w)
            V[j+1].assign(w)
            V[j+1] /= self.H[j+1,j]

            #exponential and residual calculation
            e1 = np.zeros((j+1,1))
            e1[0] = 1
            ej = np.zeros((j+1,1))
            ej[j] = 1
            H = self.H[:j+1,:j+1]
            Hinv = sp.linalg.inv(H)
            H1 = (Hinv - np.eye(j+1))/gamma
            s0   = np.array([1./3, 2./3, 1.])
            res = 0.*s0
            self.operator_in.assign(w)
            self.operator_solver.solve()
            C = norm(self.operator_out)
            for q, s in enumerate(s0):
                expHst = sp.linalg.expm(-H1*s*t)
                u = np.dot(expHst,e1)
                factor = np.dot(ej.T, np.dot(Hinv, u))
                res[q] = C/gamma*factor
            residual = sp.linalg.norm(res, ord=np.inf)
            print("Residual", residual)
        y.assign(0.)
        print(u)
        for j in range(its):
            self.uw.assign(V[j])
            self.uw *= u[j][0]
            y += self.uw

        y *= beta
