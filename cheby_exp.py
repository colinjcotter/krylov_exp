from firedrake import *

class krylov_exp(object):
    def __init__(self, operator_solver, operator_in, operator_out):
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
