from firedrake import *
import numpy as np
from scipy import fftpack

class cheby_exp(object):
    def __init__(self, operator_solver, operator_in, operator_out,
                 ncheb, tol, L):
        """
        Class to apply the exponential of an operator
        using chebyshev approximation

        operator_solver: a VariationalLinearSolver implementing
        the forward operator for Shift-and-Invert (used for residual
        calculation)
        operator_in: the input to operator_solver
        operator_out: the output to operator_solver
        ncheb: number of Chebyshev polynomials to approximate exp
        tol: tolerance to compress Chebyshev expansion by 
        (removes terms from the high degree end until total L^1 norm
        of removed terms > tol)
        L: approximate exp on range [-L*i, L*i]
        """

        self.operator_solver = operator_solver
        self.operator_in = operator_in
        self.operator_out = operator_out

        t1 = np.arange(np,pi, -dpi/2, -np.pi/(ncheb+1))
        x = L*np.cos(t1)
        thetas = np.concatenate((np.flipud(t1), -t1[1:-1]))
        valsUnitDisc = np.concatenate((np.flipud(fvals), fvals[1:-1]))
        FourierCoeffs = np.real(fftpack.fft(valsUnitDisc))/ncheb
        
        self.ChebCoeffs = FourierCoeffs[:ncheb+2]
        self.ChebCoeffs[0] = ChebCoeffs[0]/2
        self.ChebCoeffs[-1] = ChebCoeffs[-1]/2

        self.ncheb = ncheb

        FS = solver_in.function_space()
        self.Tm1_r = Function(FS)
        self.Tm1_i = Function(FS)
        self.Tm2_r = Function(FS)
        self.Tm2_i = Function(FS)
        self.T_r = Function(FS)
        self.T_i = Function(FS)

        self.dy = Function(FS)
        
    def apply(self, x, y, t):
        #T_0(x) = x^0 i.e. T_0(A) = I, T_0(A)U = U
        self.Tm1_r.assign(x)
        self.Tm1_i.assign(0)

        y.assign(0.)
        self.dy.assign(self.Tm1_r)
        self.dy *= self.ChebCoeffs[0]
        y += self.dy
        
        #T_0(x) = x^1/(i*L) i.e. T_1(A) = -i*A/L, T_0(A)U = -i*AU/L
        self.operator_in.assign(x)
        self.operator_solver.apply()
        self.T_r.assign(0)
        self.T_i.assign(self.operator_out)
        self.T_i *= -1/L

        self.dy.assign(self.T_r)
        self.dy *= self.ChebCoeffs[1]
        y += self.dy
        
        for i in range(2, self.ncheb+1):
            self.Tm2_r.assign(self.Tm1_r)
            self.Tm2_i.assign(self.Tm1_i)
            self.Tm1_r.assign(self.T_r)
            self.Tm1_i.assign(self.T_i)

            self.operator_in.assign(self.Tm1_r)
            self.operator_solver.apply()
            self.T_i.assign(self.operator_out)
            self.T_i *= -2/L
            self.operator_in.assign(self.Tm1_i)
            self.operator_solver.apply()
            self.T_r.assign(self.operator_out)
            self.T_r *= 2/L

            self.T_i -= self.Tm2_i
            self.T_r -= self.Tm2_r

            self.dy.assign(self.T_r)
            self.dy *= self.ChebCoeffs[i]
            y += self.dy
            
