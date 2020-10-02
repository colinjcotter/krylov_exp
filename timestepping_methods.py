#timestepping options for averaged_sw_explicit.py
#rk2/heuns/ssprk3/leapfrog

from firedrake import Function

def rk2(U, USlow_in, USlow_out, DU, V, W,
        expt, ensemble, cheby, cheby2, SlowSolver, wt, dt):
    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Step forward V
    V.assign(U + 0.5*V)

    #transform forwards to U^{n+1/2}
    cheby2.apply(V, DU, dt/2)

    #Average the nonlinearity
    cheby.apply(DU, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Advance U
    cheby2.apply(U, DU, dt/2)
    V.assign(DU + V)

    #transform forwards to next timestep
    cheby2.apply(V, U, dt/2)


def rk4(U, USlow_in, USlow_out, DU, U1, U2, U3, V1, V2, V3, V, W,
          expt, ensemble, cheby, cheby2, SlowSolver, wt, dt):
    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V1)
    #Step forward U1
    U1.assign(U + 0.5*V1)

    #Average the nonlinearity
    cheby2.apply(U1, DU, dt/2)
    cheby.apply(DU, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V2)
    #Step forward U2
    cheby2.apply(U, V, dt/2)
    U2.assign(V + 0.5*V2)

    #Average the nonlinearity
    cheby.apply(U2, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V3)
    #Step forward U1
    U3.assign(V + V3)

    #Average the nonlinearity
    cheby2.apply(V2, U1, dt/2)
    cheby2.apply(V3, U2, dt/2)

    cheby2.apply(U3, DU, dt/2)
    cheby.apply(DU, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U3)

    cheby2.apply(V1, DU, dt)
    cheby2.apply(U, V, dt)

    U.assign(V + 1/6*DU + 1/3*U1 + 1/3*U2 + 1/6*U3)


def heuns(U, USlow_in, USlow_out, DU, U1, U2, W,
          expt, ensemble, cheby, cheby2, SlowSolver, wt, dt):
    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U1)
    #Step forward U1
    U1.assign(U + U1)

    #Average the nonlinearity
    cheby2.apply(U1, DU, dt)
    cheby.apply(DU, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U2)
    #Step forward U2
    cheby2.apply(U, DU, dt)
    U2.assign(DU + U2)

    #transform forwards to next timestep
    cheby2.apply(U1, U, dt)
    U.assign(0.5*U + 0.5*U2)


def ssprk3(U, USlow_in, USlow_out, DU, U1, U2, W,
           expt, ensemble, cheby, cheby2, SlowSolver, wt, dt):
    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U1)
    #Step forward U1
    DU.assign(U + U1)
    cheby2.apply(DU, U1, dt)

    #Average the nonlinearity
    cheby.apply(U1, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U2)
    #Step forward U2
    DU.assign(U1 + U2)
    cheby2.apply(DU, U2, -dt/2)
    cheby2.apply(U, U1, dt/2)
    U2.assign(0.75*U1 + 0.25*U2)

    #Average the nonlinearity
    cheby.apply(U2, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, U1)
    #Advance U
    DU.assign(U2 + U1)
    cheby2.apply(DU, U2, dt/2)
    cheby2.apply(U, U1, dt)
    U.assign(1/3*U1 + 2/3*U2)


def leapfrog(U, USlow_in, USlow_out, U_old, U_new, DU, U1, U2, V, W,
             expt, ensemble, cheby, cheby2, SlowSolver, wt, dt, asselin):
    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Step forward V
    cheby2.apply(U_old, DU, dt)
    V.assign(DU + 2*V)
    cheby2.apply(V, U_new, dt)
    #Asselin filter
    cheby2.apply(U_old, U1, dt)
    cheby2.apply(U_new, U2, -dt)
    V.assign((U1+U2)*0.5 - U)
    U_old.assign(U + asselin*V)
    #Advance U
    U.assign(U_new)
