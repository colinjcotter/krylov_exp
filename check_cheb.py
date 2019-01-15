from numpy import *
from scipy import fftpack
import matplotlib.pyplot as pyplot

n = 1000

L = 50

dpi = pi/(n+1)
t1 = arange(pi,0 - dpi/2,-pi/(n+1))
x = L*cos(t1)

#approximating exp(ix) on interval [-L,L]
#by \sum_i a_i T_i(ix)

fvals = real(exp(pi*1j*x))
thetas = concatenate((flipud(t1), -t1[1:-1]))
valsUnitDisc = concatenate((flipud(fvals), fvals[1:-1]))
FourierCoeffs = real(fftpack.fft(valsUnitDisc))/n

ChebCoeffs = FourierCoeffs[:n+2]
ChebCoeffs[0] = ChebCoeffs[0]/2
ChebCoeffs[-1] = ChebCoeffs[-1]/2

#reconstruct fourier
fvals0 = 0*valsUnitDisc

nf = 500
for i in range(nf):
    fvals0 += ChebCoeffs[i]*cos(i*thetas)

pyplot.plot(thetas,fvals0,'.')
pyplot.plot(thetas,valsUnitDisc,'-')
pyplot.show()


#Compute Forward transform using iterative formula
fvals0 = 0*fvals

#initialise T0

A = 1j*x
Tnm1 = 1.0
Tn = A/(L*1j)

nc = 500
fvals0 = real(ChebCoeffs[0]*Tnm1 + ChebCoeffs[1]*Tn)

for i in range(2,nc+1):
    Tnm2 = Tnm1
    Tnm1 = Tn
    Tn = 2*A*Tnm1/(L*1j) - Tnm2

    fvals0 += real(ChebCoeffs[i]*Tn)

pyplot.plot(x,fvals0,'-r')
pyplot.plot(x,fvals,'-k')
pyplot.show()
