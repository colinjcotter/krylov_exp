from numpy import *
from scipy import fftpack
import matplotlib.pyplot as pyplot

n = 1000

L = 0.003465*60*60

dpi = pi/(n+1)
t1 = arange(pi,0 - dpi/2,-pi/(n+1))
x = L*cos(t1)

#approximating exp(ix) on interval [-L,L]
#by \sum_i a_i T_i(ix)

fvals = exp(1j*x)
thetas = concatenate((flipud(t1), -t1[1:-1]))
valsUnitDisc = concatenate((flipud(fvals), fvals[1:-1]))
FourierCoeffs = fftpack.fft(valsUnitDisc)/n

ChebCoeffs = FourierCoeffs[:n+2]
ChebCoeffs[0] = ChebCoeffs[0]/2
ChebCoeffs[-1] = ChebCoeffs[-1]/2

#reconstruct fourier
fvals0 = 0*valsUnitDisc

nf = 500
for i in range(nf):
    fvals0 += ChebCoeffs[i]*cos(i*thetas)

#pyplot.plot(thetas,real(fvals0),'.')
#pyplot.plot(thetas,real(valsUnitDisc),'-k')
#pyplot.show()

#pyplot.plot(thetas,imag(fvals0),'.')
#pyplot.plot(thetas,imag(valsUnitDisc),'-k')
#pyplot.show()


#Compute Forward transform using iterative formula
nc = 500

#initialise T0
A = 0.*1j
Tnm1 = 1.0+0.0j
Tn = A/(L*1j)
f0 = ChebCoeffs[0]*Tnm1 + ChebCoeffs[1]*Tn
for i in range(2,nc+1):
    Tnm2 = Tnm1
    Tnm1 = Tn
    Tn = 2*A*Tnm1/(L*1j) - Tnm2

    f0 += ChebCoeffs[i]*Tn

print(f0)
ChebCoeffs /= f0

fvals0 = 0*fvals
x2 = arange(-L,L,L*0.001)
A = 1j*x2
Tnm1 = 1.0
Tn = A/(L*1j)

fvals0 = ChebCoeffs[0]*Tnm1 + ChebCoeffs[1]*Tn

for i in range(2,nc+1):
    Tnm2 = Tnm1
    Tnm1 = Tn
    Tn = 2*A*Tnm1/(L*1j) - Tnm2

    fvals0 += ChebCoeffs[i]*Tn
    
print(absolute(fvals0).max())
    
#pyplot.plot(x,real(fvals),'-b')
pyplot.plot(x2,real(fvals0),'.r')
pyplot.plot(x2,sqrt(real(fvals0)**2 + imag(fvals0)**2),'--b')
pyplot.plot(x,cos(x),'--r')
#pyplot.plot(x,imag(fvals),'-g')
pyplot.plot(x2,imag(fvals0),'.k')
pyplot.plot(x,sin(x),'--k')
pyplot.show()

A = 0.1*array([[0,-1],[1,0]])
v = array([[1],[0]])

y = 0.*v

Tm1_r = 1.0*v
Tm1_i = 0.0*v
y += real(ChebCoeffs[0]*Tm1_r + ChebCoeffs[0]*Tm1_i)
T_r = 0.0*v
T_i = -dot(A,v)/L
y += real(ChebCoeffs[1]*T_r + ChebCoeffs[1]*T_i)

for i in range(2,nc+1):
    Tm2_r = 1.0*Tm1_r
    Tm2_i = 1.0*Tm1_i
    Tm1_r = 1.0*T_r
    Tm1_i = 1.0*T_i

    T_i = -2*dot(A,Tm1_r) - Tm2_i
    T_r = 2*dot(A,Tm1_i) - Tm2_r
    
    y += real(ChebCoeffs[i]*T_r + ChebCoeffs[i]*T_i)

