import numpy as np
import matplotlib.pyplot as plt
import sys
try:
    fn=sys.argv[1]
except:
    fn='log.dat'
d=np.loadtxt(fn)
##index, dx, fy, dfydx

t=np.argmin(d[:,2])
print(d[t,1])


t=np.argmin(np.abs(d[:,3]))
print(d[t,1])

#-0.122

dx = d[1,1]-d[0,1]

x1 = d[1:-1,1]
f1 = (d[2:,2]-d[:-2,2])/dx/2.0
plt.figure(0)
plt.plot(d[:,1],d[:,3],'r',label='from code')
plt.plot(x1,f1,'g--', label='numerical diff')
plt.plot(d[:,1], np.zeros_like(d[:,1]))
plt.legend()
plt.ylabel('dfy_dx (eV/Angstrom^2)')
#plt.xlim([0.0,10])

plt.figure(1)
df = d[1:-1,3]-f1
plt.plot(x1[:],df[:])
plt.ylabel('dfy_dx difference (eV/Angstrom)')
#plt.xlim([0.0,10])


plt.figure(2)
plt.plot(d[:,1], d[:,2])
plt.ylabel('fy (eV/Angstrom)')
#plt.xlim([0.0,10])

plt.show()
