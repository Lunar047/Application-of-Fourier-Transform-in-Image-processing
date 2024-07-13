#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

dx = 0.075
# L = np.pi #We can change the limit of function from -pi to pi to any other domain -l to l as we want
xr =  np.arange(-1+dx,1+dx,dx) 
# f = np.array([0, 0, .1, .2, .25, .2, .25, .3, .35, .43, .45, .5, .55, .5, .4, .425, .45, .425, .4, .35, .3, .25, .225, .2, .1, 0, 0])
# g = np.array([0, 0, .025, .1, .2, .175, .2, .25, .25, .3, .32, .35, .375, .325, .3, .275, .275, .25, .225, .225, .2, .175, .15, .15, .05, 0, 0])
# g = g - 0.025 * np.ones_like(g)
g = xr*xr*xr
f = -xr*xr*np.sin(xr)

fig, ax = plt.subplots()
ax.plot(xr,f,'-',color='k')
ax.plot(xr,g,'-',color='b')
# f = abs(x)
# f = x*x*(np.sin(x)) 
# y = 3*x
# f = x*(np.sin(x))
# x = 0.1 * np.arange(1,len(f)+1)
# xf = np.arange(0.1,x[-1],0.01)

# f_interp = interpolate.interp1d(x, f, kind='cubic')
# g_interp = interpolate.interp1d(x, g, kind='cubic')

# ff = f_interp(xf)  
# gf = g_interp(xf)

# plt.plot(xf[10:-10],ff[10:-10],color='g')
# plt.plot(x[1:-2],f[1:-2],'o',color='y')

# plt.plot(xf[10:-10],gf[10:-10],color='k')
# plt.plot(x[1:-2],g[1:-2],'o',color='r')

plt.show()

# %%
