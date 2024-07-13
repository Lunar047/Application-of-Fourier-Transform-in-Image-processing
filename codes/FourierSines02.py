#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# plt.rcParams['figure.figsize'] = [8, 8]
# plt.rcParams.update({'font.size': 18})

# Define domain
dx = 0.001
L = np.pi #We can change the limit of function from -pi to pi to any other domain -l to l as we want
x = L * np.arange(-1+dx,1+dx,dx) # This represent the x axis as discontinuos point at dx interval
n = len(x) # no of points from -L to L at discreate value of dx intervals
nquart = int(np.floor(n/4)) #quarter of n
# for i in(x):
#     print(i,end='  ')
# Define hat function
f = np.zeros_like(x) #Gives zeros of same data type as of x 
# # print(f)
# f[nquart:2*nquart] = (4/n)*np.arange(1,nquart+1)
# f[2*nquart:3*nquart] = np.ones(nquart) - (4/n)*np.arange(0,nquart)
#let's try to replace f with our own function of f(x) = x*x
# f = x*x
# f = abs(x)
f = x*x*(np.sin(x)) 
# y = 3*x
# f = x*(np.sin(x))


# f = 3*x
# f[nquart:3*nquart] = 1
#Again let's chage f to 
fig, ax = plt.subplots()
ax.plot(x,f,'-',color='k')
# ax.plot(x,y,'-',color = 'r')
# Compute Fourier series
name = "Accent"
cmap = get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle(color=colors)

A0 = np.sum(f * np.ones_like(x)) * dx
fFS = A0/2

A = np.zeros(200)
B = np.zeros(200)
# Summing the series of sine and consine with inner product of f(x)
for k in range(0,5):
    A[k] = np.sum(f * np.cos(np.pi*(k+1)*x/L)) * dx # Inner product
    B[k] = np.sum(f * np.sin(np.pi*(k+1)*x/L)) * dx
    fFS = fFS + A[k]*np.cos((k+1)*np.pi*x/L) + B[k]*np.sin((k+1)*np.pi*x/L)
ax.plot(x,fFS,'-')

#%%
## Plot amplitudes

fFS = (A0/2) * np.ones_like(f)
kmax = 100
A = np.zeros(kmax)
B = np.zeros(kmax)
ERR = np.zeros(kmax)

A[0] = A0/2
ERR[0] = np.linalg.norm(f-fFS)/np.linalg.norm(f)

for k in range(1,kmax):
    A[k] = np.sum(f * np.cos(np.pi*k*x/L)) * dx
    B[k] = np.sum(f * np.sin(np.pi*k*x/L)) * dx
    fFS = fFS + A[k] * np.cos(k*np.pi*x/L) + B[k] * np.sin(k*np.pi*x/L)
    ERR[k] = np.linalg.norm(f-fFS)/np.linalg.norm(f)
    
thresh = np.median(ERR) * np.sqrt(kmax) * (4/np.sqrt(3))
r = np.max(np.where(ERR > thresh))

fig, axs = plt.subplots(2,1)
axs[0].semilogy(np.arange(kmax),A,color='k')
axs[0].semilogy(r,A[r],'o',color='b')
plt.sca(axs[0])
plt.title('Fourier Coefficients')

axs[1].semilogy(np.arange(kmax),ERR,color='k')
axs[1].semilogy(r,ERR[r],'o',color='b')
plt.sca(axs[1])
plt.title('Error')

plt.show()

#%%
# Gibbs Phenomenaa
dx = 0.01
L = 2*np.pi
x = np.arange(0,L+dx,dx)
n = len(x)
nquart = int(np.floor(n/4))

f = np.zeros_like(x)
f[nquart:3*nquart] = 1

A0 = np.sum(f * np.ones_like(x)) * dx * 2 / L
fFS = A0/2 * np.ones_like(f)

for k in range(1,101):
    Ak = np.sum(f * np.cos(2*np.pi*k*x/L)) * dx * 2 / L
    Bk = np.sum(f * np.sin(2*np.pi*k*x/L)) * dx * 2 / L
    fFS = fFS + Ak*np.cos(2*k*np.pi*x/L) + Bk*np.sin(2*k*np.pi*x/L)
    
plt.plot(x,f,color='k')
plt.plot(x,fFS,'-',color='r')
plt.show()
# %%
#temp
import numpy as np
import matplotlib.pyplot as plt
limit = 20
x = np.linspace(1,100,limit)
y = np.linspace(1,100,limit)
x,y = np.meshgrid(x,y)
z = x+y
plt.imshow(z,cmap="Greens")

