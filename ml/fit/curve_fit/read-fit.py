import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

#height = w ,center = μ,width = σ，offset = basevalue
def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, offset):
    return (gaussian(x, h1, c1, w1, offset=0) + gaussian(x, h2, c2, w2, offset=0) + gaussian(x, h3, c3, w3, offset=0) + offset)

def two_gaussians(x, h1, c1, w1, h2, c2, w2, offset):
    return three_gaussians(x, h1, c1, w1, h2, c2, w2, 0,0,1,offset)

# calculate err 
errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y)**2
errfunc2 = lambda p, x, y: (two_gaussians(x, *p) - y)**2

a = np.genfromtxt('../../rdf.txt')
data = a[128:187,:]

# get start value
guess2 = [0.38, 1.42, 0.6, 0.62, 1.54,0.6, 0]  

#optim3, success = optimize.leastsq(errfunc3, guess3[:], args=(data[:,0], data[:,1]))
optim2, success = optimize.leastsq(errfunc2, guess2[:], args=(data[:,0], data[:,1]))
optim2


plt.plot(data[:,0], data[:,1], lw=5, c='g', label='g(r)')
#plt.plot(data[:,0], three_gaussians(data[:,1], *optim3),lw=3, c='b', label='fit of 3 Gaussians')
plt.plot(data[:,0], two_gaussians(data[:,1], *optim2),lw=1, c='r', ls='--', label='fit of 2 Gaussians')
plt.legend(loc='best')
plt.savefig('result.png')

# calculate err
err2 = np.sqrt(errfunc2(optim2, data[:,0], data[:,1])).sum()
print('Residual error when fitting 2 Gaussians: {}'.format(err2))