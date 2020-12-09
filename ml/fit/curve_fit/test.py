import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm

#height = w ,center = μ,width = σ，offset = basevalue
def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

def two_gaussians(x, h1, c1, w1, h2, c2, w2, offset):
    return (gaussian(x, h1, c1, w1, offset=0) + gaussian(x, h2, c2, w2, offset=0) + offset)

# calculate err 
errfunc2 = lambda p, x, y: (two_gaussians(x, *p) - y)**2


# test data 
np.random.seed(100)
x = np.linspace(0, 3, 100)
y1 = norm.pdf(x, 1.2, 0.3)
y2 = norm.pdf(x, 1.8, 0.3)
y_noise = 0.02 * np.random.normal(size=x.size)
date = 0.4*y1 + 0.6*y2 + y_noise 
data = np.stack((x,date), axis=1)

# get start value 
guess2 = [0.4, 1.42, 0.3, 0.6, 1.54,0.3, 0]  

optim2, success = optimize.leastsq(errfunc2, guess2[:], args=(data[:,0], data[:,1]))
print(optim2)

# the two peaks
z1 = gaussian(x,optim2[0],optim2[1],optim2[2],0)
z2 = gaussian(x,optim2[3],optim2[4],optim2[5],0)
z = z1 + z2
# plt graph
plt.plot(data[:,0], data[:,1],'g.' ,label='original data')
#plt.plot(data[:,0] ,two_gaussians(data[:,1], *optim2) ,'r.', label='fitted')
plt.plot(data[:,0] ,z ,'b--', label='fitted data')
plt.plot(data[:,0],z1,'g-', label='fitted-peak1')
plt.plot(data[:,0],z2,'r-', label='fitted-peak2')
plt.legend(loc='best')
plt.savefig('test.png')

# calculate err
err2 = np.sqrt(errfunc2(optim2, data[:,0], data[:,1])).sum()
print('Residual error when fitting 2 Gaussians: {}'.format(err2))