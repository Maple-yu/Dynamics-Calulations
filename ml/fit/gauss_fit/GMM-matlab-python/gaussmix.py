#GMM --maple
import numpy as np
import pandas as pd
#import matplotlib.pylab as plt
from scipy import stats

# read data of rdf
rdf = np.loadtxt('../rdf.txt')

# fit and predict
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2).fit(rdf)
labels = gmm.predict(rdf)
labels

# separate the two peaks
predict_0 = [rdf[i] for i in range(len(rdf)) if labels[i] == 0]
predict_1 = [rdf[i] for i in range(len(rdf)) if labels[i] == 1]
pd.Series(predict_0).hist(bins=200)
pd.Series(predict_1).hist(bins=200)