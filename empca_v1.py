from __future__ import division

import numpy as np 
from sklearn.decomposition import PCA, IncrementalPCA
#import tensorflow as tf 
X = np.random.random((3,4))


pca = PCA(n_components = 1)
inc_pnca = IncrementalPCA(n_components =1)
pca.fit(X.T)
phi = pca.components_
print phi
inc_pnca.fit(X.T)
print inc_pnca.components_

def EMPCA(X, n_var, n_obs, n_epochs):
	X = X-np.mean(X)
	assert X.shape ==(n_var, n_obs), "shape error in dataset"
	phi = np.random.rand(n_var)
	c = np.zeros(n_obs)
	for i in range(n_epochs): #repeat until convergence
		for j, x_j in enumerate(X.T): #E-step
			#print np.dot(x_j, phi)
			#print x_j.shape, phi.shape
			c[j] = np.dot(x_j,phi) 
			#print ",",(c*X).shape
			#print np.sum(c*X,axis=1).shape
		phi = np.sum(c*X, axis = 1)/np.sum((c**2))
		#print phi.shape
		phi = phi/np.linalg.norm(phi)
		#print phi
	return phi

phi_2 = EMPCA(X, X.shape[0],X.shape[1],1000)
print phi_2

