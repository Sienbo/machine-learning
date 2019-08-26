'''自适应神经元，用于分类
Version = V.0.1
优化算法使用SGD代替BGD'''

import numpy as np
from numpy.random import seed

class AdlineSGD(object):
	"""AdlineSGD Classifier"""
	def __init__(self,eta=0.01,n_iter=10,shuffle=True,random_state=None):
		self.eta = eta
		self.n_iter = n_iter
		self.w_initialized = False
		self.shuffle = shuffle
		if random_state:
			seed(random_state)

	def fit(self,X,y):
		self._initialized_weights(X.shape[1])
		self.cost_ = []

		for _ in range(self.n_iter):
			if self.shuffle:
				X,y = self._shuffle(X,y)
			cost = []
			for xi,target in zip(X,y):
				cost.append(self._upgrade_weights(xi,target))
			avg_cost = np.mean(cost)
			self.cost_.append(avg_cost)
		return self

	def _initialized_weights(self,m):
		self.w_ = np.zeros(m)
		self.b_ = np.zeros(1)
		self.w_initialized = True

	def _shuffle(self,X,y):
		new_sort = np.random.permutation(len(y))
		return X[new_sort],y[new_sort]

	def _upgrade_weights(self,xi,target):
		output = self.net_input(xi)
		error = target - output
		self.w_ += self.eta * xi*error
		self.b_ += self.eta * error
		cost = (error**2)/2
		return cost

	def partial_fit(self,X,y):
		if not self.w_initialized:
			self._initialized_weights(X.shape[1])
		if y.ravel().shape[0] > 1:
			for xi,target in zip(X,y):
				self._upgrade_weights(xi,target)
		else:
			self._upgrade_weights(X,y)
		return self

	def net_input(self,x):
		return x.dot(self.w_) + self.b_

	def activation(self,X):
		return self.net_input(X)

	def predict(self,X):
		return np.where(self.activation(X) >= 0.0,1,-1)

