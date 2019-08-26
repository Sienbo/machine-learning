'''自适应神经元，用于分类
Version = V.0.0'''

import numpy as np

class AdlineGD(object):
	"""AdlineGD Classifier"""
	def __init__(self,eta=0.01,n_iter=50):
		self.eta = eta
		self.n_iter = n_iter

	#训练函数
	def fit(self,X,y):
		self.w_ = np.zeros(X.shape[1])
		self.b_ = np.zeros(1)
		self.cost_ = []

		for _ in range(self.n_iter):
			output = self.net_input(X)
			errors = y - output
			cost = (errors**2).sum()/2.0
			self.cost_.append(cost)
			self.w_ += self.eta * np.dot(X.T,errors)
			self.b_ += self.eta * errors.sum()
		return self


	#神经元输出
	def net_input(self,x):
		return np.dot(x,self.w_) + self.b_

	#激活函数
	def activation(self,x):
		return self.net_input(x)

	#预测函数
	def predict(self,x):
		return np.where(self.activation(x) > 0.0,1,-1)
