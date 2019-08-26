'''感知器学习算法,用于二分类
version = V1.0.0'''
import numpy as np

class Perceptron(object):
	"""Perceptron classifier"""
	def __init__(self,eta=0.01,n_iter=10):
		self.eta = eta
		self.n_iter = n_iter

	#感知器输出函数
	def net_input(self,x):
		return (np.dot(x,self.w_) + self.b_)

	#激活函数
	def predict(self,x):
		return np.where(self.net_input(x) >= 0.0,1,-1)

	#训练感知器
	def fit(self,X,y):
		self.w_ = np.random.random(X.shape[1])
		self.b_ = np.random.random(1)
		self.errors_ = []

		for _ in range(self.n_iter):
			errors = 0
			for xi,target in zip(X,y):
				update = self.eta * (target - self.predict(xi))
				self.w_ += update * xi
				self.b_ += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

