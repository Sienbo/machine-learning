'''自适应神经元，用于分类
Author = si'''

import numpy as np

class AdlineGD(object):
	"""AdlineGD Classifier
	参数：
		eta:float
			优化算法中的学习速率
		n_iter:int
			迭代次数的上限
	属性：
		w_:array
			Adline算法线性函数的参数集
		b_:float
			线性函数的偏移量
		cost_:list
			训练过程中的损失值
	"""
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

	#数据的标准化
	def standardlize(self,X):
		std_X = np.copy(X)

		m,n = X.shape
		for i in range(n):
			std_X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()

		return std_X
