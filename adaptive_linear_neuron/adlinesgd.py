'''自适应神经元，用于分类
优化算法使用SGD代替BGD
Author = si'''

import numpy as np
from numpy.random import seed

class AdlineSGD(object):
	"""AdlineSGD Classifier
	参数：
		eta:float
			优化算法中的学习速率
		n_iter:int
			迭代次数的上限
		shuffle：Bool （default:True）
			打乱排序标识，用于每次优化迭代时打乱训练数据的排序
		random_state:int (default:None)


	属性：
		w_:array
			Adline算法线性函数的参数集
		b_:float
			线性函数的偏移量
		cost_:list
			训练过程中的损失值
		w_initialized：Bool （default:False）
	"""
	def __init__(self,eta=0.01,n_iter=10,shuffle=True,random_state=None):
		self.eta = eta
		self.n_iter = n_iter
		self.w_initialized = False
		self.shuffle = shuffle
		if random_state:
			seed(random_state)

	#训练过程
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

	#初始化参数
	def _initialized_weights(self,m):
		self.w_ = np.zeros(m)
		self.b_ = np.zeros(1)
		self.w_initialized = True

	#打乱数据排序
	def _shuffle(self,X,y):
		new_sort = np.random.permutation(len(y))
		return X[new_sort],y[new_sort]

	#更新权重
	def _upgrade_weights(self,xi,target):
		output = self.net_input(xi)
		error = target - output
		self.w_ += self.eta * xi*error
		self.b_ += self.eta * error
		cost = (error**2)/2
		return cost

	#用于在线算法训练时使用
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

