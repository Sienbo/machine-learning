#coding:utf-8
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

def sigmoid(input):
	'''
	功能：
		Sigmoid激活函数
	输入：
		input：即神经元的加权和
	输出：
   		激活后的输出值
	'''
	return 1/(1 + np.exp(-input))

def derivationOfSigmoid(input):
	'''
	功能：
		对Sigmoid函数求导
	输入：
		input：Sigmoid函数的输入
	输出：
		Sigmoid函数的导数值
	'''
	return sigmoid(input) * (1 - sigmoid(input))

def relu(input):
	'''
	功能：
		relu激活函数
	输入：
		input：即神经元的加权和
	输出：
   		激活后的输出值
	'''
	return np.maximum(0, input)

def derivationOfRelu(input):
	'''
	功能：
		对relu函数求导
	输入：
		input：relu函数的输入
	输出：
		relu函数的导数值
	'''
	output = np.ones(input.shape)
	output[input < 0] = 0
	return output

class BpNeuralNetwork():
	'''
	概括：
		神经网络类，使用反向传播BP算法进行训练
		神经网络结构使用list表示，list的长度表示网络的层数，其中的值代表每一层网络的神经元个数；例如[3,4,5,2]表示一个神经网络：总共4层，
		输入层3个神经元、第一个隐藏层4个神经元、第二个隐藏层5个神经元，输出层为2个神经元
	输入：
		必选：
			layer_struct:神经网络结构；
		可选：
			learning_rate:学习速率；
			n_iters:最大迭代次数
			action_function:激活函数，默认为sigmoid函数，可选relu和tanh
			plot_flag:画损失值图标志；
	'''
	def __init__(self,layer_struct,
				action_function='sigmoid',
				learning_rate=0.05,n_iters=2000,
				cost_function="cross_entropy"):
		#神经网络结构
		self.layer_struct = layer_struct
		#网络的层数（包括输入层）
		self.layer_num = len(layer_struct)

		self.learning_rate = learning_rate
		self.n_iters = n_iters
		#选择激活函数
		self.activated_function = self.activated(action_function)
		self.derivation_function = self.activatedBackPropagation(action_function)
		self._w = dict()
		self._b = dict()
		self.cost_function = cost_function
		self.initWAndB()

	def activated(self,action_function):
		#选择激活函数
		if action_function == "sigmoid":
			return sigmoid
		elif action_function == "relu":
			return relu
		else:
			return sigmoid

	def activatedBackPropagation(self,action_function):
		if action_function == "sigmoid":
			return derivationOfSigmoid
		elif action_function == "relu":
			return derivationOfRelu
		else:
			return derivationOfSigmoid

	def initWAndB(self):
		'''
		功能：
			初始化神经网络参数w和b；
			输入层没有参数，从第二层开始到输出层，假设总共有n层，那么有
			W1~Wn-1,b1~bn
			使用字典来储存这些参数
		'''
		layer_num = self.layer_num
		layer_struct = self.layer_struct
		for i in range(1,layer_num):
			self._w["w" + str(i)] = np.random.randn(layer_struct[i],layer_struct[i-1])/np.sqrt(self.layer_struct[i-1])
			self._b["b" + str(i)] = np.zeros((layer_struct[i],1))
		return self._w,self._b

	def foward_propagation(self,X_train):
		'''
		功能：
			输入数据的前向传播
			层数从0开始
			ai表示第i层神经网络的输出
			zi表示为第i层神经网络的输入
			a1即训练数据
			没有z1
		'''
		cache = dict()
		lay_num = self.layer_num
		cache["a0"] = X_train
		for i in range(1,lay_num):
			cache["z" + str(i)] = np.dot(self._w["w" + str(i)],cache["a" + str(i-1)]) + self._b["b" + str(i)]
			cache["a" + str(i)] = self.activated_function(cache["z" + str(i)])
			cache["w" + str(i)] = self._w["w" + str(i)]
		return cache

	def backPropagation(self,caches):
		'''
		功能：
			反向传播算法
		输入：
			神经网络节点信息
		输出：
			反向传播的梯度
		'''
		layer_num = self.layer_num
		grads = {}
		#delta(L) = aL - y 
		grads["delta_" + str(layer_num - 1)] = caches["a" + str(layer_num - 1)] - self.y_train
		for i in reversed(range(1,layer_num-1)):
			grads["delta_" + str(i)] = np.multiply(np.dot(caches["w" + str(i + 1)].T,grads["delta_" + str(i + 1)]),
				self.derivation_function(caches["z"+str(i)]))
		return grads

	def calcCost(self,output):
		'''
		功能：
			计算损失值
			使用交叉熵函数
		'''
		#样本数目
		m = output.shape[1]
		error = -np.sum(np.multiply(np.log(output),self.y_train) + np.multiply(np.log(1 - output), 1 - self.y_train))/m
		return error

	def updateWAndB(self,grads,caches):
		for i in range(1,self.layer_num):
			self._w["w" + str(i)] -= self.learning_rate * np.dot(grads["delta_" + str(i)],caches["a" + str(i - 1)].T)
			self._b["b" + str(i)] -= self.learning_rate * np.sum(grads["delta_" + str(i)],axis=1).reshape(self.layer_struct[i],1)

	def fit(self,X_train,y_train):
		'''
		功能：
			训练神经网路
		输入：
			X_train：训练数据
			y_train：训练数据的分类数据
		'''
		#判断输入输出的维度是否正确
		n_input = X_train.shape[0]
		n_output = y_train.shape[0]
		m_x = X_train.shape[1]
		m_y = y_train.shape[1]
		#训练样本的输入输出和网络结构不符
		if self.layer_struct[0] != n_input or self.layer_struct[-1] !=n_output:
			raise KeyError
		#训练样本和标签样本数目不同
		if m_x != m_y:
			raise KeyError
		#训练使用样本数目
		self.X_train = X_train
		self.y_train = y_train

		self.cost_list = []
		#训练迭代过程
		for _ in range(self.n_iters):
			#前向传播
			caches = self.foward_propagation(X_train)
			output = caches["a" + str(self.layer_num - 1)]
			cost = self.calcCost(output)
			self.cost_list.append(cost)

			#反向传播
			grads = self.backPropagation(caches)
			#更新权值
			self.updateWAndB(grads,caches)

	def predict(self,x):
		'''
		功能：
			使用训练好的模型对样本进行预测
		输入：
			样本x
		输出：
			预测出来的类别
		'''
		caches = self.foward_propagation(x)
		output = caches["a" + str(self.layer_num - 1)]
		result = output / np.sum(output,axis=0,keepdims=True)
		return np.argmax(result,axis=0)

def plotCostValue(iter,cost_list):
	#画出训练完成后的误差曲线
	plt.plot(cost_list)
	plt.xlabel("iter_num")
	plt.ylabel("Cost Value")
	plt.title("BpNeuralNetwork.{0}".format(iter))

def plotDecisionBoundary(X_train,colors,pred_func):
	 # xy是坐标点的集合，把集合的范围算出来
    # 加减0.5相当于扩大画布的范围，不然画出来的图坐标点会落在图的边缘，逼死强迫症患者
    x1_min, x1_max = X_train[0, :].min() - 0.5, X_train[0, :].max() + 0.5
    x2_min, x2_max = X_train[1, :].min() - 0.5, X_train[1, :].max() + 0.5
    # 以h为分辨率，生成采样点的网格，就像一张网覆盖所有颜色点
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    # 把网格点集合作为输入到模型，也就是预测这个采样点是什么颜色的点，从而得到一个决策面
    Z = pred_func(np.array([xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # 利用等高线，把预测的结果画出来，效果上就是画出红蓝点的分界线
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    # 训练用的红蓝点点也画出来
    plt.scatter(X_train[0,:], X_train[1,:], c=colors, marker='o', cmap=plt.cm.Spectral, edgecolors='black')


if __name__ == "__main__":
    # 用sklearn的数据样本集，产生2种颜色的坐标点，noise是噪声系数，噪声越大，2种颜色的点分布越凌乱
	xy,colors = sklearn.datasets.make_moons(60,noise=0.2)

    # 因为点的颜色是1bit，我们设计一个神经网络，输出层有2个神经元。
    # 标定输出[1,0]为红色点，输出[0,1]为蓝色点
	expect_outputed = list()
	for c in colors:
		if c == 1:
			expect_outputed.append([0,1])
		else:
			expect_outputed.append([1,0])
	
	#训练数据的一列代表一个样本；即行数代表样本特征值个数，列数代表样本个数
	X_train = xy.T
	expect_outputed = np.array(expect_outputed).T

    # 设计3层网络，改变隐藏层神经元的个数，观察神经网络分类红蓝点的效果
	hidden_layer_neuron_num_list = [1,2,4,8,16,32]
	#生成一个图形对象，用于可视化
	plt.figure(figsize=(16,32),dpi=80)

	for i, hidden_layer_neuron_num in enumerate(hidden_layer_neuron_num_list):
		plt.subplot(2, 3, i + 1)
		plt.title('Num of hidden layer:{0}'.format(hidden_layer_neuron_num))

		#开始训练神经网络
		nn = BpNeuralNetwork(layer_struct=[2, hidden_layer_neuron_num, 2],action_function="sigmoid",learning_rate=0.02)
		nn.fit(X_train,expect_outputed)
		#画出边界图
		plotDecisionBoundary(X_train, colors, nn.predict)
		#画出损失值趋势图
		#plotCostValue(i,nn.cost_list)

	plt.show()
