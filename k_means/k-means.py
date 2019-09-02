'''
date:2019-09-01
@Author:si
'''
import numpy as np
import pandas as pd

class Kmean():
	'''
	聚类算法
	参数：
		n_clusters：int
			簇数
		max_iter:int (default:100)
			最大迭代次数
		init:str (defalut:"random")
			中心点初始化方式
	属性：
		center：dict
			簇中心点的坐标，字典内key代表簇名，value该中心点的坐标
		cluster：dict
			分簇后的结果，字典内key代表簇名，value是簇内数据的索引列表
	'''
	def __init__(self,n_clusters,max_iter=10,init="random"):
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.init = init

	def fit(self,X):
		#初始化簇中心点和簇内点
		self.center = self.initial_center(X)
		self.cluster = self.initial_cluster(self.center.keys())
		self.cost = []
		#开始迭代
		for _ in range(self.max_iter):

			cluster_dict = {}
			self.cluster = self.initial_cluster(self.center.keys())
			length = len(X)
			for j in range(length):
				distance = {}
				for key,value in self.center.items():
					distance[key] = self.calc_distance(value,X[j,:])

				#选出距离最近的簇，并将该点加入到该簇中
				label = min(distance.keys(),key=lambda x:distance[x])
				self.cluster[label].append(j)

			self.update_centers()

			self.cost.append(self.calc_cost(X))

		return self


	def calc_cost(self,X):
		cost = 0

		for key,value in self.center.items():
			for data in X[self.cluster[key]]:
				cost += self.calc_distance(value,data)

		return cost

	#计算两点之间的距离
	def calc_distance(self,spot1,spot2):
		spot1 = np.array(spot1)
		spot2 = np.array(spot2)

		return np.sum((spot1 - spot2)**2)

	def initial_center(self,X):
		center_dict = {}

		if self.init == "random":
			n,m = X.shape
			#特征边界
			scale_list=[]
			for i in range(m):
				min_ = X[:,i].min()
				max_ = X[:,i].max()
				scale_list.append([min_,max_])

			for i in range(self.n_clusters):
				center_name = "k" + str(i)

				#中心点坐标
				center_coordinates = []
				for coord in scale_list:
					center_coordinates.append(np.random.uniform(coord[0],coord[1]))

				center_dict[center_name] = center_coordinates

		elif self.init == "k++":
			'''
			方法：将所有点中距第一个点最远的点作为第一个中心点，然后再找出据第一个中心点最远的点作为第二个中心点，
			再选距第一个第二个中心点距离累加最远的点作为第三个中心点·····
			'''
			length = len(X)

			#先寻找第一个点
			max_distance = 0
			for i in range(length):
				distance = self.calc_distance(X[0,:],X[i,:])
				if distance > max_distance:
					max_distance = distance
					center_dict["k0"] = X[i,:]

			#根据第1个点寻找后面的初始化中心点				
			while len(center_dict) < self.n_clusters:
				max_distace_sum = 0
				name = "k" + str(len(center_dict))

				for i in range(length):
					distance_sum = 0

					for key,value in center_dict.items():
						distance_sum += self.calc_distance(X[i,:],value)

					if distance_sum > max_distace_sum:
						max_distace_sum = distance_sum
						center_dict[name] = X[i,:]

		else:
			raise NameError('no this init style')

		return center_dict

	def initial_cluster(self,center_name_list):
		cluster_dict = {}

		for name in center_name_list:
			cluster_dict[name] = []

		return cluster_dict

	def update_centers(self):

		for key in self.center.keys():
			self.center[key] = X[self.cluster[key],:].mean(axis=0)



if __name__ == '__main__':
	from sklearn.datasets import make_blobs 
	import matplotlib.pyplot as plt

	X,y = make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.8,shuffle=True,random_state=0)

	km = Kmean(n_clusters=3,max_iter=10,init="k++")
	km.fit(X)

	clusters = km.cluster
	centers = km.center
	print(km.cost)

	for key,value in clusters.items():
		#簇内的点
		plt.scatter((X[value])[:,0],(X[value])[:,1],label=key)
	for key,value in centers.items():
		plt.scatter(value[0],value[1],c="red",marker="*",s=150)
	plt.legend()
	plt.show()




