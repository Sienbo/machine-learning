#coding=utf-8
#@filename  = 'decisionTreeOfId3'
#@author    = 'si'
#@time      = '20180908'
#modified 2 2019-08-27

import numpy as np
import pandas as pd

class DecisionTreeID3(object):
	'''
	-简介：
	决策树类（基于ID3算法）
	'''
	def __init__(self,varepsilon):
		'''
		输入：
			varepsilon：float
				信息增益的阈值，当最大的信息增益小于此值后，不再进行分裂
		输出：
			无
		'''
		self.varepsilon = varepsilon

	#计算样本的香农熵(经验熵)
	def calcEntropy(self,data):
		'''
		输入：
			data: array
				样本数据，data数据中的最后一列代表样本的类别
		返回: 
			float
				样本的香农熵
		'''
		sample_size = data.shape[0]
		label_list = list(data[:,-1])

		entropy = 0.0
		for label in set(label_list):
			prob = float(label_list.count(label) / sample_size)
			entropy -= prob * np.log2(prob)

		return entropy

	#计算条件熵
	def calcConditionalEntropy(self,data,row):
		'''
		输入：
		data: array
				样本数据，data数据中的最后一列代表样本的类别
			row:int 
				选中的特征所处的列号
		返回：
			float
				条件熵的数值
		'''
		length = data.shape[0]
		feature_list = list(data[:,row])

		conditional_entropy = 0.0
		for feature in set(feature_list):
			temp_data = data[data[:,row] == feature]
			prob = len(temp_data)/length

			conditional_entropy += prob*self.calcEntropy(temp_data)

		return conditional_entropy

	#选取最优的样本进行分裂
	def chooseBestFeatureToSplit(self,data):
		'''
		输入：
			data: array
				样本数据，data数据中的最后一列代表样本的类别
		返回: 
			int
				最优的分裂特征的列号
			float
				最大的信息增益
		'''
		#特征个数
		feature_size = data.shape[1] - 1
		#样本总量
		sample_size = data.shape[0]

		#样本经验熵
		base_entropy = self.calcEntropy(data)

		info_gain = []
		for i in range(feature_size):
			#计算条件熵
			conditional_entropy = self.calcConditionalEntropy(data,i)
			info_gain.append(base_entropy - conditional_entropy)

		return info_gain.index(max(info_gain)),max(info_gain)

	#根据特征指定值生成节点新样本
	def generateNodeData(self,data,label,value):
		'''
		输入：
			data: Dataframe
				样本数据，data数据中的最后一列代表样本的类别
			label:str
				样本的标签
			value:float/int
				特征值
		返回: 
			Dataframe
				新的根据特征值划分的样本集
		'''
		#所有该特征值为value的数据
		new_data = data[data[label]==value]
		#丢弃该特征
		new_data = new_data.drop(columns=label)

		return new_data

	#生成ID3决策树模型
	def fit(self,df):
		'''
		输入：
			df: Dataframe
			样本数据，data数据中的最后一列代表样本的类别
			varepsilon：float
			信息增益的阈值，当最大的信息增益小于此值后，不再进行分裂
		返回: 
			class
			类对象
		'''
		self.labels = list(df.columns)[:-1]
		self.dc_tree = self.createTree(df)
		return self

	#使用多数表决法计算节点类标
	def majorityLabelCount(self,data):
		'''
		输入：
			df: array
				样本数据，data数据中的最后一列代表样本的类别
		返回：
			int
				类标
		'''
		return max(list(data),key=list(data).count)

	#创建树
	def createTree(self,df):
		'''
		 输入：
		    data: DataFrame
		        样本数据，data数据中的最后一列代表样本的类别
		返回: 
		    dict
		        决策树
		'''
		#特征列表
		feature_list = list(df.columns)[:-1]

		#将Dataframe转换为array类型
		data_value = df.values

		#类标列表
		label_list = list(data_value[:,-1])

		#1.所有数据类别时直接返回此类别作为叶子节点
		if len(set(label_list)) == 1:
			return label_list[0]

		#2.没有特征可使用，返回数据中类标个数最多的那个类标
		if len(data_value[0]) == 1:
			return self.majorityLabelCount(data_value)

		#3.选取信息增益比最大的特征
		best_fea_num,best_info_gain = self.chooseBestFeatureToSplit(data_value)

		#4.如果信息增益比小于阈值，则形成叶子节点，并返回类标中数量最多的那个类标
		if best_info_gain < self.varepsilon:
			return self.majorityLabelCount(data_value[:,-1])

		#5.否则，对选中特征中所有取值进行划分
		best_fea_label = feature_list[best_fea_num]
		dc_tree = {best_fea_label:{}}


		for val in np.unique(data_value[:,best_fea_num]):
			#根据特征的不同值生成新的子集
			subset = self.generateNodeData(df,best_fea_label,val)
			dc_tree[best_fea_label][val] = self.createTree(subset)

		return dc_tree

	#预测数据类别
	def predict(self,sample):
		'''
		输入：
			sample:list
				输入的数据
		输出：
			str
				所属的类别
		'''
		tree = self.dc_tree
		labels = self.labels
		sample_dict = {labels[i]:sample[i] for i in range(len(sample))}
		return self.predict_class(tree,sample_dict)

	def predict_class(self,tree,sample_dict):
		if type(tree)  == dict:
			key = list(tree.keys())[0]
			tree = tree[ key ][ sample_dict[ key ] ]
			del (sample_dict[ key ])
			return self.predict_class(tree, sample_dict)
		else:
			return tree

def main():
	test_data = pd.DataFrame([
        [1,1,1,'YES'],
        [1,1,2,'YES'],
        [1,2,1,'YES'],
        [2,1,1,'YES'],
        [1,2,2,'NO'],
        [2,1,2,'NO'],
        [2,2,1,'NO'],
        [2,2,2,'NO']
    ],columns=['X','Y','Z','label'])
	my_tree = DecisionTreeID3(varepsilon=0.00)
	my_tree.fit(test_data)
	print(my_tree.dc_tree)
	test_data = [2,1,2]
	print(my_tree.predict(test_data))
if __name__ == '__main__':
	main()