 #coding=utf-8
from math import log2
import numpy as np
import pandas as pd

#计算样本熵,定最后一列为样本类别
def calcEntropy(data):
	labels_dict = {}
	sample_size = data.shape[0]
	sample_entropy = 0

	for sample in data:
		label = sample[-1]

		if label not in labels_dict:
			labels_dict[label] = 0
		labels_dict[label] += 1

	for v in labels_dict.values():
		prob = v/sample_size
		sample_entropy -= prob * log2(prob)

	return sample_entropy

#按照节点信息生成新的样本数据
'''
选择指定特征的值为value的数据作为新的样本数据，并删除指定的特征
'''
def generateNodeData(data,idx,value):
	new_data = data[data[:,idx]==value]
	new_data = np.delete(new_data,idx,axis=1)#删除indx列代表的特征
	return new_data


#选择最合适的特征进行分裂
def chooseBestFeatureToSplit(data):
	feature_num = data.shape[1] - 1
	base_enctropy = calcEntropy(data)#计算原始样本熵
	infogain_list = {}

	for fea in range(feature_num):
		feature_value = np.unique(fea)#当前特征所有的特征取值
		new_entropy = 0.0
		for val in feature_value:
			temp_data = generateNodeData(data,fea,value):
			new_entropy += calcEntropy(temp_data)
		infogain = base_enctropy - new_entropy
		infogain_list[fea] = infogain

	best_feature = max(infogain_list,key=lambda x:infogain_list[x])
	return best_feature

#所有特征用完后，采用多数表决计算节点分类
def majorityCnt(class_list):
	return max(class_list,key=class_list.count)

#创建决策树
def createTree(data):#data是DataFrame类型
	lables = list(data.columns)#特征的标签
	sample_num,feature_num = data.shape
	data = data.values	#将数据转换为array类型
	class_list = list(data[:,-1])
	if len(set(class_list)) == 1:	#只有一个类别，停止分裂
		return class_list[0]
	if feature_num == 1:		#所有特征已经用完，返回类别个数最多的类别
		return majorityCnt(class_list)





def	createTree1(data, labels):
	sample_num,feature_num = data.shape
	#类别列表
	class_list = [example[-1] for example in data]
	if len(set(class_list)) == 1:  # 只有一个类别则停止划分
		return class_list[0]
	if feature_num == 1:# 所有特征已经用完，返回最多个数的类别
		return majorityCnt(class_list)
	#寻找最优分裂特征
	bestFeat = chooseBestFeatureToSplit(data)
	#分裂特征名称
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel: {}}
	#分裂完成后删除此特征
	del (labels[bestFeat])
	#最优分裂特征的特征值
	featValues = [example[bestFeat] for example in data]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]  # 为了不改变原始列表的内容复制了一下
		myTree[bestFeatLabel][value] = createTree(generateNodeData(data,bestFeat,value),subLabels)
	return myTree

def main():
	pass
if __name__ == '__main__':
	main()


