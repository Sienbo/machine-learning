#coding=utf-8
#@filename  = 'decisionTreeOfId3'
#@author    = 'sienbo'
#@time      = '20190908'

import numpy as np
from math import log2
import pandas as pd

class DecisionTreeID3(object):
    '''
    -简介：
        决策树类（基于ID3算法）

    '''
    def __init__(self):
        pass

    #计算样本的香农熵
    def calc_entropy(self,data):
        '''
        :param data: 样本数据（样本类别的array类型）
        :return: 样本的香农熵
        '''
        sample_size = data.shape[0]
        label_list = list(data[:,-1])
        entropy = 0.0
        for label in set(label_list):
            prob = float(label_list.count(label) / sample_size)
            entropy -= prob * log2(prob)

        return entropy

    #多数表决发计算节点类标
    def majorityLabelCount(self,data):
        '''
        :param data:样本数据，一般为类别的数组
        :return: 使用多数表决发后的出的类标结果
        '''
        return max(list(data),key=list(data).count)

    #选取最优的样本进行分裂
    def chooseBestFeatureToSplit(self,data):
        '''
        :param data:样本数据集（array类型）
        :return: 最优的分裂特征
        '''
        feature_size = data.shape[1] - 1
        sample_size = data.shape[0]
        base_entropy = self.calc_entropy(data)
        info_gain = []

        for i in range(feature_size):
            feature_value_set = set(data[:,i])
            temp_entropy = 0.0
            for value in feature_value_set:
                temp_data = data[data[:,i]==value]
                prob = float(temp_data.shape[0] / sample_size)
                temp_entropy += prob * self.calc_entropy(temp_data)
            info_gain.append(base_entropy - temp_entropy)

        return info_gain.index(max(info_gain))

    #根据特征指定值生成节点新样本
    def generateNodeData(self,data,label,value):
        new_data = data[data[label]==value]
        new_data = new_data.drop(columns=label)
        return new_data

    #生成ID3决策树模型
    def fit(self,data):
        '''
        :param data:
            -data:原始样本集合(DataFrame类型)
        :return: 决策树模型（使用多层字典储存）
        '''
        self.labels = list(data.columns[:-1])
        self.dc_tree = self.createTree(data)
        return self

    def createTree(self,data):
        columns = list(data.columns)[:-1]
        data_ar = data.values
        class_list = list(data_ar[:,-1])
        if len(set(class_list)) == 1:
            return class_list[0]
        if data_ar.shape[0] == 1:
            return self.majorityLabelCount(data_ar)

        best_fea = self.chooseBestFeatureToSplit(data_ar)
        best_fea_label = columns[best_fea]
        dc_tree = {best_fea_label:{}}
        #删除此特征
        del(columns[best_fea])
        fea_value_set = set(data_ar[:,best_fea])

        for val in fea_value_set:
            #生成节点数据
            node_data = self.generateNodeData(data,best_fea_label,val)
            dc_tree[best_fea_label][val] = self.createTree(node_data)
        return dc_tree

    #预测数据类别
    def predict(self,sample):
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
    ],columns=['X','Y','Z','class'])
    my_tree = DecisionTreeID3()
    my_tree.fit(test_data)
    print(my_tree.dc_tree)
    test_data = [2,1,2]
    print(my_tree.predict(test_data))
if __name__ == '__main__':
    main()