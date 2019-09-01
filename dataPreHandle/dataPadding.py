import pandas as pd
from sklearn.preprocessing import Imputer

#缺失数据的查看
def lossDataCheck(df):
	'''
	输入：
		df:Dataframe
	输出：
		打印每列数据缺失的情况
	'''
	result = df.isnull().sum()
	print(result)

#缺失数据的丢弃
def lossDataDrop(df,axis=0,how='any',thresh=None,subset=None,inplace=False):
	'''
	输入：
		df:Dataframe
		axis:int(default 0)
			选择的丢弃方式，0为丢弃该行，1为丢弃该列
		how:str
			为"all"时，仅当一行/列全部为空时才丢弃
		thresh：int
			激活数据丢弃行为的阈值
		subset：list
			列名的列表，指定subset后，仅对subset中存在的列做数据缺失判断，其他列不进行判断
	输出：
		DataFrame
	'''
	return df.dropna(axis=axis)

#数据补齐
def lossDataImpute(df):
	'''

	'''	
	imr = Imputer(missing_values="NaN",strategy="mean",axis=0)
	imr = imr.fit(df)
	imputed_data = imr.transform(df.values)
	print(imputed_data)


if __name__ == '__main__':
	df = pd.DataFrame([\
		[1,2,3,4],\
		[5,6,None,8],\
		[9,None,11,None,],\
		[13,14,15,None,]\
		],\
		columns=["A","B","C","D"])
	lossDataImpute(df)

import numpy as np

class DataPreHandl():
	'''
	数据预处理类
	'''
	def __init__(self):
		pass

	#有序特征编码
	def ordinalEncode(self,df,column,mapping):
		'''
		输入：
			df:DataFrame
				需要处理的数据集
			column:str
				需要编码的列名
			mapping:dict
				编码表
		输出：
			编码后的数据集
		'''
		df[column] = df[column].map(mapping)

		return df

	#类标特征的编码
	def labelEncode(self,df,class_label):
		'''
		输入：
			df:DataFrame
				需要处理的数据集
			class_label:str
				类标的列名
		输出：
			编码后的数据集
		'''
		label_mapping = {label:idx for idx,label in enumerate(np.unique(df[class_label]))}

		df[class_label] = df[class_label].map(label_mapping)
		
		return df




