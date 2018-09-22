#-*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler    #引入归一化的类

def linearRegression():
    print("加载数据...\n")
    data = loadtxtAndcsv_data("data_1.txt",",",np.float64)  #读取数据
    X = np.array(data[:,0:-1])      # X对应0到倒数第2列                  
    y = np.array(data[:,-1])        # y对应最后一列  
        
    # 归一化操作
    scaler = StandardScaler()   
    scaler.fit(X)
    x_train = scaler.transform(X)
    
    # 线性模型拟合
    model = linear_model.LinearRegression()
    model.fit(x_train, y)
    
    #预测结果
    print("The model is :",str(model.coef_[0]),"* x +",str(model.intercept_))
    print(model.coef_)       # Coefficient of the features 决策函数中的特征系数
    print(model.intercept_)  # 又名bias偏置,若设置为False，则为0

# 加载txt和csv文件
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

# 加载npy文件
def loadnpy_data(fileName):
    return np.load(fileName)

if __name__ == "__main__":
    linearRegression()
