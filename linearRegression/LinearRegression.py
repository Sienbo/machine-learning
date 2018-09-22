#-*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

class LinearRegression():
    '''
    这是一个线性回归类
    优化算法使用梯度下降法
    使用实例：
        lr = LinearRegression() #实例化类
        lr.fit(X_train,y_train) #训练模型
        y_predict = lr.predict(X_test) #预测训练数据
        lr.plotFigure()用于画出样本散点图与预测模型（仅限于样本特征为1时）
    '''
    def __init__(self,alpha=0.02,n_iter=1000):
        self._alpha = alpha     #步长
        self._n_iter = n_iter    #最大迭代次数

    #初始化模型参数
    def initialPara(self,n):
        # 只有一个特征时
        if n == 1:
            return 0,0
        else:
            return np.zeros(n),0

    #训练模型
    def fit(self,X_train,y_train):
        #保存原始数据
        self.X_source = X_train.copy()
        self.y_source = y_train.copy()

        # 初始化w，w0
        sample_num = X_train.shape[0]
        if len(X_train.shape) == 1:
            self._w, self._w0 = self.initialPara(1)
        else:
            self._w,self._w0 = self.initialPara(X_train.shape[1])

        #创建列表存放每次每次迭代后的偏差值
        self.cost = []

        #开始训练迭代
        for _ in range(self._n_iter):
            y_predict = self.predict(X_train)
            y_bias = y_train - y_predict
            self.cost.append(np.dot(y_bias,y_bias)/(2 * sample_num))
            self._w += self._alpha * np.dot(X_train.T,y_bias)/sample_num
            self._w0 += self._alpha * np.sum(y_bias)/sample_num

    def predict(self,X_test):
        if len(X_test.shape) == 1:
            return self._w * X_test + self._w0
        else:
            return X_test.dot(self._w) + self._w0

    #画出样本散点图以及使用模型预测的线条
    def plotFigure(self):
        #画出样本散点图
        plt.scatter(self.X_source,self.y_source,c='r',label="samples",linewidths=0.4)

        #画模型图
        x1_min = self.X_source.min()
        x1_max = self.X_source.max()
        X_predict = np.arange(x1_min,x1_max,step=0.01)
        plt.legend(loc='upper left')
        #画出预测图形
        plt.plot(X_predict,self._w*X_predict+self._w0)
        plt.show()


if __name__ == "__main__":
    import pandas as pd

    #读取样本文件数据
    df = pd.read_csv("data.txt")
    X_train = df.iloc[:,0].values
    y_train = df.iloc[:,1].values

    #使用线性回归类生成模型
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    #打印出参数
    print(lr._w,lr._w0)
    #画出损失值随迭代次数的变化图
    plt.plot(lr.cost)
    plt.show()
    #画出样本散点图以及模型的预测图
    lr.plotFigure()

    #预测x
    x = np.array([3])
    print("The input x is{0},then the predict of y is:{1}".format(x,lr.predict(x)))
