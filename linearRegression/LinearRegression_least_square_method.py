#-*- coding: utf-8 -*-
'''优化方法使用最小二乘法,实现一个线性回归过程'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题

def linearRegression(alpha=0.1,num_iters=400):
    print("加载数据\n···\n···")
    
    data = loadtxtAndcsv_data("data.txt",",",np.float64)  #读取数据
    X = data[:,0:-1]      # X对应0到倒数第2列                  
    Y = data[:,-1]        # y对应最后一列  
    m = data.shape[0]         # 总的数据条数
    col = data.shape[1]      # data的列数
    X1 = np.column_stack((np.ones(m),X))
    theta = ((np.linalg.inv(X1.T.dot(X1))).dot(X1.T)).dot(Y)
    
    plot_fit(X,Y,theta)         # 画图看一下效果
    
# 加载txt和csv文件
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

# 加载npy文件
def loadnpy_data(fileName):
    return np.load(fileName)

# 画二维图
def plot_fit(x,y,theta):
    plt.scatter(x,y,c='r',s=15)
    x1 = np.arange(0,10,0.01)
    y1 = np.dot(x1,theta[1]) + theta[0]
    plt.plot(x1,y1,c='b')
    plt.show()
    print('linear_regression is: ',str(theta[0]),' + ',str(theta[1]),'* x')


# 测试linearRegression函数
def testLinearRegression():
    linearRegression(0.5,1000)
   
if __name__ == "__main__":
    testLinearRegression()
