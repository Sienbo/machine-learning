#-*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

def linearRegression(alpha=0.1,num_iters=400):
    print("加载数据\n···\n···")
    
    data = loadtxtAndcsv_data("data.txt",",",np.float64)  #读取数据
    X = data[:,:-1]      # X对应0到倒数第2列                  
    y = data[:,-1]        # y对应最后一列  
    m = data.shape[0]         # 总的数据条数
    col = data.shape[1]      # data的列数
    X = featureNormaliza(X)    # 归一化
    
    print("执行梯度下降算法....\n")
    
    theta = np.zeros(col)
    theta,J_history = gradientDescent(X, y, theta, alpha, num_iters)
    plot_fit(X,y,theta)         # 画图看一下效果
    plotJ(J_history, num_iters)
    
# 加载txt和csv文件
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

# 加载npy文件
def loadnpy_data(fileName):
    return np.load(fileName)

# 归一化feature
def featureNormaliza(X):
    mu = np.mean(X,0)          # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X,0)        # 求每一列的标准差
    for i in range(X.shape[1]):     # 遍历列
       X[:,i] = (X[:,i]-mu[i])/sigma[i]  # 归一化
    return X

# 画二维图
def plot_fit(x,y,theta):
    plt.scatter(x,y,color='r',s=10)
    x1 = np.arange(-2,2,0.01)
    y1 = np.dot(x1,theta[1]) + theta[0]
    plt.plot(x1,y1,color='b')
    plt.show()
    print('linear_regression is: ',str(theta[0]),' + ',str(theta[1]),'* x')


# 梯度下降算法
def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)      
    n = len(theta)
    J_history = []
    
    for i in range(num_iters):  # 遍历迭代次数    
        y_predict = np.dot(X,theta[1:]) + theta[0]     # 计算预测输出
        errors = y_predict - y 						   # 预测值和实际值之间的差
        cost_value = np.dot(errors,errors)/(2*m)	   # 代价值
        theta[1:] -= alpha * np.dot(X.T,errors)/m 				   #梯度的计算
        theta[0] -= alpha * np.sum(errors)/m
        J_history.append(cost_value)      #添加代价值
    #print(J_history)
    return theta,J_history  

# 画每次迭代代价的变化图
def plotJ(J_history,num_iters):
    x = np.arange(1,num_iters+1)
    plt.plot(x,J_history)
    plt.xlabel(r"n_iers") # 注意指定字体，要不然出现乱码问题
    plt.ylabel(r"cost_value")
    plt.title(r"linearRegression",)
    plt.show()

# 测试linearRegression函数
def testLinearRegression():
    linearRegression(0.5,1000)
    
if __name__ == "__main__":
    testLinearRegression()