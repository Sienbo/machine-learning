import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from perceptron import Perceptron

#二位数据集边界的可视化
def plot_decision_region(X,y,classifier,resolution=0.01):
	markers = ('s','x','o','^','v')
	colors = ('red','blue','lightgreen','gray','cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	x1_min,x1_max = X[:,0].min() - 1,X[:,0].max() + 1
	x2_min,x2_max = X[:,1].min() - 1,X[:,1].max() + 1

	xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
						np.arange(x2_min,x2_max,resolution))
	z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
	z = z.reshape(xx1.shape)
	plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)

	for idx,c1 in enumerate(np.unique(y)):
		plt.scatter(x=X[y == c1,0],y=X[y == c1,1],
			alpha=0.8,c=cmap(idx),
			marker=markers[idx],label=c1)
	plt.xlabel('sepal length')
	plt.ylabel('petal length')
	plt.legend(loc='upper left')
	plt.show()

#读取训练数据
filename = r"irisdata.csv"
df = pd.read_csv(filename)
X_data,y_data = df.iloc[:100,[0,2]].values,df.iloc[:100,-1].values
y_data = np.where(y_data == 'Iris-setosa',-1,1)

#画出数据分布
plt.scatter(X_data[:50,0],X_data[:50,1],
	color='red',marker='x',label='setosa')
plt.scatter(X_data[50:100,0],X_data[50:100,1],
	color='blue',marker='o',label='versioicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#数据训练过程中错误分类的个数
ppn = Perceptron()
ppn.fit(X_data,y_data)
plt.plot(ppn.errors_,
	marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassification')
plt.show()

#决策边界可视化
plot_decision_region(X_data,y_data,classifier=ppn)

