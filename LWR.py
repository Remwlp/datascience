# 局部加权线性回归
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    # numFeat = len(open(filename).readline().split())
    dataSet = []
    Labels = []
    for line in open(filename).readlines():
        dataSet.append([float(1), float(line.split(',')[0])])
        Labels.append(float(line.split(',')[1]))
    return dataSet, Labels


# 得到某个点的预测值
def lwlr(testpoint, dataSet, Labels, k):
    xMat = np.mat(dataSet)
    yMat = np.mat(Labels)
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testpoint - xMat[j, :]
        # 取矩阵中某一特定的值应该写在一个[]里，不能分开写
        #   a = np.array([[1,2],[2,3]])
        # print(a[0][1])
        # 2
        # a = np.mat([[1,2],[2,3]])
        # print(a[0,1])
        # 2

        weights[j, j] = np.exp(diffMat * diffMat.T / (-2 * (k ** 2)))
    xTx = xMat.T * weights * xMat
    if np.linalg.det(xTx) == 0.0:
        print("行列式为0，不能求逆")
        return
    ws = xTx.I * xMat.T * weights * yMat.T
    return testpoint * ws


# 得到所有的预测值
def lwlrTest(textArr, xArr, yArr, k=0.01):
    m = np.shape(textArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(xArr[i], xArr, yArr, k)
    return yHat


def draw():
    dataSet, Labels = loadDataSet("sample.txt")
    plt.scatter([x[1] for x in dataSet], Labels)
    # 由于得到的结果是曲线，因此要对x轴进行排序
    xIndex = np.array([x[1] for x in dataSet]).argsort()  # 返回x排序后的下标
    xSort = np.array([x[1] for x in dataSet])[xIndex]
    ySort = lwlrTest(dataSet, dataSet, Labels,0.01)[xIndex]
    kxHat=[]
    kyHat=[]
    num=0
    for i in range(np.shape(dataSet)[0]-1):
        if pow((pow(xSort[i+1]-xSort[i-1],2)+pow(ySort[i+1]-ySort[i-1],2)),0.5)>0.008:
            kxHat.append(xSort[i])
            kyHat.append(ySort[i])
        elif num==6:
            kxHat.append(xSort[i])
            kyHat.append(ySort[i])
            num=0
        else:
            num+=1
    plt.plot(np.array(kxHat), np.array(kyHat), 'r')
    plt.xlim(114.2, 114.65)
    plt.ylim(30.45, 30.7)
    plt.legend()
    plt.show()
    file = open('sp.txt','w')
    for i in range(len(kyHat)):
        file.write(str(kxHat[i])+','+str(kyHat[i])+'\n')
    file.close()
    return kxHat,kyHat




dataSet, Labels = loadDataSet("sample.txt")
kx, ky=draw()
print(len(kx))
