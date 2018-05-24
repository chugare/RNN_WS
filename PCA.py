# -*- coding: utf-8 -*-
#   Project name : RNN_WS
#   Edit with PyCharm
#   Create by Dell at 2018/5/22 16:28
#   南京大学软件学院 Nanjing University Software Institute
#   对向量类型的数据进行主成分分析
import numpy as np

def zeroMean(dataMat):
    #input :a numpy matrix with shape of (data_count,num_unit)
    dataMat = np.mat(dataMat,dtype=np.float32)
    mean = np.mean(dataMat,axis=0)
    return dataMat-mean,mean

def PCA(dataMat,n):
    newdata,meanVal = zeroMean(dataMat)
    covMat = np.cov(newdata,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat,dtype=np.float32))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n+1):-1]
    n_eigVects = eigVects[:,n_eigValIndice]
    lowDDataMat = newdata*n_eigVects
    reconMat = (lowDDataMat*n_eigVects.T)+meanVal
    return lowDDataMat,reconMat

