# -*- coding: utf-8 -*-
#   Project name : RNN_WS
#
#   Edit with PyCharm
#
#   Create by simengzhao at 2018/5/14 下午4:21
#
#   南京大学软件学院 Nanjing University Software Institute
#
import one_hot
import numpy as np
from Constant import *
def sentence_gen(fname,isNum = False):
    Onehot = one_hot.OneHotEncoder()
    df = open(fname,'r',encoding='utf-8')
    for line in df:
        res = Onehot.one_hot_single(line,isNum)
        res2 = res[1:]
        if isNum:
            eod = 0
        else:
            eod = np.zeros(VEC_SIZE,np.int32)
            eod[0] = 1
        res2 = np.insert(res2,len(res2),values=eod,axis=0)
        yield res,res2
        #print(res2)