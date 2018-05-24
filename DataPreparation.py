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
import json
import re
import codecs
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
class TupleGenerator:
    def __init__(self):
        self.lawDic = {}
        self.con_list = []
        self.law_list = []
    def write_lawDic(self):
        with codecs.open('lawDic.json','w',encoding='uf-8') as jfile:
            json.dump(self.lawDic,jfile)
    def read_lawDic(self):
        with codecs.open('lawDic.json','r',encoding='uf-8') as jfile:
            self.lawDic=json.load(jfile)
    def tuple_gen(self,fname):
        s_file = open(fname, 'r', encoding='utf-8')
        count = 0
        for line in s_file:
            laws = re.findall('\[(.*?)\]',line)[0]
            content = line.split(']')[-1].strip()
            self.con_list.append(content)
            laws = re.findall('\'(.+?)\'',laws)
            self.law_list.append(laws)
            for law in laws:
                if  law not in self.lawDic:
                    self.lawDic[law] = []
                self.lawDic[law].append(count)
                count +=1

                yield content
    def generateLabel(self,mat):
        mat = np.array(mat)
        sorted_laws = sorted(self.lawDic,key= lambda x:len(x),reverse=True)
        self.color = {}
        color_ind = 1
        point_list = []
        label_list = []
        for l in sorted(sorted_laws):
            self.color[l] = color_ind
            for art in self.lawDic[l]:
                tmp_p = mat[art]
                point_list.append(tmp_p)
                label_list.append(color_ind)
            color_ind += 1
        return np.array(point_list),np.array(label_list)
