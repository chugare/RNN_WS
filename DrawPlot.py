# -*- coding: utf-8 -*-
#   Project name : RNN_WS
#   Edit with PyCharm
#   Create by Dell at 2018/5/22 21:00
#   南京大学软件学院 Nanjing University Software Institute
#
import matplotlib.pyplot as plt
import numpy as np
import datetime
def drawScatter(data,label):

    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.scatter(data[:,0],data[:,1],10,label)

    plt.savefig('figure_on%s.png'%datetime.datetime.now().date())