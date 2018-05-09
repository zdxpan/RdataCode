#!/usr/bin/env python
# -*- coding: utf-8 -*-
#code:myhaspl@qq.com
#8-1.py

"""
Rosenblatt 感知器
1) 将数据输入神经元(这里是一个线性模型)或者将其转换为线性模型  

2) 通过权值 W 和  输入 共同计算局部诱导域   V =   sum [ w[i]x[i] ]  + b

3) 调整权值W 找到能分类的超平面 ： sum [ w·[i]x[i] ]  + b = 0

单样本修正算法：   
        神经网络每次读入一个样本，进行修正
批量样本修正法：
        使用代价函数来进行分类误差率的控制。批量样本修正算法对样本进行多次读取，直到神经
        网络误差率降到合适的程度才停止样本训练。其误差率使用嘴直观的错分类样本数量准则
        
    核心：
        权值更新策略：
        
        

"""
import numpy as np
b=0 #偏置
a=0.5  #学习速率，调整更新的步伐
x = np.array([[b,1,1],[b,1,0],[b,0,0],[b,0,1]])#输入向量，
d =np.array([1,1,0,1])#期望输出 == len(x)

w=np.array([b,0,0])#权值向量---

def sgn(v):#硬限幅函数（输入值）
        if v>0:
                return 1
        else:
                return 0
def comy(myw,myx):
        return sgn(np.dot(myw.T,myx))

a=='''_____________这是训练的核心算法部分___________________________ '''
def neww(oldw,myd,myx,a):#权值的更新过程 ------
        return oldw+a*(myd-comy(oldw,myx))*myx
i=0
for xn in x:
        w=neww(w,d[i],xn,a)
        i+=1

        
for xn in x:
        print ("%d or %d => %d "%(xn[1],xn[2],comy(w,xn)))
        
        
        
ax___________________________________________________________= 0

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#8-2.py
'''
2x+1 = y  第一类
7x+1 = y  第二类
单样本训练法
'''
import numpy as np
b=1
a=0.3
x=np.array([[b,1,3],[b,2,5],[b,1,8],[b,2,15],[b,3,7],[b,4,29]])
d=np.array([1,1,-1,-1,1,-1])
w=np.array([b,0,0])#初始全职
def sgn(v):
    if v>=0:
        return 1 
    else :
        return -1
        
def comy(myw,myx):
        return sgn(np.dot(myw.T,myx))
def neww(oldw,myd,myx,a):
        return oldw+a*(myd-comy(oldw,myx))*myx
i=0
for xn in x:
        w=neww(w,d[i],xn,a)
        i+=1

   
test=np.array([b,9,19])
print( "%d ~ %d => %d "%(test[1],test[2],comy(w,test)))
test=np.array([b,9,64])
print( "%d ~ %d => %d "%(test[1],test[2],comy(w,test)))
 #w[:1. ,  1.2, -0.6]
得到神经网络的分类方程=' 1.2 x - 0.6 y + 1 =0  => y = 2x+1.68'

''' 批量样本修正的神经网络算法  代价函数 用到了 梯度更新权值  ————————————————————————————— '''
"""  w(k+1) = w(k) - n(k)`J(w(k))
·J(w(k)) = sum[-y]  y is incorrect 
检查误差率： 或者训练次数

1，初始权值，学习率，期望误差率
2，读取所有样本
3，依次对样本进行训练，更新权值
4，检查 误差是否小于指定值
"""
import numpy as np
b = 1  #偏置
a = 0.5 #
x = np.array([[1,1,3],[1,2,3],[1,1,8],[1,2,15]])
d = np.array([1,1,-1,-1])#样本所属类别
w = np.array([b,0,0])
wucha = 0
ddcount = 50
def sgn(v):v>0 and 1 or -1#--------------------------
def sgn(v):
    if v>0:
        return 1
    else:
        return -1

def comy(myw,myx):
    return sgn(np.dot(myw.T,myx))
def tiduxz(myw,myx,mya):
    i = 0
    sum_x = np.array([0,0,0])
    for xn in myx:
        if comy(myw,xn)!=d[i]:
            sum_x += d[i]*xn            
        i+=1
    return mya*sum_x

i=0
while True:
    tdxz = tiduxz(w,x,a)
    print(w)
    w= w + tdxz
    i = i+1
    if abs(tdxz.sum())<= wucha or i>=ddcount:break
test = np.array([b,9,19])
print(comy(w,test))
test = np.array([b,9,64])



'''
_________________ = 0#'LMS 最小均方根算法'
MSE = mean((observed - predicted)**2)
e(n) = d(n) - X.T(n)W(n)im
梯度向量 = -X(x)e(n)
W(n+1) = W(n) + nX(n)e(n)

使用LMS实现逻辑或的运算，使用神经网络推导


使用固态退火来改变一层不变的学习率，让学习率随着时间变化


'''
b = 1 
a = 0.1 #学习率

a = 0.0
a0 = 0.1
r = 5.0
mycount = 0


x = np.array([[1,1,1],[1,1,0],[1,0,1],[1,0,0]])
d = np.array([1,1,1,0])#是与不是标记为1或者0
def sgn(v):
    if v>0:
        return 1
    else:
        return 0
#----------------------------

    #学习n% ==6  为第一类，若结果为3 输出为一类，输出-1
    x = np.array([[1,1,6],[1,2,12],[1,3,9],[1,8,24]])
    d = np.array([1,1,-1,-1])#是与不是标记为1或者-1
    def sgn(v):
        if v>0:
            return 1
        else:
            return -1
#------------------------------------------
w = np.array([b,0,0])
expect_e=0.005
maxtrycount=20


#
        
def get_v(myw,myx):
        return sgn(np.dot(myw.T,myx))
def get_e(myw,myx,myd):
        return myd-get_v(myw,myx)    
def neww(oldw,myd,myx,a):

        mye=get_e(oldw,myx,myd)
        return (oldw+a*mye*myx,mye)
'''   _____________________________________使用固体退货原理的改变学习率的权值更新算法'''
                def neww(oldw,myd,myx,a):
                        mye=get_e(oldw,myx,myd)
                        a = a0/(1+float(mycount)/r)
                        return (oldw+a*mye*myx,mye)
'''---____________-------------------------______________________------'''
#迭代次数或者迭代目标函数
mycount=0
while True:
        mye=0#用于累加误差
        i=0          
        for xn in x:
                w,e=neww(w,d[i],xn,a)
                i+=1
               # x = '#误差平方的累加'
                mye+=pow(e,2)  
        mye/=float(i)
       # print(mye)
        mycount+=1
        print (u"第 %d 次调整后的权值："%mycount)
        print (w)
        print (u"误差：%f"%mye )      
        if mye<expect_e or mycount>maxtrycount:break 
               
for xn in x:
        print ("%d or %d => %d "%(xn[1],xn[2],get_v(w,xn)))

#验证被整除余数是属于哪一类的
test = np.array([1,9,27])
print("%d   %d => %d"%(test[1],test[2],get_v(w,test)))
test = np.array([1,11,66])
print("%d   %d => %d"%(test[1],test[2],get_v(w,test)))

Rosenblatt 感知器的局限性 只适用于线性方程分类，不适用于非线性分类


#基于梯度下降的额线性分类器
import mplannliner as nplann
traindatal = [[[9,25],-1],[[5,8],-1],[[15,31],-1],[[35,62],-1],[[19,40],-1],[[28,65],1],[[20,59],1],[[9,41],1],[[12,60],1],[[2,37],1]]
myann = nplann.mplannliner()
#样本初始化
myann.samples_init(traindatal)
#学习率初始化
myann.a_init(0.1)
#搜索时间初始化
myann.r_init()




















