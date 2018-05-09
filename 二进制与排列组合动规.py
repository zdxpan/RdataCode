# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 16:09:53 2017

@author: zdx

二进制相关运算在动态规划中的使用

对于一个边长为N，每个边有N个节点，每个节点垂直构造成的矩阵方格中，只允许从一个顶点走到对面的终点

顶点，请问，有多少种不同的走法。

@zdx:有一次跟学弟李方杰讨论，得到一种结论就是无论如何走，往下和往右的走法步数一定是相等的。

使用0表示向下走，使用1表示向右走，那么得到的是 0 跟 1 相等的全排列种数

刚才在楼下跟女朋友讨论了一个排列组合的问题，题目是我有8个0和8个1混在一起，从中挑选出任意8个

来做全排列，请问有多少种排列方式，最终答案是255种

我的解法就是不管你按照规则走哪一步，都会符合 向下 步数 = 向右 步数  。

然后将总步数的每一步变成1，求从1111xn   到0000xn的排列组合，

然后找这里面0个数跟1个数相等的那几个数字

"""

#import random#随机数



def nbit(n:int):#我要的是一个数，作为最大值边界，然后来构造所有可能的值
    #n = [1 for i in range(n)]
    x=[2**i for i in range(n)] 
    n = sum(x)+1
    return( x, n)

count = 0
n = 4#设置边长，每个边有多少个节点
x = 2*n#构造无论如何都会走2*n步到达终点的路径算法
for i in range(nbit(x)[1]):
    xin = list(bin(i)[2:])#转换成二进制数组列表
    while len(xin)<x:
        xin.insert(0,'0')
    if(xin.count('0') == xin.count('1') ):
        print(xin)
        count += 1
print("对于边长是："+str(n)+"的方阵所有的步数是:" + str(count) )


Cmn_从总步长中任意选择n步横着走 = ''

a_________________________________________= 0

def f(X:int) -> (int,int):    
    B = list(bin(X)[2:])#转换成二进制数组列表
    while len(B)<x:
        B.insert(0,'0')
    N:int = B.count('0');print(N)
    return(B,len(B)-N)   

a = int(input("input an interger:"))

def builtbin(x:int):#返回该整数的二进制字符串数组
    a = x
    count = 0
    b_in = []
    if(a==1):
        b_in = [1]
    while a>1:
        b_in.append( int(a%2))
        a = a/2
        count += 1
    while len(x)<4:
        x.insert(0,'0')
    b_in = b_in[::-1]
    #b_in = "".join(str(i) for i in b_in)
    return(b_in)



#------------------笛卡儿积----------------
from itertools import product
for x,y,z in product(['a','b','c'],['d','e','f'],['m','n']):
    # python大法好
    print(x,y,z)



#组合问题
import itertools
list1 = 'abc'
list2 = []
for i in range(1,len(list1)+1):
    iter = itertools.combinations(list1,i)
    list2.append(list(iter))





































    
    
    
    
    
    