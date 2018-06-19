# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 09:03:31 2017

@author: cj
"""
d = {'a':1,2:'b','c':3,4:'d'}

del(d['a'])
del[d[2]]
print(d)
d = {1,2,3,4}
if not d:
    print('d is empty')
print('')
d = {'a':1,2:'b','c':3,4:'d'}
for k in d.keys():
    print(str(k) + ': ' + str(d[k]))
for k, v in d.items():
    print(str(k) + ': ' + str(v))

d = ['a',1,2,'b','c',3,4,'d']
s = 'abcd'
d = list(s) #list 将字符串转化为数组
d[0] = 'A'
d[1] = 'B'
print(d)
s = ''.join(d)
print(s)

d = [4,5,6]
for i in d:
    print(i)
for i in range(len(d)):
    print(d[i])

s_b = set([1,1,2,2,3,4,5,5,6])






import  requests#网络请求模块
import time
import configparser#新建ini 配置文，我的弹幕
import random
'''
i=挖掘机小王子
2 = 老司机
3 = 翻车
4 = 老鼠吃大米
'''


url  = ' '
form = {'front‘：
        
        
        }
cookie = {'Cookie':""}#to file文件
requests.request('POST',url,data = form,cookie = cookie)
while True:
    message = target['我的檀木'][str(]




def fib(limit):
    n,a,b = 0,0,1
    while n < limit:
        yield b
        a,b = b,a+b
        n += 1
    return 'done'
for i in fib(10):
    print(i)


print(r'''hello,
      world''')



n = 123
f = 456.789
s1 = 'Hello, world'
s2 = 'Hello, \'Adam\''
s3 = r'Hello, "Bart"'
s4 = r'''Hello,
Lisa!'''
print(s4)

'ABC'.encode('ascii')
 b'ABC'.decode('ascii',errors= 'ignore')
print('%2d-%02d' % (3, 1))
print('%.2f' % 3.1415926)
 'Age: %s. Gender: %s' % (25, True)



L = [
    ['Apple', 'Google', 'Microsoft'],
    ['Java', 'Python', 'Ruby', 'PHP'],
    ['Adam', 'Bart', 'Lisa']
]

print(L[0][1])


heigh = 1.75
weigh = 80.5
BMI = weigh / heigh**2
if BMI<18.5:
    print('guoqing')
elif 



s = {'iven':100,'kven':97,'linus':89,'lary':79,'harri':90}
name = input('请输入要查找的姓名:') 
ssd =s.get(name,False) 
if ssd == False: 
    print('搜索结果:您输入的名字不存在!\n') 
else: 
    x = int(s[name])
    print(name,'同学你好,你的分数是:',x,)

def f2(a=1,b=2, c=0, *, d, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw)
f2(1, 2,d=9)











































































