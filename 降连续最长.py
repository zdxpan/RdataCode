# -*- coding: utf-8 -*-
#动态规划解决方案
input = [2,1,0,-1,-2,3,1,2,3,-1,-2,-3]

input = [2,1,3,1,2,                      3,-1,-2,-3]
jilu = []#-------------记录断点---位置---index-------------
for i in range(0,len(input)):
    cond = input[i] < input[i+1]#符合条件吗
    if(cond):
        jilu.append(i)#记录不连续的断点
        
    if(i==(len(input)-2)):
        jilu = [x + 1 for x in jilu] #将每次不合规律的下标index+1，指向下一个合规律的元素位置
        #为了不超出寻址范围，限定访问倒数第二个元素，就break
        break
    
sta_jilu = jilu.copy()       
jilu.insert(0,0)      #方便运算
jilu.append(len(input))#方便运算
le2 = []#------------每个子序列的长度
pos = []#------------每个子序列的起始位置
for x in range(0,len(jilu)-1):                 #--------记录的是index
    if(jilu[x+1]-jilu[x]>1):
        le2.append(jilu[x+1]-jilu[x] - 1)#记录长度，
        pos.append(jilu[x])              #记录长度对应的位置

print("max length is:",max(le2))#寻找最大长度
look =  pos[le2.index(max(le2))]#
print("result is:",input[  pos[le2.index(max(le2))]: jilu[jilu.index(look) + 1] ])
    







def removeDuplicates(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:return
        a = nums[0]    
        i=1
        while i < len(nums):
            if nums[i] == a :
                del nums[i]       
            else:
                a = nums[i]
            i += 1
        return len(nums)
             
nums = [1,1,2,2,3,4]
i=2
