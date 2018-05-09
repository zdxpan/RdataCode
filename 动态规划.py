# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:32:13 2017
---------------------递归，动态规划

@author: zdx
题目描述：
如下三角形数组

7
3 8
8 1 0
2 7 4 4
4 5 2 6 5
计算上面的数字三角形中寻找一条从顶部到底边的路径，使得路径上所经过的数字之和最大。
路径上的每一步都只能往左下或 右下走。只需要求出这个最大和即可，不必给出具体路径。
 三角形的行数大于1小于等于100，数字为 0 - 99
"""

#import pdb


D = [
      [7],
      [3, 8],
      [8,1,0],
      [2,7,4,4],
      [4,5,2,6,5],
]
a = ''' 状态转移过程
D1 =	4	5	2	6	5	=	D[5]
D2 =	2	7	4	4	0		D[4]
R1 =	6	12	6	10	5		
R2 = 	7	9	10	9			
M =	7	12	10	10			

'''
x = range(len(D)-1,0,-1)
#D[::-1]  ----倒序输出
M = D[len(D)-1]  
for i in x :
    d1 = D[i-1].copy()
    d2 = M
    #长度的判断   补充长度算法：：：：：：
    R1 = []
    R2 = []
    d1.append(0)
    R1 = list(map(lambda x: x[0]+x[1], zip(d2, d1)))   
    
    R2 = list(map(lambda x: x[0]+x[1], zip(d2[1:], d1)))   
    
    M = list(map(lambda x: max(x), zip(R1, R2)))   
    print(R1)
    print(R2)
    print(M)
    print("")
    #pdb.set_trace() 
    return M
    #input("set trace")
  
'''----------------------------使用0 来填充数组------------------- '''
D = [ [7,0,0,0,0],
      [3, 8,0,0,0],
      [8,1,0,0,0],
      [2,7,4,4,0],
      [4,5,2,6,5]]
import numpy as np

D = np.array(D,dtype = np.int0)

M = D[len(D)-1]
x = range(len(D)-1,0,-1)    
for j in x:
    d1 = D[j-1].copy()
    d2 = M
    R1 = np.array([d1[i] + d2[i]  for i in range(len(d2))  ])
    R2 = d1[:len(d2)-1] + d2[1:]#[d1[i] + d2[i+1]  for i in range(len(d2)-1)]
    print(R1)
    print(R2)
    M = np.array([max(R1[i],R2[i])   for i in range(len(R2))])
    
    #M = list(map(lambda x: max(x), zip(R1, R2)))
    
    #M = np.array(M)    
    print(M)
    input("xxx\n")
    
    





'''
http://blog.csdn.net/baidu_28312631/article/details/47418773

接下来，我们就进行一下总结：

    递归到动规的一般转化方法

    递归函数有n个参数，就定义一个n维的数组，数组的下标是递归函数参数的取值范围，数组元素的值是递归函数的返回值，这样就可以从边界值开始， 逐步填充数组，相当于计算递归函数值的逆过程。

    动规解题的一般思路

    1. 将原问题分解为子问题

        把原问题分解为若干个子问题，子问题和原问题形式相同或类似，只不过规模变小了。子问题都解决，原问题即解决(数字三角形例）。
        子问题的解一旦求出就会被保存，所以每个子问题只需求 解一次。

    2.确定状态

        在用动态规划解题时，我们往往将和子问题相关的各个变量的一组取值，称之为一个“状 态”。一个“状态”对应于一个或多个子问题， 所谓某个“状态”下的“值”，就是这个“状 态”所对应的子问题的解。
        所有“状态”的集合，构成问题的“状态空间”。“状态空间”的大小，与用动态规划解决问题的时间复杂度直接相关。 在数字三角形的例子里，一共有N×(N+1)/2个数字，所以这个问题的状态空间里一共就有N×(N+1)/2个状态。

    整个问题的时间复杂度是状态数目乘以计算每个状态所需时间。在数字三角形里每个“状态”只需要经过一次，且在每个状态上作计算所花的时间都是和N无关的常数。

    3.确定一些初始状态（边界状态）的值

    以“数字三角形”为例，初始状态就是底边数字，值就是底边数字值。

    4. 确定状态转移方程

     定义出什么是“状态”，以及在该“状态”下的“值”后，就要找出不同的状态之间如何迁移――即如何从一个或多个“值”已知的 “状态”，求出另一个“状态”的“值”(递推型)。状态的迁移可以用递推公式表示，此递推公式也可被称作“状态转移方程”。

    数字三角形的状态转移方程: 

   
 

    能用动规解决的问题的特点

    1) 问题具有最优子结构性质。如果问题的最优解所包含的 子问题的解也是最优的，我们就称该问题具有最优子结 构性质。

    2) 无后效性。当前的若干个状态值一旦确定，则此后过程的演变就只和这若干个状态的值有关，和之前是采取哪种手段或经过哪条路径演变到当前的这若干个状态，没有关系。
﻿﻿

'''
最长公共子序列（POJ1458) = 1
“我为人人”递推型动归

    状态i的值Fi在被更新（不一定是 最终求出）的时候，依据Fi去更 新（不一定是最终求出）和状态i 相关的其他一些状态的值 Fk,Fm,..Fy


1.找子问题 ::
    “求以ak（k=1, 2, 3…N）为终点的最长上升子序列的长度”，一个上升子序列中最右边的那个数，称为该子序列的 “终点”。
    
2.2.确定状态

    子问题只和一个变量—— 数字的位置相关。因此序列中数的位置k就是“状态”，而状态 k 对应的“值”，
    
    就是以ak做为“终点”的最长上升子序列的长度。 状态一共有N个。    
    
    
3.初始状态：maxLen (1) = 1

 maxLen (k) = max { maxLen (i)：1<=i < k 且 ai < ak且 k≠1 } + 1       若找不到这样的i,则maxLen(k) = 1    
 
 
 
a = [1,7,3,5,9,4,8]
'''
------rell == 1 3 5 8  ==4  :   
'''
out = 4    

maxLen = [ 1 for x in range(len(a) - 1) ]
#---------------------------结果不对
for i in range(len(a)-1):
    for  j in  range(0,i):#----------人人为我: 遍历每个元素，查找以该节点为终点的最长子式
        print(j)
        if(a[j] > a[i]  ):
            print(True)
            maxLen[j] = max(maxLen[j],maxLen[i] + 1)

'''-------------------------------************************---------------'''

for i in range(len(a)-1):
    for  j in  range(i+1,len(a)-1):#--我为人人-查找当前每个节点都能做后面谁的小 子式------
        print(j)
        if(a[j] > a[i]  ):
            print(True)
            maxLen[j] = max(maxLen[j],maxLen[i] + 1)


                        


def two_sum(self,nums,target):
    if len(nums) <= 1:
        return False
    dictionary = {}
    for i in range(len(nums)):#0,1,2,3,4#字典查找不费时间？？？？
        compliment = target - nums[i]# 6  -5  2   7
        print(i,compliment)
        if compliment in dictionary:
            print([dictionary[compliment],i  ])
            #return [dictionary[compliment],i  ]
        else:
            dictionary[nums[i]] = i
    return False


nums = [3, 14, 7, 2];target = 9





import datetime

starttime = datetime.datetime.now()

#long running

endtime = datetime.datetime.now()

print (endtime - starttime).seconds


a = [1,2,3,5,7]
target = 9


def twoSum(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i, value in enumerate(nums):#转换为list中的元组形式 index,   
            diftar = target - value   #0,1,2,3,4
                                      #1,2,3,5,7
                                      #8,7,6,4,2
            if diftar in nums[i+1:]:#在之后的数组中查找
                index2 = nums[i+1:].index(diftar) + i + 1#使用index 搜索，不算时间浮渣度
                #a[1+1:].index(7) + 1 + 1
                return [i, index2]

twoSum(a,target)


class Solution(object):
    def merge(self, nums1, m, nums2, n):
        
        if not nums1 or not nums2:
            return
        
        i = m-1
        j = n-1
        k = m+n-1
        
        nums1 = nums1+nums2
        
        while i >= 0 and j >= 0 :
            if nums1[i] > nums2[j]:
                nums1[ k ] = nums1[i]
                i = i-1
            else:
                nums1[ k] = nums2[j]
                j=j-1
            k = k - 1
            
        if j > 0 :
            nums1[:k+1] = nums2[:k+1]
        return nums1
    
    
merge(1,nums1,len(nums1),nums2,len(nums2))

nums1 = [4,5,6,9,11]
nums2 = [1,2,3,7,8,20]


nums1 = [1,2,3,7,8,20]
nums2 = [4,5,6,9,11]
m=6;n=5


if __name__ == '__main__':
    sol = Solution()
    print(sol.merge(nums1,len(nums1),nums2,len(nums2)))

nums1 = [1]
nums2 = [ ]
sol.merge(nums1,1,nums2,0)


set(nums1)



def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        braket_dict = {'[':']', '(':')','{':'}' }
        stack = []
        for char in s:
            #print(char in braket_dict.keys())
            if char in braket_dict.keys():
                stack.append(braket_dict[char])
            else:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    return False
        if stack:return False
        return True
    


s = '()'
char = ')'



    






def threeSum(self, nums):
    res = []
    nums.sort()
    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:#avoid duplicated
            continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l +=1 
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
    return res




def qsort(a,left,right):
    i,j = left,right
    if i >= j:
        return a
    key = a[i]
    while i < j:
        while i<j and a[j] >= key:
            j = j - 1
        a[i] = a[j]
        while i < j and a[i] <= key:
            i = i + 1
        a[j] = a[i]
        
    a[i] = key
    qsort(a,left,i-1)
    qsort(a,j+1,right)
    return a
a = [1,9,2,7,3,6,4,5,8,100]
qsort(a,0,len(a)-1)        





class Solution(object):#上升序列原则
    def maxSlidingwindow(self,nums,k):
        if not nums:return []
        window = []
        res = []
        for i,v in enumerate(nums):
            #第一件事就是窗口的移入，之后的移出
            window.append(i)# 0,1,2,3,4,5,6
            #移出的规则，满足的条件，最小的值，维护最大的在右边
            if window and window[0] <= i - k:#删除第一个元素,保持窗口的移动
                window = window[1:]#window.pop(0)
            while window and nums[ window[-1] ] < v:
                window.pop()
           
            if i + 1 >= k:
                res.append(nums[window[0]])
        return res
            
    ans = []
    queue = []
    for i, v in enumerate(nums):
        if queue and queue[0] <= i - k:#移入的元素已经多于K个了？那就去掉吧
            queue = queue[1:]#queue.pop()
        while queue and nums[queue[-1]] < v:
            queue.pop()
        queue.append(i)
        if i + 1 >= k:
            ans.append(nums[queue[0]])
    return ans                
            

 

java#优先队列，大顶堆
pq  = priorityQueue<int>(k,collections.reverseOrder())#pq 是slidingwindow
for i in range(len(nums)):
    if i>=k:
        pq.remove(nums[i-k])
    pq.add(nums[i])
    if i>=k-1:
        result[i - (k - 1)] = pq.peek()
return result




#最低公共节点
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None



a = TreeNode([1,2,3,4,5,6,7])

'最低公共节点' == 0
def LCA(self,root,p,q):
    if root is None or root == p or root == q:return root
    left = self.LCA(root.left,q,p)
    right = self.LCA(root.right,q,p)
    
    if left and right:return root
    return left if left else right





def recursion(level,param1,param2,...):
    
    #recursion terminator
    
    if level > MAX_LEVEL:
        print_result
        return
    #process logic in current level
    Process_data(level,data...)
    
    #dril down
    self.recursion(level+1,p1,...)
    
    #reverse the current level status  if needed
    reverse_state(level)




#链表

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # @param {ListNode} head
    # @return {ListNode}
    def reverseList(self, head):
        if head == None:
            return head
        reservedHead = None
        cur = head
        pre = None
        while cur:
            new = cur.next
            if not new:
                reservedHead = cur
            cur.next = pre
            pre = cur
            cur = new            
        return reservedHead
#以下新建一个链表
s = Solution()
head = ListNode(0);
cur = head
for i in range(1, 10):
    node = ListNode(i)
    cur.next = node
    cur = node
#这是新建的链表    
head = s.reverseList(head);
while(head != None):
    print(head.val, end=' ')
    head = head.next


#堆排序
import random 
def maxheapfy(heap,heapsize,root):
    left = (root << 1) + 1
    right = (root << 1) + 2
    larger = root
    if left < heapsize and heap[larger] < heap[left]:
        larger = left
    if right < heapsize and heap[larger] < heap[right]:
        larger = right
        
    if larger != root:
        heap[larger],heap[root] = heap[root],heap[larger]
        maxheapfy(heap,heapsize,larger)
        
def buildmaxheap(heap):
    heapsize = len(heap)
    #for i in range((heapsize -2)//2,-1,-1):
    for i in range((heapsize >>1 )-1,-1,-1):
        maxheapfy(heap,heapsize,i)
def heapsort(heap):
    buildmaxheap(heap)
    for i in range(len(heap)-1,-1,-1):
        heap[0],heap[1] = heap[i],heap[0]
        maxheapfy(heap,i,0)
    return heap

def topk(nums,k):
    buildmaxheap(nums)
    for i in range(k):
        nums[0],nums[heapsize-1] =  nums[heapsize-1],nums[0]
        heapsize -=1
        maxheapfy(nums,heapsize,0)
    return nums[heap_size]
a = [30,50,57,77,62,78,94,80,84]

heapsort(a)


import heapq
a = []
heapq.heappush(a,0)
heapq.heappop(a)
#


import random

def MAX_Heapify(heap,HeapSize,root):#在堆中做结构调整使得父节点的值大于子节点

    left = 2*root + 1
    right = left + 1
    larger = root
    if left < HeapSize and heap[larger] < heap[left]:
        larger = left
    if right < HeapSize and heap[larger] < heap[right]:
        larger = right
    if larger != root:#如果做了堆调整则larger的值等于左节点或者右节点的，这个时候做对调值操作
        heap[larger],heap[root] = heap[root],heap[larger]
        MAX_Heapify(heap, HeapSize, larger)

def Build_MAX_Heap(heap):#构造一个堆，将堆中所有数据重新排序
    HeapSize = len(heap)#将堆的长度当独拿出来方便
    for i in range((HeapSize - 2)//2,-1,-1):#从后往前出数
        MAX_Heapify(heap,HeapSize,i)

def HeapSort(heap):#将根节点取出与最后一位做对调，对前面len-1个节点继续进行对调整过程。
    Build_MAX_Heap(heap)
    for i in range(len(heap)-1,-1,-1):
        heap[0],heap[i] = heap[i],heap[0]
        MAX_Heapify(heap, i, 0)
    return heap

if __name__ == '__main__':
    a = [30,50,57,77,62,78,94,80,84]
    print( a)
    HeapSort(a)
    print( a)
    b = [random.randint(1,1000) for i in range(1000)]
    print( b)
    HeapSort(b)
    print( b)

class Solution {
public:   
    inline int left(int idx) {
        return (idx << 1) + 1;
    }
    inline int right(int idx) {
        return (idx << 1) + 2;
    }
    void max_heapify(vector<int>& nums, int idx) {
        int largest = idx;
        int l = left(idx), r = right(idx);
        if (l < heap_size && nums[l] > nums[largest]) largest = l;
        if (r < heap_size && nums[r] > nums[largest]) largest = r;
        if (largest != idx) {
            swap(nums[idx], nums[largest]);
            max_heapify(nums, largest);
        }
    }
    void build_max_heap(vector<int>& nums) {
        heap_size = nums.size();
        for (int i = (heap_size >> 1) - 1; i >= 0; i--)
            max_heapify(nums, i);
    }
    int findKthLargest(vector<int>& nums, int k) {
        build_max_heap(nums);
        for (int i = 0; i < k; i++) {
            swap(nums[0], nums[heap_size - 1]);
            heap_size--;
            max_heapify(nums, 0);
        }
        return nums[heap_size];
    }
private:
    int heap_size;
}


nums=[1,2,3]
def subsets(self, nums):
    res = [[]]
    for num in sorted(nums):
        res += [item+[num] for item in res]
    return res
subsets(1,nums)


# DFS recursively 
def subsets1(nums):
    res = []
    dfs(sorted(nums), 0, [], res)
    return res
    
def dfs(nums, index, path, res):
    res.append(path)
    for i in range(index, len(nums)):
        dfs(nums, i+1, path+[nums[i]], res)
    
nums = [1,2,3]    
subsets1(nums)

def multiply(num1, num2):
    a,b = list(num1),list(num2)
    m,n = len(num1)-1,len(num2)-1    
    rel = [0] * (m+n+2)
    for i in range(n,-1,-1):#被乘数
        for j in range(m,-1,-1):#被除数
            k = i+j
            mul = int(a[i]) * int(b[j])
            x = mul + rel[k+1]                #美味结果相加
            rel[k] += x/10
            rel[k+1] = x/10


    return ''.join( map(str,rel[::-1]) )
    

#树形结构‘’
'列举出{0,1,2,3,4}的所有子集'==0

solution = []#用来存放可能的数据
def backtrack(n):
    if n==5:
        print(solution)
        return
    ' // 取数字n，然后继续列举之后的位置'==0
    solution[n] = True
    backtrack(n+1)
    '// 不取数字n，然后继续列举之后的位置'==0
    solution [ n ] =  False 
    backtrack ( n + 1 )
    
solution(0)








