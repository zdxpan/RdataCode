input = c(2,1,0,-1,-2,3,1,2,3,-1,-2,-3)
input = c(2,1,3,1,2,3,-1,-2,-3)
out = c(3,-1,-2,-3)

f1st=T;out = 1;out=out[-1];l=1 #第一次获取数据的标志位，结果输出的记录，长度的记录
len = 1;len = len[-1]#总长度的记录,数组
inx = numeric()#记录位置
for(i in c(  1:length(input) )   ){#尝试从每个开始往后寻找比当前位置最小的值
  
  cond = input[i] > input[i+1]#符合条件吗
  if(cond)
  {  #符合就追加结果输出，记录长度
    l=l+1 #长度记录+
    if(f1st){##刚开始符合长度选取第一个元素，第二个元素
      out = c(out,input[i],input[i+1])
      f1st = F
    }else{
      out = c(out,input[i+1])#之后符合规律要求的选取第二个元素，因为第一个元素在上一轮被选取了
    }
    print(out);
    
  }
  else{
    #print(l);print(i)
    len = c(len,l)#记录每次符合规律的组合的长度值
    inx = c(inx,i)#记录每次不符合规律的index
    out = 1;out=out[-1];f1st = T;l=1#将调试结果清空;将初次符合条件的设置为真;#将默认长度变成1
  }
  
  if(i==(length(input)-1)){#为了不超出寻址范围，限定访问倒数第二个元素，就break
    len = c(len,l)##将最后符合长度的值记录在长度记录数组中
    inx = c(inx,i+1)  #将每次不合规律的下标index+1，指向下一个合规律的元素位置
    inx = c(0,inx)#在记录位置的数组中第一个插入元素0，方便后续寻址找出目标值
    break()
  }
}
max(len)#找到最大长度符合规律的组合长度值
print(max(len))
#用该长度值来寻找对应在位置记录数组中的元素
#那个元素的前一个 和 那个元素构成了我们要得目标组合
print( input[ (   inx[ which(len==max(len)) ]+1  ) : inx[which(len==max(len)) +1] ])
