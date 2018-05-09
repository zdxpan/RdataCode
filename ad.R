
rm()


buu2016 = read.delim('C:/Users/zdx/Documents/Rtpro/buu2016new.csv',header = T,sep=",")
openxlsx::read.xlsx()

#------------------------------------------
library(RMySQL)
conn <- dbConnect(MySQL(), dbname = "test", username="localhost", password="", host="127.0.0.1", port=3306)
RMySQL::dbWriteTable(conn, "buu1", buu2016,append=T,row.names=F) 
t = dbReadTable(conn, "buu1")
sql = "select * from buu1 where '姓名' like '张锦华'"
sql = "select * from buu1 where Code='p001'"
res1 <- dbSendQuery(conn,sql) 


dbSendQuery(conn,'SET NAMES UTF8')
adt1 = dbFetch(res1, n = -1)  
names(t)[2]
t[t$姓名=='张锦华',]
#------------------------------------------




library(splitstackshape);library(openxlsx);library(dplyr)
'https://stackoverflow.com/questions/30886397/installing-gputools-on-windows'
library(rJava);library(utils)
library(Rwordseg)  install.packages("Rwordseg", repos = "http://R-Forge.R-project.org")


summarise(group_by(buu2016,学院,性别),n = n())

summarise(group_by(subset(buu2016,行政班=="管理实验班1601B"),性别),n = n())


write.csv(subset(t,学院=="机器人学院"),"C:/Users/zdx/Desktop/李方杰.csv")
t = subset(buu2016,层次 == '本科')

subset(buu2016,姓名=="郑星雨")[,"行政班"]


subset(buu2016,手机号码==13717769959)#王诗影
summarise(group_by(subset(buu2016,行政班=="艺术表演1601B"),性别),n = n())
View(summarise(group_by(subset(buu2016,学院=="艺术学院"),特长),n = n()))#行政班，民族，来源地区
#得出的结论，北京的，经济发达的地区艺术生比较多
grep("方杰",subset(buu2016,行政班=="管理实验班1601B")[,"姓名"])
grep("李方杰",buu2016$姓名)#buu2016[c(3290,7764),"姓名"]

buu2016[grep("李方杰",buu2016$姓名),'出生日期']
names(buu2016)[grep("出生",names(buu2016))]


subset(buu2016,姓名=="王泽然")
utils::winDialog("yesno", "Is it OK to delete file blah")

adname = "ilady365" ;adid = 18245

library(dplyr)
library(jsonlite)
#

beday = Sys.Date() - 1
today = Sys.Date()-1
ur1 = "http://console.mobile.dianru.com/ads_ios_api/callback.do?&actions=4&ad_from=&adid="
ur2 = "&invalid=&mac=&page=1&process=&saved=&time=1498612777&udid=&uid=&secret=e8e766d90edafc7939e230d5d54c3e1e"

url = paste(ur1,adid,"&appid=&beginDate=",beday,"&cid=&data_from=&dbtype=1&endDate=",today,ur2,sep = "")
order =  fromJSON(url)#JSON string contains (illegal) UTF8 byte-order-mark! 
order = order$list
order = dplyr::select(order,udid,create_time_title,open_time)

names(order) = c("idfa","date","time")

te = order
t = order
t$渠道 = "点入"

t = read.table("clipboard", header = T, sep = ",")#区分是哪天的数据

t = t[,c("日期","IDFA")]
t = t[,c(1,2)]
names(t) = c("date","idfa")
t = splitstackshape::cSplit(t, 'date', sep=" ", type.convert=FALSE)
names(t) = c("idfa","date","time")
t = order
te = t
t = rbind(t,te)


{ 
  nm = as.Date.character(t[1,"date"])
  nm = paste(nm,adname,nrow(t),sep = "-")
  nm = paste(nm,".xlsx",sep = "")
  nm = paste(tar,nm,sep = "")
  #write.csv(t,tar)
  openxlsx::write.xlsx(x = t,file = nm)#addWorksheet(createWorkbook("Fred"),sheetName = "zdx",gridLines = T,tabColour = "red"),asTable = T)
  rm(t,nm,adname)
}

if(1){
  wb <- createWorkbook("Fred")
  for(i in unique(t$date))
  {
    #i = unique(t$date)[1]
    temp =   t[which(t$date==i),]
    #temp$渠道 = "点入"
    sheet = as.character(i)
    sheet = stringi::stri_replace_all(sheet,replacement = "-",regex = "/")
    
    sheet = paste(sheet,"-",sep="")
    sheet = paste(sheet,as.character(nrow(temp)),sep="")
    addWorksheet(wb, sheet)
    writeData(wb, sheet = sheet, temp)
  }
  unique(t$date)
  saveWorkbook(wb, paste(tar,"2017-06-",adname,".xlsx",sep = ""), overwrite = TRUE)
  rm(i,sheet,t,temp,wb)
}

t=gm
t = arrange(t,as.Date(date))
if(1){
  wb <- createWorkbook("Fred")
  for(i in unique(t$date))
  {
    #i = unique(t$date)[1]
    temp =   t[which(t$date==i),]
    for(ii in unique(temp$type))
    {
      temp1 = temp[which(temp$type==ii),]
      #temp$渠道 = "点入"
      sheet = as.character(i)
      sheet = stringi::stri_replace_all(sheet,replacement = "-",regex = "/")
      
      sheet = paste(sheet,"-",as.character(ii),"-",sep="")
      sheet = paste(sheet,as.character(nrow(temp)),sep="")
      addWorksheet(wb, sheet)
      writeData(i,wb, sheet = sheet, temp)
    }
  }
  saveWorkbook(wb, paste(tar,adname,"-7.xlsx",sep = ""), overwrite = TRUE)
  rm(sheet,t,temp,wb,i,ii,tar,adname)
}
summarise(group_by(t,date),n = n())
#-----------------------------------------------------------------------------------------------
t = read.table("clipboard", header = T, sep = "\t",stringsAsFactors = F)#区分是哪天的数据
t$profit = t$sum_income - t$sum_cost
t$fake_rate = t$active_invalid / t$active_count
t$profit_rate = t$profit / t$sum_income
t = t[!is.na(t$fake_rate),]
t$reduce_sum = t$fake_rate * t$sum_income
t$price = t$sum_cost / t$active_count

t <- t %>% 
  mutate(profit = sum_income - sum_cost,fake_rate = active_invalid/active_count) %>%
  mutate(profit_rate = profit/sum_income,reduce_sum = fake_rate*sum_income,price = sum_cost/active_count) %>%
  filter(!is.na(fake_rate)) #%>%
  #arrange(desc(price))
names(t) = c("媒体id","媒体名称","开发者","激活有效","激活无效","激活独立","激活扣量","点入流水","二次激活成本","点入支出","线下激活"
             ,"线下激活收入","线下激活支出","激活率","总收入","总支出","总利润","假量比率","利润率","核减金额","单价")
require(openxlsx)
wb <- createWorkbook("Fred")
addWorksheet(wb, "媒体收支情况与单价",tabColour = "red")
writeData(wb, sheet = 1, t)
saveWorkbook(wb, "2017年6月媒体收支.xlsx", overwrite = TRUE)
rm(wb)

-----------------------------------------------------------------------------
#
  ur1 = "http://console.mobile.dianru.com/ads_ios_api/callback.do?&actions=4&ad_from=&adid="
  ur2 = "&invalid=&mac=&page=1&process=&saved=&time=1498612777&udid=&uid=&secret=e8e766d90edafc7939e230d5d54c3e1e"

  beday = "2017-06-01"
  beday = as.Date(beday)
  beday = beday - 1
  ilady = order[1,]
  ilady = ilady[-1,]
for(i in 1:30){
  beday = as.Date("2017-05-31")+i
  
  today = beday
  
  ur1 = "http://console.mobile.dianru.com/ads_ios_api/callback.do?&actions=4&ad_from=&adid="
  ur2 = "&invalid=&mac=&page=1&process=&saved=&time=1498612777&udid=&uid=&secret=e8e766d90edafc7939e230d5d54c3e1e"
  
  url = paste(ur1,adid,"&appid=&beginDate=",beday,"&cid=&data_from=&dbtype=1&endDate=",today,ur2,sep = "")
  order =  fromJSON(url)#JSON string contains (illegal) UTF8 byte-order-mark! 
  order = order$list
  order = dplyr::select(order,udid,create_time_title,open_time)
  names(order) = c("idfa","date","time")
  ilady = rbind(ilady,order)
  warning(beday)
}
  
  
  
  
  

url = paste(ur1,adid,"&appid=&beginDate=",beday,"&cid=&data_from=&dbtype=1&endDate=",today,ur2,sep = "")
order =  fromJSON(url)#JSON string contains (illegal) UTF8 byte-order-mark! 
order = order$list
order = dplyr::select(order,udid,create_time_title,open_time)



library(RMySQL)
conn <- dbConnect(MySQL(), dbname = "test", username="root", password="", host="127.0.0.1", port=3306)
dbGetInfo(conn)
dbListTables(conn)
dbWriteTable(conn, "adx", ad,append=T,row.names=F) #写表--------------a 哇呀
adx = dbReadTable(conn, "adx")  #读表
dbSendQuery(conn,'SET NAMES UTF8')
dbDisconnect(conn) #关闭连接

#很多项目是空字符，一律将其设定为NA
ad_t = readLines("final_ccf_test_0919",encoding = 'UTF-8',n = 10000)
ad_t=read.table(text=ad_t,sep="\x01",head = T,stringsAsFactors=F)
names(ad) = rownames
#write(ad,"adx.txt")
'分析思路：先将数据按照天读入，每天的数据按照媒体类型分类，再将每个类别按照作弊标记分类，求出每个分类所占的比率。'

#将数据导入mysql，方便以后读取
fl=file.choose()  
c=file(fl,"r")
on.exit(c)
rlnm=readLines(c,encoding = 'UTF-8',n=1)  #第一行为列名
mks=1
nrows=1000             #设置处理的行数
while(T)
{
  rl=readLines(c,n=nrows)   #读nrows行
  #if(mks>640*10000)
  if(length(rl)==0)
  {
    break
  }
  else
  {
    rl=c(rlnm,rl)
    mks=mks+nrows
    data=read.table(text=rl,sep="\x01",head = T,stringsAsFactors=F) 
    names(data) = rownames
    #print(data[1,])
    #从rl_all中读到data.frame
    #myfun(data)
    dbWriteTable(conn, "adtest", data,append=T,row.names=F) #写表--------------a 哇呀
    cat(cat("已处理",mks,"行\n"))
  }
}
close(c)

mydata[USERDATA1 %in% sec_type,] unique(mydata$USERDATA1)------某列选取 
sec_type <- c("ZF债", "公司债", "金融债", "可转债", "短期融资券")

ad_afect[which(ad_afect$count_date=="2016/10/8" & ad_afect$keywordId==24593),]

i="2016/10/7"
j=17125
k = 24521
m = 0
for (i in unique(iosadafect$count_date)) {
  print(paste("共统计第",i,"天的数据"))
  temp = iosadafect[which(iosadafect$count_date == i),]
  for (j in unique(temp$adid)) {
  print(paste("广告为",j))
  tempj = temp[which(temp$adid==j),]
    for (k in unique(tempj$keywordId)) {
      print(paste("关键词是",k))
      tempi = tempj[which(tempj$keywordId==k),]
      #此处结合这个键来对原始数据库框补充处理
      s = sum(tempi$num)
      m = m +1  
      #tempi[i]/
      iosadafect[which(iosadafect$count_date==i & iosadafect$adid==j & iosadafect$keywordId==k ),c("rate")] = tempi$num*tempi$ads_effect*10/s
    }
  }
}

library(dplyr)

n = 0
who = names(ad)[2:8]
who <- lapply(who, as.symbol)

temp =   adx %>%
  group_by_(.dots=who) %>%
  summarise(n = n())
---------------------++++++++++++++++++'**********************************'+++++++++++------------------------------


pheatmap(data, color = colorRampPalette(c("navy", "white", "firebrick3"))(50), fontsize=9, fontsize_row=6) #自定义颜色

#s    
矩阵转置
a = matrix(c(1:10),2,5)
b = matrix(c(101:110),2,5)
cbind(a,b)
rbind(a,b)
a = array(a,dim=c(2,5))
aperm(a,perm=c(2,1)) = b
b = t(a)
矩阵乘积
a %*% b
向量内外积
a=啊 = array(c(1:3))
b=哦 = t(array(c(4:6)))
b= array(c(4:6))
啊 %o% 哦
周到先 = 123
crossprod(哦,啊)
crossprod(a,b)
list(name="student",class="101",atdt.ages=c(22,12,20),stdt.name=c("zhang","lily","wangwu")) -> mystu
--------------------------->>>>>>>>>>>>>........2017年1月23日-------------------
R语言的聚类方法层次聚类K
Kmeans dbscan

1.距离和相似系数
dist(x,method = "euclidean",diag = FALSE,upper = FALSE,p=2)计算距离，x是样本矩阵或者数据狂，
    method取值：euclidean 欧几里得距离
    maximun 切比雪夫距离
    manhattan  绝对值距离
    canberra   lance距离，使用时要制定P值
    binary  定性变量距离
定性变量距离
diag = T,对角线上距离，
2 中心化
scale(x,center = T,scale = T)  
3 sweep(x,MARGIN = ,STATS = ,FUN = "-")对矩阵进行运算 margin = 1对行方向进行运算，2表示列方向进行运算
  STATS 是运算参数，FUN为运算函数，默认是剑法，下面利用sweep对矩阵x进行极差标准化变换
  center <- =sweep(x,2,apply(x,2,mean))#在列的方向上减去均值
  R <- apply(x,2,max) - apply(x,2,min)#算出极差，即列上最大值-最小值
  x_star <- sweep(center,2,R,"/") #把减去均值后的举证在lieder方向上减去极差向量

4.有时候我们不是对样本进行分类，而是对变量进行分类，这时候，我们不计算距离，而是计算变量间的相似系数
  R语言计算两向量的夹角余弦
  y <- scale(x,center=F,scale = T)/sqrt(nrow(x)-1)
  C <-t(y) %*% y
5.层次聚类法
  先计算样本之间的距离，每次将距离最近的点合并到同一类，然后再计算类与类之间的距离，将距离最近的类合并为一个大类，不停的合并直到合并成了一个类，
  其中类与类之间的距离的计算方法有：最短距离发，最长距离法，中间距离发，类平均法，比如最短距离发，将类与类的定义为类与类之间同样本的最短距离

  R hclust(d,method - 'complete',members = NULL)  #层次聚类，其中d,为矩阵距离。
  method表示类的合并方法，有：
    single 最短距离法
    conplete 最长距离法
    median 中间距离法
    mcqulity 相似法
    average 类平均法
    centrold 重心法
    ward 离差平方和法
   try:
   x <- c(1,2,6,8,11)
   dim(x) <- c(5,1)
   d <- dist(x)
   hcl <-hclust(d,"single")
   plot(hcl)
   plot(hcl,hang=-1,type = "trangle")
   #type = c("retangle","trangle")默认树形图
   #horiz T 表示竖着放，FLALSE，表示横着放

   例如：对305名同学测量8个体型指标，相应的相关矩阵如表，将相关系数定义成相似系数，定义距离为
   dij = 1-rij
   用最长距离法做系统分析
   

if(match("***** 2.REG",x))#必须全部比配
  stri_detect(x,"***** 2.REG",)
temp = left_join(unmatch,dat,by = c("idfa"))

# to load an twitter data api interface of
library("rjson")
install.packages(c("rjson", "bit64", "httr"))
#处理一个文本数据文件
match("***** 2.REG",x)
stri_detect(x,regex = '2.REG')

x[stri_detect(x,regex = '2.REG')]
x[1:5]
1.reg = [a:b]
 stri_detect(x,regex = '1.reg') #T FFFFFF T
 stri_detect(x,regex = '2.REG') #F FFT FFFF

x_in1 = union(which(stri_detect(x,regex = '1.reg')) , (which(stri_detect(x,regex = '2.REG'))-1))
x_in1 = union(c(which(stri_detect(x,regex = '1.reg'))) , c(which(stri_detect(x,regex = '2.REG'))-1))
a=c(which(stri_detect(x,regex = '1.reg')))
b=c(which(stri_detect(x,regex = '2.REG'))-1)


which(c(1:5) %in% c(3:7) &  c(1:10) %in% c(4:7))


a[grep("pp_",a)]
rm(a)
save(pp_dianru,pp_office,pp_inner,pp_dr,pp_of,pp_of_dr,pp_office_had_while_dianru_not, file="E:/1workdianru/pp_office_dr.RData")
load("E:/1workdianru/pp_office_dr.RData")









#新意互动：
x = seq(-10,10,by = 0.01)
y = 1/(1+exp(-x))
y = 1+exp(-x)
plot(x,y)
plot(x,1/(1+exp(-x)))
plot(-log(1/(1+exp(-x))))




