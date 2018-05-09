install.packages("installr");

require(installr) #load / install+load installr

updateR() 
'
监督学习：决策树
        随机森林
        KNN神经网络
        逻辑回归
非监督学习
          K——卡尔曼
          层次聚类
加强学习：
        Q学习
        马克决策过程
1：线性回归 Liner regression
'
  #load Train and test datasets
  #Identify feature and feature and response Variables() and
  #values must be numeric and numpy arrays
  x_train  = input_variables_values_training_datasets
  y_train = target_variables_values_training_datasets
  x_test = input_variables_values_test_datasets
  x = cbind(x_train,y_train)
  #train the model using the training sets and check score
  Linear = lm(y_train ~.,data = x)
  summary(Linear)
  #predect Output
  predicted = predict(linnear,x_test)
  
'2.逻辑回归 Logic regression'
  x = cbind((x_train,y_train))
  #train the model using the training set and check score
  logisitic = glm(y_train ~.,data = x,family = 'binomial')
  summary(logisitic)
  predicted = predict(logisitic,x_test)
'3.---------------------->>>>----->>>>>>>>>>> Decesion Tree 决策树'
  #import Library
  library(rpart)
  x<- cbind(x_train,y_train)
  #grow tree
  fit = rpart(y_train ~.,data = x,method = "class")
  summary(fit)
  #predict output
  predicted = predict(fit,x_test)
  
'4.---------->>>>>>----  SVM支持向量机'
  #Import Library 
  install.packages("e1071")
  library(e1071)
  x = cbind(x_train,y_train)   
  #Fitting model
  fit = svm(y_train ~.,data = x)
  summary(fit)
  #predict outputs
  predicted = predict(fit,x_test)
  
'5------------>>>贝叶斯回归'
  #import Library
  library(e1071)
  x = cbind(x_train,y_train)
  #fitting model
  fit = naiveBayes(y_train ~.,data=x)
  #这样据说部分可以解决该问题
  options(download.file.method = "wget")
  install.packages("e1071", dependencies=TRUE, repos='http://cran.rstudio.com/')
'6.KNN K近邻分类'
  #Import library
  install.packages("knn")
  library(knn)
  x <- cbind(x_train,y_train)
  #fitting model
  fit <- knn(y_train ~.,data=x,k=5)
  summary(fit)
  #predict output
  predicted = predict(fit,x_test)
'7.  K-means算法是很典型的基于距离的聚类算法，采用距离作为相似性的评价指标，
  即认为两个对象的距离越近，其相似度就越大。该算法认为簇是由距离靠近的对象组成的，因此把得到紧凑且独立的簇作为最终目标。'
  #importlibrary
  library(cluster)
  fit <- kmeans(x, centers = 3)
  fit.center
'#5 clutter solution
  python code:
    from sklearn.cluster import kmeans
    #asnemde you hava x(attributes),for training data set and x_test 
    #(attritubutes of test dataset
    )
    #create KMeighbors classfiter object_states=0)
    #train the model using the training sets and check score 
    model.fit(x)
    #predicted = mode.predict(x_test)
    #predict  oyr
    predicted = model,predict(x_test)
'
'8.Random Forest 随机森林'
  #import library
  library(randomForest)
  x <- cbind(x_train,y_train)  
  #fitting model
  fit <- randomForest(Species ~.,x,ntree=500)
  summary(fit)
  #predict
  1+1
  predicted = predict(fit,x_test)
'9.Dimensionality reduction algorithms 降维算法'
  #Import Library
  library(stats)
  pca <- princomp(train,cor = TRUE)
  train_reduced <- predict(pac,train)
  test_reduced <- predict(pca,test)
'10.Gredient boosting&adboost,tidings上升下降'
  Library(caret);#install.packages('caret')
  x <- cbind(x_train,y_train)
  #fitting model
  fitControl <- trainControl(method = "repeatedcv",+number = 4 ,repeats = 4)
  fit <- train(y ~.,data = x,method = "gbm",+ trControl = fitControl,verbose = FALSE)
  predicted = predict(fit,x_test,type = "prob")[,2]



y<-list.files(pattern=".xlsx")
z<-lapply(y,function(x) read.xlsx(x,sheetIndex = 1,header=T,encoding = 'UTF-8'))
t= read.xlsx("2016-10ios.xlsx",sheetIndex = 5,header=T,encoding = 'UTF-8')

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
  outer(a,b,"*") 
  crossprod(哦,啊)

  crossprod(a,b)
  '5.求解线性方程组 一般通过solve函数来求解a%*%x = b向量值，求解线性方程组仅使用solve函数的前两个参数，
  第一个为系数矩阵，第二个为常数矩阵，当第二个b缺失时，默认为单位矩阵'
  a %*% x  = b
  b = array(c(4:6),dim = c(1,3))#系数矩阵
  a = array(c(1:3),dim = c(1,3))#常数矩阵
 ' 6。 矩阵求逆'
  solve(a)
  '7 . 求解矩阵特征值
      对于线性系统 (A-rl)x = 0 的x解
      特征多项式零点'
    eigen(x,symmetric,only.value = FALSE )
    ' only.value = T 返回特征值和特征向量'
 ' 8 求解矩阵行列式
  将nxn矩阵映射到一个标量 |a| 行列式可看成是有向面积在欧几里得空间里的推广'
   det(a) #0---------------------
  '9 奇异分解 
  M (m*n)阶矩阵，其中的元素全部属于K
  存在一个分解使得M=UEv*'
  svd(M)




list(name="student",class="101",atdt.ages=c(22,12,20),stdt.name=c("zhang","lily","wangwu")) -> mystu
'--------------------------->>>>>>>>>>>>>........2017年1月23日-------------------
R语言的聚类方法层次聚类K
Kmeans dbscan
'
'1.距离和相似系数'
dist(x,method = "euclidean",diag = FALSE,upper = FALSE,p=2)#计算距离，x是样本矩阵或者数据狂，
'   method取值：euclidean 欧几里得距离
    maximun 切比雪夫距离
    manhattan  绝对值距离
    canberra   lance距离，使用时要制定P值
    binary  定性变量距离
定性变量距离
diag = T,对角线上距离，
2 中心化
'
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
  先计算样本之间的距离，每次将距离最近的点合并到同一类，然后再计算类与类之间的距离，将距离最近的类合并为一个大类，
  不停的合并直到合并成了一个类，  其中类与类之间的距离的计算方法有：最短距离发，最长距离法，中间距离发，类平均法，
  比如最短距离发，将类与类的定义为类与类之间同样本的最短距离

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



1....基础   
  文件载入并执行代码：
  source("f:/pro/r/test.r")
  最后将执行结果写入文件
  sink("f/pro/r/test.lis")
  在代码行尾使用“+” 续写代码
1.1 可以使用 sort length sqrt 排序，求长度，求平方根
  复数
  c(3:(22))
  c(3*1:10) #冒号优先权很高
  seq生成序列 seq(from, to, by = 3)
  rep(x,2),times 参数作为整体拼接，each作为元素拼接
1.4 字符串向量
  可以使用转义字符\:
  \n 新行
  \r 回车  
  \t tab
  \b 退格
  \a 鸣叫
  \' '
  \" "
  paste(1:12)#可将他们一次链接在字符串元素中，sep指定了相隔的字符，默认空格
  paste("A",1:6)# "A 1" "A 2" "A 3" "A 4" "A 5" "A 6"
  paste("A",1:6,sep = "")#"A1" "A2" "A3" "A4" "A5" "A6"
  paste(c("A","B"),1:6,sep = "")
  
1.5 索引向量
  ·1 逻辑型索引中，索引向量的元素为逻辑值  
  ·2 正整数索引
  x[c(1,2,3,4,56,8)]
  ·3 负数索引：  会将除索引以外的所有元素输出到结果
  ·4 字符串索引  那就用字符串标注元素所在的位置
1.6 对象集属性
  对象集雇佣属性mode和length两种， mode为对象集的类型 mode可以完成数据类型的转化，比如as.character转化字符串等
  attr方法设置对象属性自定义，具体方法attract(object,name)

1.7 银子和有序因子
  用于储存类别变量和有序变量，可以用来分组或者分类
  levels是因子水平变量，labels是因子的标签便向量
  ·1 cut函数将数据转换为因子或者有序因子，并进行分组，下面对一组学生成绩进行分组
  score <- c(88,99,12,2321,312,312,312,4,,5322,2)  
  cut(score,breaks = 3)#将因子分成三组
1.8 循环
  for 不能设置步长之类的
  While检测条件

3.3  R 语言科学计算  
  分类分分组，统计
  furit = c("apple","piaaple","dadawdwa ")
  furit_prices = (1,2,3,4,5)
  1 平均价格统计 通过 tapply 函数中指定mean作为参数，可以实现分组求平均值，计算结果分2行
   tapply(furit_prices,furit,mean)
  2，最低价格统计
     tapply(furit_prices,furit,min)
  3, 标准差统计 
   tapply(furit_prices,furit,sd)
  4，标准误
    stderr = function(x) sqrt(var(x)/length(x))
    tapply(furit_prices,furit,stderr)  
3.4 . 数组与矩阵  
  a = array(10:20,dim = c(2,5))
  数组转换为向量
  as.vector(x)
  matrix 矩阵 参数byrow是否按照行顺序分配元素
  对角矩阵 diag(a)
--------------------------------------------------------
3.4.2 最小二乘法
 函数：lsfit 最小二乘法拟合
 X: 一个矩阵的行对应的情况和其中列对应的变量
 Y: 结果，可以是一种矩阵
 Wt:可选参数，加权最小二乘法的执行权重向量
 Intercept: 
 如：y=2*x  x=c(1,2,3,4) y = c(2,4,6,8)
 lsfit(x,y)
3.4.3 交叉因子频率分析
  交叉因子频率分析用于分析数据的分布区间及其统计指标
  1:
  y = c(11,22,13,14,11,22,31,31,31,14)
  cut(y,5) -> cuty
  2: 使用table统计数据在每个区间出现的频率
  table(cuty)
  3: 使用hist生成分布直方图
    指定breaks参数（设定各区间的边界值）和axes参数 = F 表示手动刻画刻度
    将数据在table函数生成的区间内划分
  bins = seq(min(y),max(y),by = 4)
  hist(y,breaks = bins,col="lightblu",axes = FALSE)
  axis(1,bins)
  axis(2)
3.4.4 向量模长计算  


jiuye$行业名称[grepl("电子",jiuye$行业名称)] = jyhy
jiuye$行业劳动报酬[grepl("电子"，jiuye$行业名称)] = jygz

5.ch 统计分析基础

  概率分布，概率密度等
  指数分布，
  正太分布
  随机变量数字特征以及相关公式
5.3  回归分析：确定两种或者两种以上变量间相互依赖的定量关系
  理想：所有数据都落在了回归线上
  (1) 先建立散点图
    y = c(5,7,9,11,16,20)
    x = c(1,2,3,4,7,9)
    plot(x,y)      
  (2) 通过
    lsfit(x,y)
    数据带入，计算预测值和残差
    绘制散点图以及回归线
    abline(lsfit(x,y))
  (3)事实是，可以使用lm函数进行更详细的回分析
    lm(y~x) ->xy
    lm(y~x)
    summary(xy)
    Coefficients:
      Estimate：斜率与截距的估计值
      Std.   斜率与截距的估计标准差
      t value :斜率与截距的假设检验t值
      Pr(>|t|)  : 星号越多说明线性关系越明显
        Estimate   Std.      Error      t    value   Pr(>|t|)    
    (Intercept)  3.33803    0.16665   20.03 3.67e-05 ***
     x            1.84507    0.03227   57.17 5.60e-07 ***
5.3.2  多元线性回归
    多元线性回归可建立多个自变量和应变量之间的关系，起回归模型方程一般为：
      y= bx +b1x+b2x+.....+bkxk+e
    x2 =c(3,4,5,6,8,10)
    lm(y~x+x2) ->xy2
    summary(xy2)
5.3.3 非线性回归    
    非线性回归模型较多，其中应用得多的有一下几个模型
    1)多项式模型：
    y = B0+B1x+B2x^2+B3x^3 +   +++++
    2)指数模型 
     y = ae^bkx E
    3)幂指数模型
      y = ax1^b1 x2^b2 E
    4)成长曲线模型 
      y = 1/(B0 + B1E^-x+E)
    1)已知回归模型，
    x= c(1,2,3,4,7,8,9)
    y = 100+10*exp(x/2) + rnorm(B*x)
    2) 使用R语言的nls函数，应用最小二乘法原理，实现非线性回归分析
    nlmd = nls(y ~ Const + A * exp(B * x))
    3)使用summary(nlmod)函数分析你和结果
    4)绘制拟合效果图
    y= 100+10*exp(x/2)+rnorm(x)
    plot(x,y,main = "nls(o)")
    curve(100+10 * exp(x/2),col= 4，add=TRUE)
    lines(x,predict(nlmod),col=2)
    
    x=seq(1,10,0.1)
    y=100+10*exp(x/2)+rnorm(x)*100
    nlmod = nls(y ~  Const + A * exp(B * x))
    plot(x,y,main="nls(o)")
    curve(100+10*exp(x/2),col=4,add = T)
    lines(x,predict(nlmod),col=2)
    
5.4数据分析基础    
    将数据分成5个区间并且建立因子
  factor(cut(mag,5))
  绘制核密度计算
  hist(data,probability = T)
  lines(density(data))
5.5 按照年龄分类汇总肿瘤的数量
  attach(agedpatients)
  tapply(肿瘤,年龄,sum)
  tapply(急腹症,年龄,sum)
  分析年龄统计肿瘤患者的数量
  table(factor(cut(agepatients$年龄[agedpatients$肿瘤==1],5)))

6.ch 描述性分析案例
  jiuye$行业名称[grepl("电子",jiuye$行业名称)] -> jyhy
  jiuye$平均劳动报酬[grepl("电子",jiuye$行业名称)] -> jygz
  names(jygz) -> jyhy#这是什么
  barplot(jygz,horiz = T)
  pie(jygz)
6.2数据趋势案例解析
 cbind(jiuye[["平均劳动报酬"]],jiuye[["平均教育经费"]])
 apply(jiuyeinfo,2,mean)
 
 加权平均值 
 weighted.mean(cp$单机成本.元.台)
6.3 正太分布系数
  正态分布函数
  峰度系数分析
  mean(jiuye$平均劳动报酬) = mymean
  sd(jiuye$平均劳动报酬) = mysd
  length(jiuye$平均劳动报酬) = le
  jiuye$平均劳动报酬 = x
  ((myn * (myn +1))) /  ( ( myn-1)* (myn-2) * (myn -3)) * sum( (x-mymean) ^4 )/mysd^4 - 
    
6.3.4 概率密度函数一个连续性的随机变量的盖里密度函数
  计算，dnorm(变量,平均值,标准差) 求解正太分布概率密度函数,下面的代码计算产品产量的盖里密度
  mean(cp$产量.台) = mymean
  sd(cp$c产量.台. ) -> mysd
  length(cp$产量.台) = myn
  cp$产量.台. -> x
  dnorm(x,mymean,mysd)
  绘制散点图如图示：
  plot(x,dnorm(x,mymean,mysd))
  
  此外，rnorm()可以返回正太分布随机数，调用格式为:rnorm(长度,平均值,标准差)
  rnorm(50,0,1) -> rx
  plot(rx,dnorm(rx))
6.3.5 分位点
  下分位点
  fx = f(from = -oo,to = a)dx
  表示连续性随机变量X小于等于Za 的概率为a ，称Za为X的下a 分位点
  可用函数qnorm()
  qnorm(0.25,mean,sd) = nnn
  具体代码如下计算结果表明 变量小于nnn时的概率是。0.25
  
  上分位点 = 1-下分位点 的概率的对应点
  表示变量或者产量大于a 的概率
  fx = f(from = a,to = +oo)dx
  qnorm(1-0.25,mean,sd)
  
  绘图效果
  mymean = mean(cp$产量.)
  mysd = sd(cp$敞亮)
  x = cp$产量
  plot(x,dnorm(x,mymean,sd))
  abline(v = 6815)
6.3.7 概率密度与正太概率分布图
  1.概率密度与正态概率
  让概率密度和正太分布在一张图上显示出来，这样就能更好的看吹数据的分布情况，
  hist(jiuye[["平均劳动报酬"]]，freq = F)
  lines(density(jiuye[["平均劳动报酬"]]),col = "red")
  x<- c(0:ceiling(max(jiuye$平均劳动报酬)))
  lines(x,dnorm(x,mean(jiuye[["平均劳动报酬"]]),sd(jiuye[["平均劳动报酬"]]),col = "blue"))
  
  经验累积分布与正态分布  
  plot(ecdf(jiuye[["平均劳动报酬"]]),sd(jiuye[["平均劳动报酬"]])，col = "blue")  
  lines(x,pnorm(x,mean(jiuye[["平均劳动报酬"]])，sd(jiuye[["平均劳动报酬"]])),col = "blue")
  
  正态检验和分布拟合
  1.QQ图   可以预测数据分布是否近视某种类型分布，如果近似于正态分布，则数据点接近下面方程表示的直线
  
  y = ox+u   标准差，u平均数
  qqnorm(cp$产量)
  qqline(cp$产量)
  2.正态检验与分布拟合
    W检验，可以检验是否符合正态分布
  使用shapiro.test 进行正态W检验
  shapiro.test(cp$产量)
  
  
  
6.4 多变量分析
  多变量数据分析，多元数据分析
  使用 pairs(A)  绘制两列之间的散点图矩阵
    
  学生成绩协同图 多变量的探索性分析图形
  基本形式： coplot(y~x|z)  x,y 是数值型向量，Z是同长度的因子 ，对于Z的每一水平，均绘制相应的x和y的散点图，
  首先，设置因子
  sex = factor(df$性别,labels = c("女","男"))
  然后以性别来分组，绘制图形
  
  coplot(df$平时成绩~df$期末成绩|sex) 
  
  学生成绩点图----------------------
  首先设置因子------
  pf = factor(cut(df$平时成绩,5))
  dotchart(table(pf))
    
  
  产品销量三维图  
  需要安装scatterplot3d库
  #install.packages("scatterplot3")
  source("http://bioconductor.org/biocLite.R")
  biocLite("scatterplot3d")
    
  library(scatterplot3d)
  scatterplot3d(地区编码,月份,销量,highlight.3d = T,pch = 20,col.axis = "blue",col.grid = "lightblue",mian = "销售一览表",lab = c(4,12))
  
  最后分析一下1号地区的销售形势图
  
  subset(goods,地区编码 ==1) -> sdqu1
  nolfx = diqu1[2,3]
  
  plot(no1fx,type = "o",main = "1号地区形势")
  abline(h = mean(nolfx$销量))
  axis(4,mean(nolfx$销量))
  
  产品销量气泡图#2 --------
  气泡图是一个将点表示为圆圈的散点图
  #2  wef
  
  
  yue4 = subset(goods,地区编码==4,select.list(销量))
  row.names(mygoods) = c(1:4)
  
  
  
  
6.4.2  多元数据相关性分析
  1.皮尔森相关系数
  2.协方差 
  read.csv() = mygoods
  cov(mygoods) = my.cov  协方差矩阵
  cor(mygoods) = my.cor  相关系数矩阵
  
  调用 R cor.test进行检测，它采用 pearson检验，默认置信区间为0.95 (conf.level)   p值小于0.05则拒绝假设，并认为两变量线性相关
  cor.test(~A原料 + B原料,data = mygoods)#A原料与B原料
  
  
  2 影响因素分组
  读入数据并计算相关系数mysales = read.csv()
  cor(mysales)  #计算协相关系数
  然后对各个指标的相关度进行分析，使得相关系数高的指标归为同一组
  
7.假设检验与回归模型案例
  
  二项分布假设检验  
   原假设p>0.25,而在此测试中，p值为：2.2e-16，因此可接受该假设
  binom.test(x=20,n=300,p=0.25,alternative = "less")
  Exact binomial test
  
  data:  20 and 300
  number of successes = 20, number of trials = 300, p-value < 2.2e-16
  alternative hypothesis: true probability of success is less than 0.25
  95 percent confidence interval:
    0.00000000 0.09540198
  sample estimates:
    probability of success 
  0.06666667 
  
  
  游戏出产宝石检测   程序设定比率：3:11:23，实际上抽样671玩家的采掘情况 ： 70:190:411
  chisq.test(c(70,190,411),p=c(3,11,23)/37)
  Chi-squared test for given probabilities
  
  data:  c(70, 190, 411)
  X-squared = 5.0106, df = 2, p-value = 0.08165   结果p值大于0.。5 接受原假设，实际出产比率符合理论要求
7.12 数据分布检测
  使用 shapiro.test()函数可以检测数据非正态分布，例如，显著水平为0.05 检测以下数据是否为正态分布。
  x<- c(12,22,67,89,56,10,124,235,77,88,66,79,80,82)
  shapiro.test(x)
  结果p值小于0.05，因此拒绝假设，则该数据不符合正态分布
  
  
  
  
  
回归模型 
  研究一个随机变量Y对一个变量(X)n 或者一组变量X1,X2,X3,----Xk 的相依存关系的 统计方法，
  
  library(plotrix)
  pie3D(slices,labels=lbls,explode=0.1, main="Pie Chart of Countries ")
  pie3D(unapp$count,labels = unapp$appname,explode=0.3,col=rainbow(length(unapp$appname)),main="媒体作弊比率图")
  


第8章 机器学习算法
  神经网络  兴奋与抑制神经元模型
  感知机，自适应原件
  Rosentblatt 感知器，建立在一个线性神经元 ，沈锦元的求和节点计算作用于突触输入的线性组合
  两类线性可分的模式    ，则算法收敛，


$#-------------------------------------------------------------------
机器学习与R语言 -> ""
@author: zdx
@chapter:5  = 分而治之

#贵则算法：
  Ripper  oneR
library(RWeka)
m = OneR(class~predictor,data = mydata)  
p = predict(m,test)
m = JRip(class~predictor,data = mydata)  
p = predict(m,test)
library(C50)
C5.0(train,class,trails = 1,cost = NULL)
library(gmodels)
评估模型性能
CrossTable(credit_test$default,crediect_pred,prop.c = F,prop.c,prop.r = F,dnn = c("actual default","predict_default"))















a = Puromycin
pa = subset(a,state == "treated")
plot(rate ~ conc,data = pa)
with(a,plot(conc,rate))
with(a,plot(conc,rate,pch =2,col=4,cex=2.5,xlim = c(0,1.2),ylim = c(40,210)

x = pmin(3,pmax(-3,stats::rnorm(50)))
y = pmin(3,pmax(-3,stats::rnorm(50)))
xhist = hist(x,breaks = seq(-3,3,0.5),plot = FALSE)
yhist = hist(x,breaks = seq(-3,3,0.5),plot = FALSE)
top = max(c(xhist$counts,yhist$counts))
xrange = c(-3,3)
yrange = c(-3,3)
layout(matrix(c(2,0,1,3),2,2,byrow=T),c(3,1),c(1,3),T)
layout.show(3)
par(mar = c(3,3,1,1))
plot(x,y,xlim = xrange,ylim = yrange,xlab="",ylab = "")
par(mar = c(3,0,1,1))
barplot(xhist$counts,axes = F,ylim = c(0,top),space = 0)
par(mar = c(3,0,1,1))
barplot(yhist$counts,axes = F,xlim = c(0,top),space = 0,horiz = T)





我的意思是分组的同时执行这部分计算，通过该方式，能极大提高运行效率，否则还是写C语言吧


get_forecast_unit <- function(data){
  uni = group_by(data,DMU,Series,Display.Size,CPU.Family) %>%#按照PN分组
    summarise(n = n()) %>%#对每组数据分别统计数量
    filter(n>13) %>%#每组数据的总销售量统计，并且计算sellout.不能汇总，因为设计到历史数据的时间处理
    subset(grouped data)
  
  colnames(uni) <- c("DMU","Series","Display.Size","CPU")
  uni
}
