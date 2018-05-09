# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 14:59:06 2017

@author: zdx
"""

 
 '''
 我安装的xgboost 跟用python安装的有什么区别，：1：python setup.py install xgboost cd
 2:之前从网上下载一个xgboost 几十kb的 whl文件  xgboost-0.6-cp36-cp36m-win_amd64.whl
 我现在的版本是xgboost 0.6 不是2.0 这个有很大的问题
 
 
 
 UCI机器学习库的Mushroom 数据集 (XGBoost安装包中的demo数据)
 根据蘑菇的22个特征判断蘑菇是否有毒
 总样本数：8124
 Demo中22维特征经过处理，变成了126维特征量
 
 '''
 
import xgboost as xgb
from sklearn.metrics import accuracy_score
#读取数据 在xgboost安装路径下的demo目录 
 
my_workpath = 'C:/Users/zdx/xgboost/demo/data/'
dtrain = xgb.DMatrix(my_workpath+'agaricus.txt.train')
dtest = xgb.DMatrix(my_workpath+'agaricus.txt.test')
dtrain.num_col() 
dtrain.num_row()
dtest.num_row()
 '设置训练参数'
 
param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
 
'''
max_depth： 树的最大深度。缺省值为6，取值范围为：[1,∞]
• eta：为了防止过拟合，更新过程中用到的收缩步长。 eta通过缩减特征
的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
• silent: 0表示打印出运行时信息，取1时表示以缄默方式运行，不打印
运行时信息。缺省值为0
• objective： 定义学习任务及相应的学习目标，“binary:logistic” 表示
二分类的逻辑回归问题，输出为概率。


模型训练
'''
# 设置boosting迭代计算次
num_round = 2

import time
starttime = time.clock()

bst = xgb.train(param, dtrain, num_round)
endtime = time.clock()
print (endtime - starttime)
'''
'预测（训练数据上评估）'
模型训练好后，可以用训练好的模型对进行预测
– XGBoost预测的输出是概率，输出值是样本为第一类的概率 à 将概率值转换
为0或1
'''
train_preds = bst.predict(dtrain)
train_predictions = [round(value) for value in train_preds]
y_train = dtrain.get_label()
train_accuracy = accuracy_score(y_train, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
 
'''
番外：模型可视化

可视化模型中的单颗树：调用XGBoost 的API plot_tree()／
to_graphviz()


'''
from xgboost import plot_tree 
xgb.plot_tree(bst, num_trees=0, rankdir= 'UD' ) #将模型可视化
 '第二个参数为要打印的树的索引（从0开始）'
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
"""
---------------------------------------------------------------
0---------------------------------------------------------------
与scikit-learn结合
XGBoost提供一个wrapper
和scikit-learn框架中其他分类器或回归器一样
"""
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

#读取数据
'支持libsvm 格式数据--------稀疏特征 每行表示一个样本，第一个开头是样本标签'
'之后为特征索引，冒号‘：’ 后面为该特征的值'

X_train,y_train = load_svmlight_file(my_workpath + 'agaricus.txt.train')
X_test,y_test = load_svmlight_file(my_workpath + 'agaricus.txt.test')
print(X_train.shape)
#设置boosting迭代计算次数，
print(X_train.shape)
print (X_test.shape)
num_round = 2

bst = xgb.XGBClassifier(max_depth=2, learning_rate=0.1,n_estimators=num_round, silent=True,
objective='binary:logistic') 
 
# setup parameters for xgboost
param = {}
param['booster'] = 'gbtree'
param['objective'] = 'binary:logistic'
param["eval_metric"] = "error"
param['eta'] = 0.3
param['gamma'] = 0
param['max_depth'] = 6
param['min_child_weight']=1
param['max_delta_step'] = 0
param['subsample']= 1
param['colsample_bytree']=1
param['silent'] = 1
param['seed'] = 0
param['base_score'] = 0.5
 


"""
    测试结果-----------------#校验集  ----------->>>>>>>>>>>>>>>>>>>>>>>>
    
"""
# 设置boosting迭代计算次数


bst.fit(X_train, y_train)

train_preds = bst.predict(X_train)
train_predictions = [round(value) for value in train_preds]

train_accuracy = accuracy_score(y_train, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))


# make prediction
preds = bst.predict(X_test)
predictions = [round(value) for value in preds]

test_accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))





'''
----------------->>>>   XGBoost快速入门——与scikit-learn一起使用-split
'''


'意思是 在实际场合中，测试数据位置，如何评估模型？'
from sklearn.model_selection import train_test_split
#split data into train and test sets,1/3 的训练数据作为校验数据
seed = 7#设置随机数种子，以至于每次迭代的时候，划分数据集合，每一个子份是一样的

test_size = 0.33#划分测试集合占训练集合的比率

my_workpath ='C:/Users/zdx/xgboost/demo/data/'
X_train,y_train = load_svmlight_file(my_workpath + 'agaricus.txt.train')
X_test,y_test = load_svmlight_file(my_workpath + 'agaricus.txt.test')

X_train.shape
#X_test.shape

X_train_part, X_validate, y_train_part, y_validate = train_test_split(X_train, y_train, test_size=test_size,
    random_state=seed)

X_train_part.shape

# 设置boosting迭代计算次数
num_round = 2

#bst = XGBClassifier(param)
#bst = XGBClassifier()
bst =XGBClassifier(max_depth=2, learning_rate=1, n_estimators=num_round, silent=True, objective='binary:logistic')

bst.fit(X_train_part, y_train_part)


validare_preds = bst.predict(X_validate)
validate_predictions = [round(value) for value in validare_preds]

train_accuracy = accuracy_score(y_validate, validate_predictions)
print ("Validation Accuary: %.2f%%" % (train_accuracy * 100.0))

"""
----------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

学习曲线：模型预测性能随某个变化的学习参数
（如训练样本数目、迭代次数）变化的情况
– 例：XGBoost的迭代次数（树的数目）
#设置boosting 迭代计算次数
"""

num_round = 100

bst = xgb.XGBClassifier(max_depth=2, learning_rate=0.1,n_estimators=num_round, silent=True,
objective='binary:logistic') 
 

'设置评估集'
eval_set = [ ( X_train_part,y_train_part), (X_validate,y_validate)  ]

bst.fit(X_train_part,y_train_part,eval_metric= ["error","logloss" ],\
        eval_set = eval_set,verbose = True )


"""
--------查看模型在训练集上的分类性能
XGBoost 预测的输出是概率，这里的分类是一个而分类问题，
"""


"""
模型每次校验集上的性能存在模型中，可用来进一步分析 model.evals result() 返回一个字典
：评估数据寄和分数

显示学习曲线

"""
#retrieve performance metrisc
results = bst.evals_result()
#print(results)
epochs  = len(results['validation_0']['error'])
x_axis = range(0,epochs)

#plot log loss
fig,ax = pyplot.subplots()
ax.plot(x_axis,results['validation_0']['logloss'],label='Train')
ax.plot(x_axis,results['validation_1']['logloss'],label='Test')
ax.legend()
pyplot.ylabel("Log Loss")
pyplot.title("XGBoost Log Loss")
pyplot.show()

#plot classification error
fig,ax = pyplot.subplots()
ax.plot(x_axis,results['validation_0']['error'],label='Train')
ax.plot(x_axis,results['validation_1']['error'],label='Test')
ax.legend()
pyplot.ylabel("Classification Error")
pyplot.title("XGBoost Classification Error")
pyplot.show()

# make prediction
preds = bst.predict(X_test)
predictions = [round(value) for value in preds]

test_accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))

"""
 ___________new way of sutitle_______________________>>>>>>>>>>>>>>>>>>>>>>
 
         Early stop: 一种防止训练复杂模型过拟合的方法

– 监控模型在校验集上的性能：如果在经过固定次数的迭代，校验集上的性能
不再提高时，结束训练过程
– 当在测试集上的训练下降而在训练集上的性能还提高时，发生了过拟合
•使用准则  val_metric="error" 查看错误率
"""

# 运行 xgboost安装包中的示例程序
import xgboost as xgb
from xgboost import XGBClassifier

# 加载LibSVM格式数据模块
from sklearn.datasets import load_svmlight_file

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot
#观察接下来的10轮如果性能没有提升那么结束  
seed = 7
test_size = 0.33
X_train_part, X_validate, y_train_part, y_validate= train_test_split(X_train, y_train, test_size=test_size,
    random_state=seed)

X_train_part.shape
X_validate.shape
 """训练参数设置"
 max_depth:树的最大深度； eta 为了防止过拟合，更新过程中用到的收缩步长
 objective 定义学习任务以及相应的学习目标
 binary：logistic 表示而分类的逻辑回归问题
 """
param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
num_round = 100 # 设置boosting迭代计算次数
param = {'max_depth':2,'eta':1,'slient':0,'objective':'binary:logistic'}
 
bst = XGBClassifier(max_depth=2, learning_rate= 0.1 \
                        ,n_estimators=num_round, silent=True, objective="binary:logistic")

eval_set =[(X_validate, y_validate)]
bst.fit(X_train_part, y_train_part, early_stopping_rounds=10, eval_metric="error",
    eval_set=eval_set, verbose=True)


#retrieve performance metrisc----------显示学习曲线
results = bst.evals_result()
#print(results)
epochs  = len(results['validation_0']['error'])
x_axis = range(0,epochs)
 
#plot classification error
fig,ax = pyplot.subplots()
ax.plot(x_axis,results['validation_0']['error'],label='Test')
ax.legend()
pyplot.ylabel("Classification Error")
pyplot.xlabel("Round")
pyplot.title("XGBoost earlly stop")
pyplot.show()


#X测试

# make prediction
preds = bst.predict(X_test)
predictions = [round(value) for value in preds]

test_accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))

"""
--------------------------->>>>>>>>>>>>>>>>>>----------->>>>>>>>>>>>>>

k-折交叉验证：将训练数据等分成k份（k通常的取值为3、 5或10）
– 重复k次
• 每次留出一份做校验，其余k-1份做训练
– k次校验集上的平均性能视为模型在测试集上性能的估计
• 该估计比train_test_split得到的估计方差更小
如果每类样本不均衡或类别数较多，采用StratifiedKFold， 将数据集中每一类样本
的数据等分
"""

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score #对给定参数的单个模型评估
# 设置boosting迭代计算次数
num_round = 2
#num_round = range(1, 101)
#param_grid = dict(n_estimators=num_round)

#bst = XGBClassifier(param)
bst =XGBClassifier(max_depth=2, learning_rate=0.1,n_estimators=num_round, 
                   silent=True, objective='binary:logistic')
#交叉验证
kfold = StratifiedKFold(n_splits = 10,random_state=7)#防止样本不均衡的方法

results = cross_val_score(bst,X_train,y_train,cv = kfold)


kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(bst, X_train, y_train, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))

 
"""
_____----------------->>>>>>>>>>>>>-----------------------------
---------------------参 数 调 整 -----------选择

参数调优GridSearchCV ：我们可以根据交叉验证评估的结
果，选择最佳参数的模型

– 输入待调节参数的范围（grid），对一组参数对应的模型进行评估，
并给出最佳模型及其参数

-----------------网格搜索 --------
"""


from sklearn.grid_search import GridSearchCV
# 设置boosting迭代计算次数搜索范围
'  树的棵数
param_test = { #弱分类器的数目以及范围
        'n_estimators':list(range(1, 51, 1))
        }
#以下参数分别为      模型             评价参数范围                评估分数：准确率    交叉验证折数
clf = GridSearchCV(estimator = bst, param_grid = param_test, scoring = 'accuracy', cv=5)
clf.fit(X_train, y_train)
'模型最佳分数，最佳参数，最佳分数
'  sholud be 50 variables
clf.grid_scores_, clf.best_params_, clf.best_score_

"""
        测试 make prediction
"""
preds = clf.predict(X_test)
predictions = [round(value) for value in preds]


test_accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy of gridsearchcv: %.2f%%" % (test_accuracy * 100.0))





