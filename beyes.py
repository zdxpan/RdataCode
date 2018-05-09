# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:02:48 2017

@author: zdx
贝叶斯分类器
"""
import pandas as pd
filename = 'C:/Users/zdx/Documents/Python Scripts/pima-indians-diabetes.csv'
dataset = pd.read_csv(filename,encoding =  'utf-8',engine='python') 
print(('Loaded data file {0} with {1} rows').format(filename, len(dataset)))

# Example of Naive Bayes implemented from Scratch in Python
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

dataset = [[1,20,1], [2,21,0], [3,22,1]]
separated = separateByClass(dataset)
import numpy as np
np.mean()



def summarize(dataset):
	summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries


dataset = [[1,20,0], [2,21,1], [3,22,0]]
summary = summarize(dataset)
print('Attribute summaries: {0}').format(summary)




def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
summary = summarizeByClass(dataset)
print('Summary by class value: {0}').format(summary)


import math
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities




{'a' :1,'b':2}.items()









