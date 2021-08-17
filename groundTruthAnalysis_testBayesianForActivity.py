import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from scipy.stats import gamma
import numpy as np
import math
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import random
# __________________________________________________________
from _4_groundTruthAnalysis_locationActivityDistribution import homeDurations, homeStarts, workDurations, workStarts, \
     otherDurations, otherStarts, tripDurations, tripStarts, activityDurations,activityStarts
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')
# ____________________ selecting random work sample as the training set
workData = pd.DataFrame()
workData['Duration(hour)'] = workDurations['duration(sec)']/3600
workData['start(hour)'] = workStarts['start_time(sec)']/3600
random.seed(10)
train_workIndex = random.sample(range(len(workData)),round(.01*len(workData))) #random number from uniform dist.
nWork = len(train_workIndex)
test_workIndex = np.delete(range(len(workData)),train_workIndex)
train_workData = workData.loc[train_workIndex]
test_workData = workData.loc[test_workIndex]
# # ____________________ extract KDE for work duration
# # Build PDF and turn into pandas Series
# x1 = np.linspace(0, 24, 10000)
# y1 = gaussian_kde(train_workData['Duration(hour)']).pdf(x1)
# pdf1 = pd.Series(y1, x1)
# plt.figure(figsize=(12, 8))
# plt.xticks((np.arange(0, 24, step=1)))
# ax1 = train_workData['Duration(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
# pdf1.plot(lw=2, label='Data', legend=True, ax=ax1)
# ax1.set_title(u'work duration . \n ,with Gaussian KDE')
# ax1.set_xlabel(u'Duration (hour)')
# ax1.set_ylabel('Frequency')
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/work_duration.png",dpi = 300)
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/work_duration.pdf",dpi = 300)
# # ____________________ extract KDE for work start time
# # plt.figure()
# # Build PDF and turn into pandas Series
# x2 = np.linspace(2, 26, 10000)
# y2 = gaussian_kde(train_workData['start(hour)']).pdf(x2)
# pdf2 = pd.Series(y2, x2)
# plt.figure(figsize=(12, 8))
# plt.xticks((np.arange(2, 26, step=1)))
# ax2 = train_workData['start(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
# pdf2.plot(lw=2, label='Data', legend=True, ax=ax2)
# ax2.set_title(u'work start time . \n ,with Gaussian KDE')
# ax2.set_xlabel(u'Start Time (hour)')
# ax2.set_ylabel('Frequency')
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/work_start.png",dpi = 300)
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/work_start.pdf",dpi = 300)

# ____________________ selecting random home sample as the training set
homeData = pd.DataFrame()
homeData['Duration(hour)'] = (homeDurations['duration(sec)'])/3600
homeData['start(hour)'] = (homeStarts['start_time(sec)'])/3600
random.seed(10)
train_homeIndex = random.sample(range(len(homeData)),round(.01*len(homeData))) #random number from uniform dist.
nHome = len(train_homeIndex)
test_homeIndex = np.delete(range(len(homeData)),train_homeIndex)
train_homeData = homeData.loc[train_homeIndex]
test_homeData = homeData.loc[test_homeIndex]
# ____________________ extract KDE for home duration
# Build PDF and turn into pandas Series
# x1 = np.linspace(0, 24, 10000)
# y1 = gaussian_kde(train_homeData['Duration(hour)']).pdf(x1)
# pdf1 = pd.Series(y1, x1)
# plt.figure(figsize=(12, 8))
# plt.xticks((np.arange(0, 24, step=1)))
# ax1 = train_homeData['Duration(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
# pdf1.plot(lw=2, label='Data', legend=True, ax=ax1)
# ax1.set_title(u'home duration . \n ,with Gaussian KDE')
# ax1.set_xlabel(u'Duration (hour)')
# ax1.set_ylabel('Frequency')
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/home_duration.png",dpi = 300)
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/home_duration.pdf",dpi = 300)
# # ____________________ extract KDE for home start time
# # Build PDF and turn into pandas Series
# x2 = np.linspace(2, 26, 10000)
# y2 = gaussian_kde(train_homeData['start(hour)']).pdf(x2)
# pdf2 = pd.Series(y2, x2)
# plt.figure(figsize=(12, 8))
# plt.xticks((np.arange(2, 26, step=1)))
# ax2 = train_homeData['start(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
# pdf2.plot(lw=2, label='Data', legend=True, ax=ax2)
# ax2.set_title(u'home start time . \n ,with Gaussian KDE')
# ax2.set_xlabel(u'Start Time (hour)')
# ax2.set_ylabel('Frequency')
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/home_start.png",dpi = 300)
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/home_start.pdf",dpi = 300)
# ____________________ selecting random other sample as the training set
otherData = pd.DataFrame()
otherData['Duration(hour)'] = otherDurations['duration(sec)']/3600
otherData['start(hour)'] = otherStarts['start_time(sec)']/3600
random.seed(10)
train_otherIndex = random.sample(range(len(otherData)),round(.01*len(otherData))) #random number from uniform dist.
nOther = len(train_otherIndex)
test_otherIndex = np.delete(range(len(otherData)),train_otherIndex)
train_otherData = otherData.loc[train_otherIndex]
test_otherData = otherData.loc[test_otherIndex]
# ____________________ extract KDE for other duration
# Build PDF and turn into pandas Series
# x1 = np.linspace(0, 24, 10000)
# y1 = gaussian_kde(train_otherData['Duration(hour)']).pdf(x1)
# pdf1 = pd.Series(y1, x1)
# plt.figure(figsize=(12, 8))
# plt.xticks((np.arange(0, 24, step=1)))
# ax1 = train_otherData['Duration(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
# pdf1.plot(lw=2, label='Data', legend=True, ax=ax1)
# plt.ylim(0,0.6)
# ax1.set_title(u'other duration . \n ,with Gaussian KDE')
# ax1.set_xlabel(u'Duration (hour)')
# ax1.set_ylabel('Frequency')
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/other_duration.png",dpi = 300)
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/other_duration.pdf",dpi = 300)
# # ____________________ extract KDE for other start time
# # plt.figure()
# # Build PDF and turn into pandas Series
# x2 = np.linspace(2, 26, 10000)
# y2 = gaussian_kde(train_otherData['start(hour)']).pdf(x2)
# pdf2 = pd.Series(y2, x2)
# plt.figure(figsize=(12, 8))
# plt.xticks((np.arange(2, 26, step=1)))
# ax2 = train_otherData['start(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
# pdf2.plot(lw=2, label='Data', legend=True, ax=ax2)
# ax2.set_title(u'other start time . \n ,with Gaussian KDE')
# ax2.set_xlabel(u'Start Time (hour)')
# ax2.set_ylabel('Frequency')
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/other_start.png",dpi = 300)
# plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/other_start.pdf",dpi = 300)
#***************************************************************************
prior_home = nHome/(nHome + nWork + nOther)
prior_work = nWork/(nHome + nWork + nOther)
prior_other = nOther/(nHome + nWork + nOther)
test_homeData['Log prob. of being home'] = np.log(prior_home)+np.log( gaussian_kde(train_homeData['Duration(hour)']).pdf(test_homeData['Duration(hour)']))+np.log( gaussian_kde(train_homeData['start(hour)']).pdf(test_homeData['start(hour)']))
test_homeData['Log prob. of being work'] = np.log(prior_work)+np.log( gaussian_kde(train_workData['Duration(hour)']).pdf(test_homeData['Duration(hour)']))+np.log( gaussian_kde(train_workData['start(hour)']).pdf(test_homeData['start(hour)']))
test_homeData['Log prob. of being other'] = np.log(prior_other)+np.log( gaussian_kde(train_otherData['Duration(hour)']).pdf(test_homeData['Duration(hour)']))+np.log( gaussian_kde(train_otherData['start(hour)']).pdf(test_homeData['start(hour)']))
test_homeData['algorithm select home'] = 0
p=0
for p in test_homeData.index:
    if (test_homeData['Log prob. of being home']>=test_homeData['Log prob. of being work'])[p] and (test_homeData['Log prob. of being home']>=test_homeData['Log prob. of being other'])[p]:
        test_homeData['algorithm select home'][p] = 1
print(sum(test_homeData['algorithm select home'])/len(test_homeData['algorithm select home']))
#***************************************************************************
test_workData['Log prob. of being home'] = np.log(prior_home)+np.log( gaussian_kde(train_homeData['Duration(hour)']).pdf(test_workData['Duration(hour)']))+np.log( gaussian_kde(train_homeData['start(hour)']).pdf(test_workData['start(hour)']))
test_workData['Log prob. of being work'] = np.log(prior_work)+np.log( gaussian_kde(train_workData['Duration(hour)']).pdf(test_workData['Duration(hour)']))+np.log( gaussian_kde(train_workData['start(hour)']).pdf(test_workData['start(hour)']))
test_workData['Log prob. of being other'] = np.log(prior_other)+np.log( gaussian_kde(train_otherData['Duration(hour)']).pdf(test_workData['Duration(hour)']))+np.log( gaussian_kde(train_otherData['start(hour)']).pdf(test_workData['start(hour)']))
test_workData['algorithm select work'] = 0
for p in test_workData.index:
    if (test_workData['Log prob. of being work'] >=test_workData['Log prob. of being home'])[p] and (test_workData['Log prob. of being work'] >=test_workData['Log prob. of being other'])[p]:
        test_workData['algorithm select work'][p] = 1
# test_workData['algorithm select work'] = test_workData['Log prob. of being work'] >=test_workData['Log prob. of being home'] and test_workData['Log prob. of being work'] >=test_workData['Log prob. of being other']
print(sum(test_workData['algorithm select work'])/len(test_workData['algorithm select work']))
#***************************************************************************
test_otherData['Log prob. of being home'] = np.log(prior_home)+np.log( gaussian_kde(train_homeData['Duration(hour)']).pdf(test_otherData['Duration(hour)']))+np.log( gaussian_kde(train_homeData['start(hour)']).pdf(test_otherData['start(hour)']))
test_otherData['Log prob. of being work'] = np.log(prior_work)+np.log( gaussian_kde(train_workData['Duration(hour)']).pdf(test_otherData['Duration(hour)']))+np.log( gaussian_kde(train_workData['start(hour)']).pdf(test_otherData['start(hour)']))
test_otherData['Log prob. of being other'] = np.log(prior_other)+np.log( gaussian_kde(train_otherData['Duration(hour)']).pdf(test_otherData['Duration(hour)']))+np.log( gaussian_kde(train_otherData['start(hour)']).pdf(test_otherData['start(hour)']))
test_otherData['algorithm select other'] = 0
for p in test_otherData.index:
    if (test_otherData['Log prob. of being other']>=test_otherData['Log prob. of being work'])[p] and (test_otherData['Log prob. of being other']>=test_otherData['Log prob. of being home'])[p]:
        test_otherData['algorithm select other'][p] = 1
# test_otherData['Log prob. of being other']>=test_otherData['Log prob. of being work'] and test_otherData['Log prob. of being other']>=test_otherData['Log prob. of being home']
print(sum(test_otherData['algorithm select other'])/len(test_otherData['algorithm select other']))
print((sum(test_otherData['algorithm select other'])+sum(test_workData['algorithm select work'])+sum(test_homeData['algorithm select home']))/(len(test_homeData['algorithm select home'])+len(test_workData['algorithm select work'])+len(test_otherData['algorithm select other'])))

#************************* confusion matrix ******************************
from sklearn.metrics import confusion_matrix
test_homeData['observed activity'] = 'Log prob. of being home' #1 is home
test_homeData['predicted activity'] = test_homeData[['Log prob. of being home','Log prob. of being work','Log prob. of being other']].idxmax(axis=1)
test_workData['observed activity'] = 'Log prob. of being work' #2 is work
test_workData['predicted activity'] = test_workData[['Log prob. of being home','Log prob. of being work','Log prob. of being other']].idxmax(axis=1)
test_otherData['observed activity'] = 'Log prob. of being other' #3 is other
test_otherData['predicted activity'] = test_otherData[['Log prob. of being home','Log prob. of being work','Log prob. of being other']].idxmax(axis=1)
y_actu = pd.concat([test_homeData['observed activity'],test_workData['observed activity'],test_otherData['observed activity']],axis=0,ignore_index=True)
y_pred = pd.concat([test_homeData['predicted activity'],test_workData['predicted activity'],test_otherData['predicted activity']],axis=0,ignore_index=True)
colName = ['observed home', 'observed work', 'observed other']
rowName = ['predicted home', 'predicted work', 'predicted other']
conMat = confusion_matrix(y_pred,y_actu) #left: actual and top: predicted
conMat = pd.DataFrame(conMat, columns=colName, index=rowName)
import openpyxl
conMat.to_excel('D:/progress meeting/17June2020(Hans&Adam)/confusion_matrix_activityType.xlsx',header=True,index=True, engine='openpyxl')
