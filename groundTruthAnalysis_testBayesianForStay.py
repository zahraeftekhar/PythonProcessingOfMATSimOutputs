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
from groundTruthAnalysis_homeWorkDistribution import  tripDurations, tripStarts, activityDurations,activityStarts
from groundTruthAnalysis_homeWorkDistribution import  tripDurations, tripStarts, activityDurations,activityStarts
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')
# ____________________ selecting random activity sample as the training set
activityData = pd.DataFrame()
activityData['Duration(hour)'] = activityDurations['duration(sec)']/3600
activityData['start(hour)'] = activityStarts['start_time(sec)']/3600
random.seed(10)
train_activityIndex = random.sample(range(len(activityData)),round(.01*len(activityData))) #random number from uniform dist.
nactivity = len(train_activityIndex)
test_activityIndex = np.delete(range(len(activityData)),train_activityIndex)
train_activityData = activityData.loc[train_activityIndex]
test_activityData = activityData.loc[test_activityIndex]
# ____________________ extract KDE for activity duration
# Build PDF and turn into pandas Series
x1 = np.linspace(0, 24, 10000)
y1 = gaussian_kde(train_activityData['Duration(hour)']).pdf(x1)
pdf1 = pd.Series(y1, x1)
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(0, 24, step=1)))
ax1 = train_activityData['Duration(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
pdf1.plot(lw=2, label='Data', legend=True, ax=ax1)
ax1.set_title(u'activity duration . \n ,with Gaussian KDE')
ax1.set_xlabel(u'Duration (hour)')
ax1.set_ylabel('Frequency')
plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/activity_duration.png",dpi = 300)
plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/activity_duration.pdf",dpi = 300)
# ____________________ extract KDE for activity start time
# plt.figure()
# Build PDF and turn into pandas Series
x2 = np.linspace(0, 24, 10000)
y2 = gaussian_kde(train_activityData['start(hour)']).pdf(x2)
pdf2 = pd.Series(y2, x2)
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(2, 26, step=1)))
ax2 = train_activityData['start(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
pdf2.plot(lw=2, label='Data', legend=True, ax=ax2)
ax2.set_title(u'activity start time . \n ,with Gaussian KDE')
ax2.set_xlabel(u'Start Time (hour)')
ax2.set_ylabel('Frequency')
plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/activity_start.png",dpi = 300)
plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/activity_start.pdf",dpi = 300)
# ____________________ selecting random trip sample as the training set
tripData = pd.DataFrame()
tripData['Duration(hour)'] = tripDurations['duration(sec)']/3600
tripData['start(hour)'] = tripStarts['start_time(sec)']/3600
random.seed(10)
train_tripIndex = random.sample(range(len(tripData)),round(.01*len(tripData))) #random number from uniform dist.
ntrip = len(train_tripIndex)
test_tripIndex = np.delete(range(len(tripData)),train_tripIndex)
train_tripData = tripData.loc[train_tripIndex]
test_tripData = tripData.loc[test_tripIndex]
# ____________________ extract KDE for trip duration
# Build PDF and turn into pandas Series
x1 = np.linspace(0, 3, 1000)
y1 = gaussian_kde(train_tripData['Duration(hour)']).pdf(x1)
pdf1 = pd.Series(y1, x1)
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(0, 3, step=.5)))
ax1 = train_tripData['Duration(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
pdf1.plot(lw=2, label='Data', legend=True, ax=ax1)
ax1.set_title(u'trip duration . \n ,with Gaussian KDE')
ax1.set_xlabel(u'Duration (hour)')
ax1.set_ylabel('Frequency')
plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/activity_duration.png",dpi = 300)
plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/activity_duration.pdf",dpi = 300)
# ____________________ extract KDE for work start time
# plt.figure()
# Build PDF and turn into pandas Series
x2 = np.linspace(0, 24, 10000)
y2 = gaussian_kde(train_tripData['start(hour)']).pdf(x2)
pdf2 = pd.Series(y2, x2)
plt.figure(figsize=(12, 8))
plt.xticks((np.arange(2, 26, step=1)))
ax2 = train_tripData['start(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='Gaussian Kernel Density Estimation', legend=True)
pdf2.plot(lw=2, label='Data', legend=True, ax=ax2)
ax2.set_title(u'trip start time . \n ,with Gaussian KDE')
ax2.set_xlabel(u'Start Time (hour)')
ax2.set_ylabel('Frequency')
plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/trip_start.png",dpi = 300)
plt.savefig("D:/progress meeting/17June2020(Hans&Adam)/trip_start.pdf",dpi = 300)
#***************************************************************************
prior_activity = nactivity/(nactivity + ntrip)
prior_trip = ntrip/(nactivity + ntrip)
test_activityData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)']).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)']).pdf(test_activityData['start(hour)']))
test_activityData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)']).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)']).pdf(test_activityData['start(hour)']))
test_activityData['algorithm select activity'] = 0
p=0
for p in test_activityData.index:
    if (test_activityData['Log prob. of being activity']>=test_activityData['Log prob. of being trip'])[p] :
        test_activityData['algorithm select activity'][p] = 1
print(sum(test_activityData['algorithm select activity'])/len(test_activityData['algorithm select activity']))
#***************************************************************************
test_tripData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)']).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)']).pdf(test_tripData['start(hour)']))
test_tripData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)']).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)']).pdf(test_tripData['start(hour)']))
test_tripData['algorithm select trip'] = 0
p=0
for p in test_tripData.index:
    if (test_activityData['Log prob. of being activity']<=test_tripData['Log prob. of being trip'])[p] :
        test_tripData['algorithm select trip'][p] = 1
print(sum(test_tripData['algorithm select trip'])/len(test_tripData['algorithm select trip']))
#******************************** overall performance *******************************************
print((sum(test_activityData['algorithm select activity'])+sum(test_tripData['algorithm select trip']))/(len(test_activityData['algorithm select activity'])+len(test_tripData['algorithm select trip'])))
#************************* confusion matrix ******************************
from sklearn.metrics import confusion_matrix
test_activityData['observed'] = 'Log prob. of being activity' #1 is home
test_activityData['predicted'] = test_activityData[['Log prob. of being activity','Log prob. of being trip']].idxmax(axis=1)
test_tripData['observed'] = 'Log prob. of being trip' #2 is work
test_tripData['predicted'] = test_tripData[['Log prob. of being activity','Log prob. of being trip']].idxmax(axis=1)
y_actu = pd.concat([test_activityData['observed'],test_tripData['observed']],axis=0,ignore_index=True)
y_pred = pd.concat([test_activityData['predicted'],test_tripData['predicted']],axis=0,ignore_index=True)
colName = ['observed stay', 'observed pass-by']
rowName = ['predicted stay', 'predicted pass-by']
conMat = confusion_matrix(y_pred,y_actu) #left: actual and top: predicted
conMat = pd.DataFrame(conMat, columns=colName, index=rowName)