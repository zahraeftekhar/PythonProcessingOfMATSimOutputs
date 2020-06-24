# you should also have a database of long-lat of each record of duration and startTime.
# Suggestion: construct a table for data whose columns are: duration, startTime, long, and lat.
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
from groundTruthAnalysis_testBayesianForActivity import train_otherData, train_homeData, train_workData,nOther,nWork,nHome
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')
# ____________________ selecting random activity sample as the training set
activityData = pd.DataFrame()
activityData['Duration(hour)'] = activityDurations['duration(sec)']/3600
activityData['start(hour)'] = activityStarts['start_time(sec)']/3600
activityData['type'] = activityStarts['type']
random.seed(10)
train_activityIndex = random.sample(range(len(activityData)),round(.01*len(activityData))) #random number from uniform dist.
nactivity = len(train_activityIndex)
test_activityIndex = np.delete(range(len(activityData)),train_activityIndex)
train_activityData = activityData.loc[train_activityIndex]
test_activityData = activityData.loc[test_activityIndex]
# ____________________ selecting random trip sample as the training set
tripData = pd.DataFrame()
tripData['Duration(hour)'] = tripDurations['duration(sec)']/3600
tripData['start(hour)'] = tripStarts['start_time(sec)']/3600
# tripData['type'] = None
random.seed(10)
train_tripIndex = random.sample(range(len(tripData)),round(.01*len(tripData))) #random number from uniform dist.
ntrip = len(train_tripIndex)
test_tripIndex = np.delete(range(len(tripData)),train_tripIndex)
train_tripData = tripData.loc[train_tripIndex]
test_tripData = tripData.loc[test_tripIndex]
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
# print(sum(test_activityData['algorithm select activity'])/len(test_activityData['algorithm select activity']))
#***************************************************************************
test_tripData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)']).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)']).pdf(test_tripData['start(hour)']))
test_tripData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)']).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)']).pdf(test_tripData['start(hour)']))
test_tripData['algorithm select trip'] = 0
p=0
for p in test_tripData.index:
    if (test_activityData['Log prob. of being activity']<=test_tripData['Log prob. of being trip'])[p] :
        test_tripData['algorithm select trip'][p] = 1
# print(sum(test_tripData['algorithm select trip'])/len(test_tripData['algorithm select trip']))
#***************************************************************************
test_data = test_activityData[ test_activityData['algorithm select activity']>0]
prior_home = nHome/(nHome + nWork + nOther)
prior_work = nWork/(nHome + nWork + nOther)
prior_other = nOther/(nHome + nWork + nOther)
test_data['Log prob. of being home'] = np.log(prior_home)+np.log( gaussian_kde(train_homeData['Duration(hour)']).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_homeData['start(hour)']).pdf(test_data['start(hour)']))
test_data['Log prob. of being work'] = np.log(prior_work)+np.log( gaussian_kde(train_workData['Duration(hour)']).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_workData['start(hour)']).pdf(test_data['start(hour)']))
test_data['Log prob. of being other'] = np.log(prior_other)+np.log( gaussian_kde(train_otherData['Duration(hour)']).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_otherData['start(hour)']).pdf(test_data['start(hour)']))
# p=0
# for p in test_data.index:
#     if (test_data['Log prob. of being home']>=test_data['Log prob. of being work'])[p] and (test_data['Log prob. of being home']>=test_data['Log prob. of being other'])[p]:
#         test_data['algorithm select home'][p] = 1
# print(sum(test_data['algorithm select home'])/len(test_data['algorithm select home']))
#************************* confusion matrix ******************************
from sklearn.metrics import confusion_matrix
test_data['observed activity'] = 'Log prob. of being '+test_data['type'].astype(str) ####todo*****!!!!!!!||||||||####### does it work??
test_data['predicted activity'] = test_data[['Log prob. of being home','Log prob. of being work','Log prob. of being other']].idxmax(axis=1)
y_actu = test_data['observed activity']
y_pred = test_data['predicted activity']
colName = ['observed home', 'observed work', 'observed other']
rowName = ['predicted home', 'predicted work', 'predicted other']
conMat = confusion_matrix(y_pred,y_actu) #left: actual and top: predicted
conMat = pd.DataFrame(conMat, columns=colName, index=rowName)
#the algorithm works correctly in the following percentage of the test set.
print((pd.value_counts(test_data['observed activity']==test_data['predicted activity'])[1]+sum(test_tripData['algorithm select trip']))/(len(test_data)+len(test_tripData)))
import openpyxl
conMat.to_excel('D:/progress meeting/17June2020(Hans&Adam)/confusion_matrix_doubleBayesian_activityType.xlsx',header=True,index=True, engine='openpyxl')