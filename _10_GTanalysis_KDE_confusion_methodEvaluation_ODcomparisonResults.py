import pandas as pd
from scipy.stats import gaussian_kde
import time
import numpy as np
import matplotlib.pyplot as plt
seed = 101
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')
# ************************start: KDE application plots on GT *****************************
# ______________ importing required data:training set _________________________
train_tripData = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_tripStarts_seed{ss}.csv".format(
        ss=seed))
train_activityData = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_activityStarts_seed{ss}.csv".format(
        ss=seed))
train_homeData = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_homeStarts_seed{ss}.CSV".format(
        ss=seed))
train_workData = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_workStarts_seed{ss}.CSV".format(
        ss=seed))
train_otherData = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/train_otherStarts_seed{ss}.CSV".format(
        ss=seed))
# ______________ importing required data:test set _________________________
tripStarts = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.tripStarts.csv")
activityStarts = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.activityStarts.csv")
tripDurations = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.tripDurations.CSV")
activityDurations = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.activityDurations.CSV")
homeStarts = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.homeStarts.CSV")
workStarts = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.workStarts.CSV")
otherStarts = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.otherStarts.CSV")
homeDurations = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.homeDurations.CSV")
workDurations = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.workDurations.CSV")
otherDurations = pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.otherDurations.CSV")
# ________________________________________________________________
# ____________________ extract KDE for activity duration
# Build PDF and turn into pandas Series
x1 = np.linspace(0, 24, 10000)
y1 = gaussian_kde(train_activityData['Duration(hour)']).pdf(x1)
pdf1 = pd.Series(y1, x1)
plt.figure(figsize=(6, 4.5))

plt.xticks((np.arange(0, 24, step=4)),fontsize=11)
plt.yticks((np.arange(0, 0.2, step=0.05)),fontsize=11)
ax1 = train_activityData['Duration(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='test set', legend=True)
pdf1.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax1)
ax1.set_title(u'activity duration.',fontsize=16)
ax1.set_xlabel(u'Duration (hour)',fontsize=14)
ax1.set_ylabel('Frequency',fontsize=14)
plt.legend(['Gaussian Kernel density estimation','test set'],fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/activity_duration.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/activity_duration.pdf",dpi = 300)
# ____________________ extract KDE for activity start time
# plt.figure()
# Build PDF and turn into pandas Series
x2 = np.linspace(0, 24, 10000)
y2 = gaussian_kde(train_activityData['start(hour)']).pdf(x2)
pdf2 = pd.Series(y2, x2)
plt.figure(figsize=(6, 4.5))
plt.xticks((np.arange(2, 26, step=4)),fontsize=11)
plt.yticks(fontsize=11)
ax2 = train_activityData['start(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='test set', legend=True)
pdf2.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax2)
ax2.set_title(u'activity start time.',fontsize=16)
ax2.set_xlabel(u'Start Time (hour)',fontsize=14)
ax2.set_ylabel('Frequency',fontsize=14)
plt.legend(fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/activity_start.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/activity_start.pdf",dpi = 300)
# ____________________ extract KDE for trip duration
# Build PDF and turn into pandas Series
x1 = np.linspace(0, 3, 1000)
y1 = gaussian_kde(train_tripData['Duration(hour)']).pdf(x1)
pdf1 = pd.Series(y1, x1)
plt.figure(figsize=(6, 4.5))
plt.xticks((np.arange(0, 3, step=.5)),fontsize=11)
ax1 = train_tripData['Duration(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='test set', legend=True)
pdf1.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax1)
plt.yticks(fontsize=11)
ax1.set_title(u'trip duration.',fontsize=16)
ax1.set_xlabel(u'Duration (hour)',fontsize=14)
ax1.set_ylabel('Frequency',fontsize=14)
plt.legend(fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/trip_duration.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/trip_duration.pdf",dpi = 300)
# ____________________ extract KDE for trip start time
# plt.figure()
# Build PDF and turn into pandas Series
x2 = np.linspace(0, 24, 10000)
y2 = gaussian_kde(train_tripData['start(hour)']).pdf(x2)
pdf2 = pd.Series(y2, x2)
plt.figure(figsize=(6, 4.5))
plt.xticks((np.arange(2, 26, step=4)),fontsize=11)
plt.yticks(fontsize=11)
ax2 = train_tripData['start(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='test set', legend=True)
pdf2.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax2)
ax2.set_title(u'trip start time.',fontsize=16)
ax2.set_xlabel(u'Start Time (hour)',fontsize=14)
ax2.set_ylabel('Frequency',fontsize=14)
plt.legend(fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/trip_start.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/trip_start.pdf",dpi = 300)
# ____________________ extract KDE for work duration
# Build PDF and turn into pandas Series
x1 = np.linspace(0, 24, 10000)
y1 = gaussian_kde(train_workData['Duration(hour)']).pdf(x1)
pdf1 = pd.Series(y1, x1)
plt.figure(figsize=(6, 4.5))
plt.xticks((np.arange(0, 24, step=4)),fontsize=11)
plt.yticks(fontsize=11)
ax1 = train_workData['Duration(hour)'].plot(kind='hist', bins=30, density=True, alpha=0.5, label='test set', legend=True)
pdf1.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax1)
ax1.set_title(u'work duration.',fontsize=16)
ax1.set_xlabel(u'Duration (hour)',fontsize=14)
ax1.set_ylabel('Frequency',fontsize=14)
plt.legend(fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/work_duration.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/work_duration.pdf",dpi = 300)
# ____________________ extract KDE for work start time
# plt.figure()
# Build PDF and turn into pandas Series
x2 = np.linspace(2, 26, 10000)
y2 = gaussian_kde(train_workData['start(hour)']).pdf(x2)
pdf2 = pd.Series(y2, x2)
plt.figure(figsize=(6, 4.5))
plt.xticks((np.arange(2, 26, step=4)),fontsize=11)
plt.yticks(fontsize=11)
ax2 = train_workData['start(hour)'].plot(kind='hist', bins=30, density=True, alpha=0.5, label='test set', legend=True)
pdf2.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax2)
ax2.set_title(u'work start time.',fontsize=16)
ax2.set_xlabel(u'Start Time (hour)',fontsize=14)
ax2.set_ylabel('Frequency',fontsize=14)
plt.legend(fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/work_start.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/work_start.pdf",dpi = 300)
# ____________________ extract KDE for home duration
# Build PDF and turn into pandas Series
x1 = np.linspace(0, 24, 10000)
y1 = gaussian_kde(train_homeData['Duration(hour)']).pdf(x1)
pdf1 = pd.Series(y1, x1)
plt.figure(figsize=(6, 4.5))
plt.xticks((np.arange(0, 24, step=4)),fontsize=11)
plt.yticks(fontsize=11)
ax1 = train_homeData['Duration(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='test set', legend=True)
pdf1.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax1)
ax1.set_title(u'home duration.',fontsize=16)
ax1.set_xlabel(u'Duration (hour)',fontsize=14)
ax1.set_ylabel('Frequency',fontsize=14)
plt.legend(fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/home_duration.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/home_duration.pdf",dpi = 300)
# ____________________ extract KDE for home start time
# Build PDF and turn into pandas Series
x2 = np.linspace(2, 26, 10000)
y2 = gaussian_kde(train_homeData['start(hour)']).pdf(x2)
pdf2 = pd.Series(y2, x2)
plt.figure(figsize=(6, 4.5))
plt.xticks((np.arange(2, 26, step=4)),fontsize=11)
plt.yticks(fontsize=11)
ax2 = train_homeData['start(hour)'].plot(kind='hist', bins=70, density=True, alpha=0.5, label='test set', legend=True)
pdf2.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax2)
ax2.set_title(u'home start time.',fontsize=16)
ax2.set_xlabel(u'Start Time (hour)',fontsize=14)
ax2.set_ylabel('Frequency',fontsize=11)
plt.legend(fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/home_start.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/home_start.pdf",dpi = 300)
# ____________________ extract KDE for other duration
# Build PDF and turn into pandas Series
x1 = np.linspace(0, 24, 10000)
y1 = gaussian_kde(train_otherData['Duration(hour)']).pdf(x1)
pdf1 = pd.Series(y1, x1)
plt.figure(figsize=(6, 4.5))
plt.xticks((np.arange(0, 24, step=4)),fontsize=11)
plt.yticks(fontsize=11)
ax1 = train_otherData['Duration(hour)'].plot(kind='hist', bins=30, density=True, alpha=0.5, label='test set', legend=True)
pdf1.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax1)
plt.ylim(0,0.6)
ax1.set_title(u'other duration.',fontsize=16)
ax1.set_xlabel(u'Duration (hour)',fontsize=14)
ax1.set_ylabel('Frequency',fontsize=14)
plt.legend(fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/other_duration.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/other_duration.pdf",dpi = 300)
# ____________________ extract KDE for other start time
# plt.figure()
# Build PDF and turn into pandas Series
x2 = np.linspace(2, 26, 10000)
y2 = gaussian_kde(train_otherData['start(hour)']).pdf(x2)
pdf2 = pd.Series(y2, x2)
plt.figure(figsize=(6, 4.5))
plt.xticks((np.arange(2, 26, step=4)),fontsize=11)
plt.yticks(fontsize=11)
ax2 = train_otherData['start(hour)'].plot(kind='hist', bins=60, density=True, alpha=0.5, label='test set', legend=True)
pdf2.plot(lw=2, label='Gaussian Kernel Density Estimation', legend=True, ax=ax2)
ax2.set_title(u'other start time.',fontsize=16)
ax2.set_xlabel(u'Start Time (hour)',fontsize=14)
ax2.set_ylabel('Frequency',fontsize=14)
plt.legend(fontsize=11)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/other_start.png",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/other_start.pdf",dpi = 300)
# ************************end of: KDE application plots on GT *****************************
#*************************start of: location confusion matrix calculations **************************
nactivity = len(train_activityData)
ntrip = len(train_tripData)
prior_activity = nactivity/(nactivity + ntrip)
prior_trip = ntrip/(nactivity + ntrip)
test_activityData = pd.concat([activityDurations,activityStarts],axis=1)
test_activityData['Duration(hour)'] = test_activityData['duration(sec)']/3600
test_activityData['start(hour)'] = test_activityData['start_time(sec)']/3600
test_activityData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)']).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)']).pdf(test_activityData['start(hour)']))
test_activityData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)']).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)']).pdf(test_activityData['start(hour)']))
test_activityData['algorithm select activity'] = 0
p=0
for p in test_activityData.index:
    if (test_activityData['Log prob. of being activity']>=test_activityData['Log prob. of being trip'])[p] :
        test_activityData['algorithm select activity'][p] = 1
print(sum(test_activityData['algorithm select activity'])/len(test_activityData['algorithm select activity']))
#***************************************************************************
test_tripData = pd.concat([tripDurations,tripStarts],axis=1)
test_tripData['Duration(hour)'] = test_tripData['duration(sec)']/3600
test_tripData['start(hour)'] = test_tripData['start_time(sec)']/3600
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
import openpyxl
conMat.to_excel('C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/confusion_matrix_stayPassby.xlsx',header=True,index=True, engine='openpyxl')
#*************************start of: activity confusion matrix calculations **************************
nHome = len(train_homeData)
nWork = len(train_workData)
nOther= len(train_otherData)
prior_home = nHome/(nHome + nWork + nOther)
prior_work = nWork/(nHome + nWork + nOther)
prior_other = nOther/(nHome + nWork + nOther)
test_homeData = pd.concat([homeDurations,homeStarts],axis=1)
test_homeData['Duration(hour)']=test_homeData['duration(sec)']/3600
test_homeData['start(hour)']= test_homeData['start_time(sec)']/3600
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
test_workData = pd.concat([workDurations,workStarts],axis=1)
test_workData['Duration(hour)']=test_workData['duration(sec)']/3600
test_workData['start(hour)']= test_workData['start_time(sec)']/3600
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
test_otherData = pd.concat([otherDurations,otherStarts],axis=1)
test_otherData['Duration(hour)']=test_otherData['duration(sec)']/3600
test_otherData['start(hour)']= test_otherData['start_time(sec)']/3600
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
conMat = confusion_matrix(y_actu,y_pred) #left: actual and top: predicted
conMat = pd.DataFrame(conMat, columns=colName, index=rowName)
import openpyxl
# conMat.to_excel('C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/confusion_matrix_activityType.xlsx',header=True,index=True, engine='openpyxl')
# \todo this does noe compute confusion matrix in the correct order

# *************************************************************************************************
# *************************************************************************************************
# ******************************* plot OD comparison results **************************************
# *************************************************************************************************
# *************************************************************************************************
import pandas as pd
from scipy.stats import gaussian_kde
import time
import numpy as np
import matplotlib.pyplot as plt
import xlrd
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')
GSSI_var = pd.read_excel("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/ODresults_tobePlot.xlsx", sheet_name="GSSI_totalOD_var")
GSSI_avg = pd.read_excel("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/ODresults_tobePlot.xlsx", sheet_name="GSSI_totalOD_avg")
sum_var = pd.read_excel("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/ODresults_tobePlot.xlsx", sheet_name="sum_totalOD_var")
sum_avg = pd.read_excel("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/ODresults_tobePlot.xlsx", sheet_name="sum_totalOD_avg")
slope_var = pd.read_excel("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/ODresults_tobePlot.xlsx", sheet_name="slope_totalOD_var")
slope_avg = pd.read_excel("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/ODresults_tobePlot.xlsx", sheet_name="slope_totalOD_avg")
rsq_var = pd.read_excel("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/ODresults_tobePlot.xlsx", sheet_name="Rsq_totalOD_var")
rsq_avg = pd.read_excel("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/ODresults_tobePlot.xlsx", sheet_name="Rsq_totalOD_avg")
plt.figure(figsize=(9, 5))
plt.xticks(np.arange(0, 3601, step=600),fontsize=12)
plt.yticks(fontsize=12)
plt.plot(GSSI_var['polling interval (sec)'],GSSI_var['var'],marker='o', markersize=12, color='darksalmon', linewidth=3,markerfacecolor='DarkBlue')
plt.xlabel(u'polling interval (sec)',fontsize=14)
plt.title('Variance of GSSI related to O-D matrices resulted \n from 25 different random seeds for 12 different polling intervals',fontsize=14)
plt.ylabel('variance of GSSI',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/GSSI_totalOD_var.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/GSSI_totalOD_var.png",dpi = 300)


plt.figure(figsize=(9, 5))
plt.xticks(np.arange(0, 3601, step=600),fontsize=12)
plt.yticks(fontsize=12)
plt.plot(GSSI_avg['polling interval (sec)'],GSSI_avg['avg'],marker='o', markersize=12, color='skyblue', linewidth=3,markerfacecolor='DarkBlue')
plt.xlabel(u'polling interval (sec)',fontsize=14)
plt.title('Average of GSSI related to O-D matrices resulted \n from 25 different random seeds for 12 different polling intervals',fontsize=14)
plt.ylabel('average of GSSI',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/GSSI_totalOD_avg.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/GSSI_totalOD_avg.png",dpi = 300)


plt.figure(figsize=(9, 5))
plt.xticks(np.arange(0, 3601, step=600),fontsize=12)
plt.yticks(fontsize=12)
plt.plot(sum_var['polling interval (sec)'],sum_var['var'],marker='o', markersize=12, color='darksalmon', linewidth=3,markerfacecolor='DarkBlue')
plt.xlabel(u'polling interval (sec)',fontsize=14)
plt.title('Variance of total trip related to O-D matrices resulted \n from 25 different random seeds for 12 different polling intervals',fontsize=14)
plt.ylabel('variance of total trip',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/sum_totalOD_var.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/sum_totalOD_var.png",dpi = 300)


plt.figure(figsize=(9, 5))
plt.xticks(np.arange(0, 3601, step=600),fontsize=12)
plt.yticks(fontsize=12)
plt.plot(sum_avg['polling interval (sec)'],sum_avg['avg'],marker='o', markersize=12, color='skyblue', linewidth=3,markerfacecolor='DarkBlue')
plt.xlabel(u'polling interval (sec)',fontsize=14)
plt.title('Average of total trip related to O-D matrices resulted \n from 25 different random seeds for 12 different polling intervals',fontsize=14)
plt.ylabel('average of total trip',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/sum_totalOD_avg.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/sum_totalOD_avg.png",dpi = 300)


plt.figure(figsize=(9, 5))
plt.xticks(np.arange(0, 3601, step=600),fontsize=12)
plt.yticks(fontsize=12)
plt.plot(slope_var['polling interval (sec)'],slope_var['var'],marker='o', markersize=12, color='darksalmon', linewidth=3,markerfacecolor='DarkBlue')
plt.xlabel(u'polling interval (sec)',fontsize=14)
plt.title('Variance of linear model coefficient related to O-D matrices resulted \n from 25 different random seeds for 12 different polling intervals',fontsize=14)
plt.ylabel('variance of linear model coefficient',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/slope_totalOD_var.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/slope_totalOD_var.png",dpi = 300)


plt.figure(figsize=(9, 5))
plt.xticks(np.arange(0, 3601, step=600),fontsize=12)
plt.yticks(fontsize=12)
plt.plot(slope_avg['polling interval (sec)'],slope_avg['avg'],marker='o', markersize=12, color='skyblue', linewidth=3,markerfacecolor='DarkBlue')
plt.xlabel(u'polling interval (sec)',fontsize=14)
plt.title('Average of linear model coefficient related to O-D matrices resulted \n from 25 different random seeds for 12 different polling intervals',fontsize=14)
plt.ylabel('average of linear model coefficient',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/slope_totalOD_avg.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/slope_totalOD_avg.png",dpi = 300)


plt.figure(figsize=(9, 5))
plt.xticks(np.arange(0, 3601, step=600),fontsize=12)
plt.yticks(fontsize=12)
plt.plot(rsq_var['polling interval (sec)'],rsq_var['var'],marker='o', markersize=12, color='darksalmon', linewidth=3,markerfacecolor='DarkBlue')
plt.xlabel(u'polling interval (sec)',fontsize=14)
plt.title('Variance of linear model R-squared related to O-D matrices resulted \n from 25 different random seeds for 12 different polling intervals',fontsize=14)
plt.ylabel('variance of R-squared ',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/rsq_totalOD_var.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/rsq_totalOD_var.png",dpi = 300)


plt.figure(figsize=(9, 5))
plt.xticks(np.arange(0, 3601, step=600),fontsize=12)
plt.yticks(fontsize=12)
plt.plot(rsq_avg['polling interval (sec)'],rsq_avg['avg'],marker='o', markersize=12, color='skyblue', linewidth=3,markerfacecolor='DarkBlue')
plt.xlabel(u'polling interval (sec)',fontsize=14)
plt.title('Average of linear model R-squared related to O-D matrices resulted \n from 25 different random seeds for 12 different polling intervals',fontsize=14)
plt.ylabel('average of R-squared ',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/rsq_totalOD_avg.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/rsq_totalOD_avg.png",dpi = 300)
























# _________________________ number of activities per user ______________________________
homeFalseNeg = pd.DataFrame(columns=['predicted','start(sec),duration(sec)'])
homeFalseNeg['predicted']=test_homeData['predicted activity']
homeFalseNeg['start(sec)']=test_homeData['start_time(sec)']
homeFalseNeg['duration(sec)']=test_homeData['duration(sec)']
homeFalseNeg['comparison'] = test_homeData['observed activity']==test_homeData['predicted activity']
homeFalseNeg['workFPs'] = homeFalseNeg['predicted']=='Log prob. of being work'
workFalsePos_home_start= homeFalseNeg['start(sec)'][homeFalseNeg['workFPs']==1]
workFalsePos_home_duration= homeFalseNeg['duration(sec)'][homeFalseNeg['workFPs']==1]
sum(homeFalseNeg['comparison'])/len(homeFalseNeg)
homeFalseNeg_start = homeFalseNeg['start(sec)'][homeFalseNeg['comparison']==0]
homeFalseNeg_duration = homeFalseNeg['duration(sec)'][homeFalseNeg['comparison']==0]
# ________________________________________________________________________
workFalseNeg = pd.DataFrame(columns=['predicted','start(sec)'])
workFalseNeg['predicted']=test_workData['predicted activity']
workFalseNeg['start(sec)']=test_workData['start_time(sec)']
workFalseNeg['duration(sec)']=test_workData['duration(sec)']
workFalseNeg['comparison'] = test_workData['observed activity']==test_workData['predicted activity']
sum(workFalseNeg['comparison'])/len(workFalseNeg)
workFalseNeg_start = workFalseNeg['start(sec)'][workFalseNeg['comparison']==0]
workFalseNeg_duration = workFalseNeg['duration(sec)'][workFalseNeg['comparison']==0]
# ________________________________________________________________________
otherFalseNeg = pd.DataFrame(columns=['predicted','start(sec)'])
otherFalseNeg['predicted']=test_otherData['predicted activity']
otherFalseNeg['start(sec)']=test_otherData['start_time(sec)']
otherFalseNeg['duration(sec)']=test_otherData['duration(sec)']
otherFalseNeg['comparison'] = test_otherData['observed activity']==test_otherData['predicted activity']
otherFalseNeg['workFPs'] = otherFalseNeg['predicted']=='Log prob. of being work'
workFalsePos_other_start= otherFalseNeg['start(sec)'][otherFalseNeg['workFPs']==1]
workFalsePos_other_duration= otherFalseNeg['duration(sec)'][otherFalseNeg['workFPs']==1]
sum(otherFalseNeg['comparison'])/len(otherFalseNeg)
otherFalseNeg_start = otherFalseNeg['start(sec)'][otherFalseNeg['comparison']==0]
otherFalseNeg_duration = otherFalseNeg['duration(sec)'][otherFalseNeg['comparison']==0]
# ________________________________________________________________________
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')
# plt.figure(figsize=(12, 8))
params = {'figure.figsize': (8, 5)}
plt.rcParams.update(params)
# __________________ HOME PredictionError_start______________
plt.figure()
plt.xticks((np.arange(0, 24, step=2)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (homeFalseNeg_start/3600).plot(kind='hist', bins=90, density=True, alpha=0.5, label='prediction error of home activity', legend=False)
ax1.set_xlabel(u'start (hour)',fontsize=14)
ax1.set_title('home false-negative error based on start of the activity ',fontsize=14)
ax1.set_ylabel('error frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/homeFNs_start.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/homeFNs_start.png",dpi = 300)
# ____________________WORK PredictionError_start _______________
plt.figure()
plt.xticks((np.arange(0, 24, step=2)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (workFalseNeg_start/3600).plot(kind='hist', bins=90, density=True, alpha=0.5, label='prediction error of work activity', legend=False)
ax1.set_xlabel(u'start (hour)',fontsize=14)
ax1.set_title('work false-negative error based on start of the activity ',fontsize=14)
ax1.set_ylabel('error frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFNs_start.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFNs_start.png",dpi = 300)
# ____________________OTHER PredictionError_start _______________
plt.figure()
plt.xticks((np.arange(0, 24, step=2)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (otherFalseNeg_start/3600).plot(kind='hist', bins=90, density=True, alpha=0.5, label='prediction error of other activity', legend=False)
ax1.set_xlabel(u'start (hour)',fontsize=14)
ax1.set_title('other false-negative error based on start of the activity ',fontsize=14)
ax1.set_ylabel('error frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/otherFNs_start.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/otherFNs_start.png",dpi = 300)

# ***********************FNs_duration *************************************


# __________________ HOME PredictionError_duration______________
plt.figure()
plt.xticks((np.arange(0, 24, step=2)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (homeFalseNeg_duration/3600).plot(kind='hist', bins=90, density=True, alpha=0.5, label='prediction error of home activity', legend=False)
ax1.set_xlabel(u'duration (hour)',fontsize=14)
ax1.set_title('home false-negative error based on duration of the activity ',fontsize=14)
ax1.set_ylabel('error frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/homeFNs_duration.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/homeFNs_duration.png",dpi = 300)
# ____________________WORK PredictionError_duration _______________
plt.figure()
plt.xticks((np.arange(0, 24, step=2)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (workFalseNeg_duration/3600).plot(kind='hist', bins=90, density=True, alpha=0.5, label='prediction error of work activity', legend=False)
ax1.set_xlabel(u'duration (hour)',fontsize=14)
ax1.set_title('work false-negative error based on duration of the activity ',fontsize=14)
ax1.set_ylabel('error frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFNs_duration.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFNs_duration.png",dpi = 300)
# ____________________OTHER PredictionError_duration _______________
plt.figure()
plt.xticks((np.arange(0, 24, step=2)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (otherFalseNeg_duration/3600).plot(kind='hist', bins=90, density=True, alpha=0.5, label='prediction error of other activity', legend=False)
ax1.set_xlabel(u'duration (hour)',fontsize=14)
ax1.set_title('other false-negative error based on duration of the activity ',fontsize=14)
ax1.set_ylabel('error frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/otherFNs_duration.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/otherFNs_duration.png",dpi = 300)





# # *****************************work FP**********************************
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (16.0, 12.0)
# plt.style.use('ggplot')
# # plt.figure(figsize=(12, 8))
# params = {'figure.figsize': (8, 5)}
# plt.rcParams.update(params)
# # __________________ work FP_HOME PredictionError_start______________
# plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
# plt.yticks(fontsize=12)
# ax1 = (workFalsePos_home_start/3600).plot(kind='hist', bins=30, density=True, alpha=0.5, label='work False positive to home', legend=False)
# ax1.set_xlabel(u'start (hour)',fontsize=14)
# ax1.set_title('work false-positive error to home based on start of the activity ',fontsize=14)
# ax1.set_ylabel('error frequency',fontsize=14)
# # plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFPs_home_start.pdf",dpi = 300)
# # plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFPs_home_start.png",dpi = 300)
# # __________________ work FP_OTHER PredictionError_start______________
# plt.figure()
# plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
# plt.yticks(fontsize=12)
# ax1 = (workFalsePos_other_start/3600).plot(kind='hist', bins=50, density=True, alpha=0.5, label='work False positive to other', legend=False)
# ax1.set_xlabel(u'start (hour)',fontsize=14)
# ax1.set_title('work false-positive error to other based on start of the activity ',fontsize=14)
# ax1.set_ylabel('error frequency',fontsize=14)
# # plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFPs_other_start.pdf",dpi = 300)
# # plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFPs_other_start.png",dpi = 300)
#
# # __________________ work FP_HOME PredictionError_duration______________
# plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
# plt.yticks(fontsize=12)
# ax1 = (workFalsePos_home_duration/3600).plot(kind='hist', bins=30, density=True, alpha=0.5, label='work False positive to home', legend=False)
# ax1.set_xlabel(u'duration (hour)',fontsize=14)
# ax1.set_title('work false-positive error to home based on duration of the activity ',fontsize=14)
# ax1.set_ylabel('error frequency',fontsize=14)
# # plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFPs_home_duration.pdf",dpi = 300)
# # plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFPs_home_duration.png",dpi = 300)
# # __________________ work FP_OTHER PredictionError_duration______________
# plt.figure()
# plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
# plt.yticks(fontsize=12)
# ax1 = (workFalsePos_other_duration/3600).plot(kind='hist', bins=50, density=True, alpha=0.5, label='work False positive to other', legend=False)
# ax1.set_xlabel(u'duration (hour)',fontsize=14)
# ax1.set_title('work false-positive error to other based on duration of the activity ',fontsize=14)
# ax1.set_ylabel('error frequency',fontsize=14)
# # plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFPs_other_duration.pdf",dpi = 300)
# # plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workFPs_other_duration.png",dpi = 300)
#
#


# *********************actual work start histogram ***********************
plt.figure()
plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (workStarts/3600).plot(kind='hist', bins=50, density=True, alpha=0.5, label='work start time', legend=False)
ax1.set_xlabel(u'start (hour)',fontsize=14)
ax1.set_title('work distrubution during the day based on the start time',fontsize=14)
ax1.set_ylabel('frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workActual_start.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workActual_start.png",dpi = 300)
# *********************actual home start histogram ***********************
plt.figure()
plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (homeStarts/3600).plot(kind='hist', bins=50, density=True, alpha=0.5, label='home start time', legend=False)
ax1.set_xlabel(u'start (hour)',fontsize=14)
ax1.set_title('home distrubution during the day based on the start time',fontsize=14)
ax1.set_ylabel('frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/homeActual_start.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/homeActual_start.png",dpi = 300)
# *********************actual other start histogram ***********************
plt.figure()
plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (otherStarts/3600).plot(kind='hist', bins=50, density=True, alpha=0.5, label='other start time', legend=False)
ax1.set_xlabel(u'start (hour)',fontsize=14)
ax1.set_title('other distrubution during the day based on the start time',fontsize=14)
ax1.set_ylabel('frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/otherActual_start.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/otherActual_start.png",dpi = 300)


# *********************actual work duration histogram ***********************
plt.figure()
plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (workDurations/3600).plot(kind='hist', bins=90, density=True, alpha=0.5, label='work duration time', legend=False)
ax1.set_xlabel(u'duration (hour)',fontsize=14)
ax1.set_title('work distrubution during the day based on the duration',fontsize=14)
ax1.set_ylabel('frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workActual_duration.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/workActual_duration.png",dpi = 300)
# *********************actual home duration histogram ***********************
plt.figure()
plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (homeDurations/3600).plot(kind='hist', bins=90, density=True, alpha=0.5, label='home duration time', legend=False)
ax1.set_xlabel(u'duration (hour)',fontsize=14)
ax1.set_title('home distrubution during the day based on the duration',fontsize=14)
ax1.set_ylabel('frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/homeActual_duration.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/homeActual_duration.png",dpi = 300)
# *********************actual other duration histogram ***********************
plt.figure()
plt.xticks((np.arange(0, 24, step=1)),fontsize=12)
plt.yticks(fontsize=12)
ax1 = (otherDurations/3600).plot(kind='hist', bins=90, density=True, alpha=0.5, label='other duration time', legend=False)
ax1.set_xlabel(u'duration (hour)',fontsize=14)
ax1.set_title('other distrubution during the day based on the duration',fontsize=14)
ax1.set_ylabel('frequency',fontsize=14)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/otherActual_duration.pdf",dpi = 300)
plt.savefig("C:/Users/zahraeftekhar/Desktop/New folder/time aggregation/kde test/otherActual_duration.png",dpi = 300)