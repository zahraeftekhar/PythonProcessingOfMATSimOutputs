import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
import math
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import random
import time
# ______________ importing required data _________________________
start_time = time.time()
from groundTruthAnalysis_homeWorkDistribution import homeDurations, homeStarts, workDurations, workStarts, \
     otherDurations, otherStarts
from groundTruthAnalysis_homeWorkDistribution import  tripDurations, tripStarts, activityDurations,activityStarts
plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')
# ________________________________________test different seeds for random sampling ______________________________________________
seedSet = range(101,151)
# seed = 101
sensitivityTable = pd.DataFrame(columns=['seed','pass-by:Bayesian','pass-by:40min','pass-by:60min', 'stay:Bayesian','stay:40min','stay:60min', 'location:Bayesian','location:40min','location:60min'])
sens_locActivity = pd.DataFrame(columns=['seed','locationRecognition','activityRecognition','overall'])
for s, seed in enumerate(seedSet):
    print(s, end='______ time:{tt}sec\n'.format(tt = time.time()-start_time))
    # ____________________ random ACTIVITY sample as the training set from ALL activities
    activityData = pd.DataFrame()
    activityData['Duration(hour)'] = activityDurations['duration(sec)']/3600
    activityData['start(hour)'] = activityStarts['start_time(sec)']/3600
    activityData['type'] = activityStarts['type']
    random.seed(seed)
    train_activityIndex = random.sample(range(len(activityData)),round(.01*len(activityData))) #random number from uniform dist.
    nactivity = len(train_activityIndex)
    test_activityIndex = np.delete(range(len(activityData)),train_activityIndex)
    train_activityData = activityData.loc[train_activityIndex]
    test_activityData = activityData.loc[test_activityIndex]
    # ____________________ selecting random trip sample as the training set from ALL trips
    tripData = pd.DataFrame()
    tripData['Duration(hour)'] = tripDurations['duration(sec)']/3600
    tripData['start(hour)'] = tripStarts['start_time(sec)']/3600
    # tripData['type'] = None
    random.seed(seed)
    train_tripIndex = random.sample(range(len(tripData)),round(.01*len(tripData))) #random number from uniform dist.
    ntrip = len(train_tripIndex)
    test_tripIndex = np.delete(range(len(tripData)),train_tripIndex)
    train_tripData = tripData.loc[train_tripIndex]
    test_tripData = tripData.loc[test_tripIndex]
    # ____________________ random WORK sample as the training set from ALL activities
    workData = pd.DataFrame()
    workData['Duration(hour)'] = workDurations['duration(sec)']/3600
    workData['start(hour)'] = workStarts['start_time(sec)']/3600
    random.seed(seed)
    train_workIndex = random.sample(range(len(workData)),round(.01*len(workData))) #random number from uniform dist.
    nWork = len(train_workIndex)
    test_workIndex = np.delete(range(len(workData)),train_workIndex)
    train_workData = workData.loc[train_workIndex]
    test_workData = workData.loc[test_workIndex]
    # ____________________ random HOME sample as the training set from ALL activities
    homeData = pd.DataFrame()
    homeData['Duration(hour)'] = (homeDurations['duration(sec)'])/3600
    homeData['start(hour)'] = (homeStarts['start_time(sec)'])/3600
    random.seed(seed)
    train_homeIndex = random.sample(range(len(homeData)),round(.01*len(homeData))) #random number from uniform dist.
    nHome = len(train_homeIndex)
    test_homeIndex = np.delete(range(len(homeData)),train_homeIndex)
    train_homeData = homeData.loc[train_homeIndex]
    test_homeData = homeData.loc[test_homeIndex]
    # ____________________ random OTHER sample as the training set from ALL activities
    otherData = pd.DataFrame()
    otherData['Duration(hour)'] = otherDurations['duration(sec)']/3600
    otherData['start(hour)'] = otherStarts['start_time(sec)']/3600
    random.seed(seed)
    train_otherIndex = random.sample(range(len(otherData)),round(.01*len(otherData))) #random number from uniform dist.
    nOther = len(train_otherIndex)
    test_otherIndex = np.delete(range(len(otherData)),train_otherIndex)
    train_otherData = otherData.loc[train_otherIndex]
    test_otherData = otherData.loc[test_otherIndex]

    # ___________________________ Probability calculations_________________________________________
    #***************************************************************************
    prior_activity = nactivity/(nactivity + ntrip)
    prior_trip = ntrip/(nactivity + ntrip)
    test_activityData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)'],bw_method=0.1).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)'],bw_method=0.1).pdf(test_activityData['start(hour)']))
    test_activityData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)'],bw_method=0.1).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)'],bw_method=0.1).pdf(test_activityData['start(hour)']))
    test_activityData['algorithm select activity'] = 0
    p=0
    for p in test_activityData.index:
        if (test_activityData['Log prob. of being activity']>=test_activityData['Log prob. of being trip'])[p] :
            test_activityData['algorithm select activity'][p] = 1
    # print(sum(test_activityData['algorithm select activity'])/len(test_activityData['algorithm select activity']))
    #***************************************************************************
    test_tripData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)'],bw_method=0.1).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)'],bw_method=0.1).pdf(test_tripData['start(hour)']))
    test_tripData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)'],bw_method=0.1).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)'],bw_method=0.1).pdf(test_tripData['start(hour)']))
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
    test_data['home'] = np.log(prior_home)+np.log( gaussian_kde(train_homeData['Duration(hour)']).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_homeData['start(hour)']).pdf(test_data['start(hour)']))
    test_data['work'] = np.log(prior_work)+np.log( gaussian_kde(train_workData['Duration(hour)']).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_workData['start(hour)']).pdf(test_data['start(hour)']))
    test_data['other'] = np.log(prior_other)+np.log( gaussian_kde(train_otherData['Duration(hour)']).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_otherData['start(hour)']).pdf(test_data['start(hour)']))
    test_data['predictedActivity'] = test_data[['home','work','other']].idxmax(axis=1)
    ####### adding time limit for location type identification #########
    test_tripData['satisfying time limit'] = test_tripData['Duration(hour)']<.6666 #40 min as the time limit
    test_tripData['satisfying time limit(1h)'] = test_tripData['Duration(hour)']<1 #60 min as the time limit
    test_activityData['satisfying time limit'] = test_activityData['Duration(hour)']>=.6666 #40 min as the time limit
    test_activityData['satisfying time limit(1h)'] = test_activityData['Duration(hour)']>=1 #60 min as the time limit
    # print("____________________trip,seed:{ss} _____________________".format(ss=seed))
    # print(sum(test_tripData['algorithm select trip'])/len(test_tripData['satisfying time limit']))#performance of Bayesian in trip identification
    # print(sum(test_tripData['satisfying time limit'])/len(test_tripData['satisfying time limit']))#performance of time limit in trip identification
    # print(sum(test_tripData['satisfying time limit(1h)'])/len(test_tripData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
    # print("____________________stay,seed:{ss}  _____________________".format(ss=seed))
    # print(sum(test_activityData['satisfying time limit'])/len(test_activityData['satisfying time limit']))#performance of time limit in trip identification
    # print(sum(test_activityData['satisfying time limit(1h)'])/len(test_activityData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
    # print(sum(test_activityData['algorithm select activity'])/len(test_activityData['satisfying time limit']))#performance of Bayesian in trip identification
    # print("____________________location,seed:{ss}  _____________________".format(ss=seed))
    # print((sum(test_activityData['satisfying time limit'])+sum(test_tripData['satisfying time limit']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
    # print((sum(test_activityData['satisfying time limit(1h)'])+sum(test_tripData['satisfying time limit(1h)']))/(len(test_tripData['satisfying time limit(1h)'])+len(test_activityData['satisfying time limit(1h)'])))
    # print((sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
    # ___________________________________________________________________________________________________________________________________________________________________________________
    sensitivityTable.loc[s,'seed'] = seed
    # print("____________________trip,seed:{ss} _____________________".format(ss=seed))
    sensitivityTable.loc[s,'pass-by:Bayesian']=(sum(test_tripData['algorithm select trip'])/len(test_tripData['satisfying time limit']))#performance of Bayesian in trip identification
    sensitivityTable.loc[s,'pass-by:40min']=(sum(test_tripData['satisfying time limit'])/len(test_tripData['satisfying time limit']))#performance of time limit in trip identification
    sensitivityTable.loc[s,'pass-by:60min']=(sum(test_tripData['satisfying time limit(1h)'])/len(test_tripData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
    # print("____________________stay,seed:{ss}  _____________________".format(ss=seed))
    sensitivityTable.loc[s,'stay:40min']=(sum(test_activityData['satisfying time limit'])/len(test_activityData['satisfying time limit']))#performance of time limit in trip identification
    sensitivityTable.loc[s,'stay:60min']=(sum(test_activityData['satisfying time limit(1h)'])/len(test_activityData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
    sensitivityTable.loc[s,'stay:Bayesian']=(sum(test_activityData['algorithm select activity'])/len(test_activityData['satisfying time limit']))#performance of Bayesian in trip identification
    # print("____________________location,seed:{ss}  _____________________".format(ss=seed))
    sensitivityTable.loc[s,'location:40min']=((sum(test_activityData['satisfying time limit'])+sum(test_tripData['satisfying time limit']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
    sensitivityTable.loc[s,'location:60min']=((sum(test_activityData['satisfying time limit(1h)'])+sum(test_tripData['satisfying time limit(1h)']))/(len(test_tripData['satisfying time limit(1h)'])+len(test_activityData['satisfying time limit(1h)'])))
    sensitivityTable.loc[s,'location:Bayesian']=((sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
    sens_locActivity.loc[s,'seed'] = seed
    sens_locActivity.loc[s,'locationRecognition'] = (sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData)+len(test_activityData))
    sens_locActivity.loc[s,'activityRecognition'] = (len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data))
    sens_locActivity.loc[s,'overall'] = (sum(test_tripData['algorithm select trip'])+len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data)+len(test_tripData))

sensitivityTable.to_excel('D:/progress meeting/15July2020(Hans&Adam)/onePercentLocationDetectionSensitivity.xlsx', header=True)
sens_locActivity.to_excel('D:/progress meeting/15July2020(Hans&Adam)/onePercentLocationActivityDetectionSensitivity.xlsx',
                              header=True)
# ***********************************************************************************************************************
#  ______________________________test bandwidth of KDE ____________________________________________________
bwSet = np.arange(.05,3.05,0.05)
# seed = 101
# sensitivityTable = pd.DataFrame(columns=['seed','pass-by:Bayesian','pass-by:40min','pass-by:60min', 'stay:Bayesian','stay:40min','stay:60min', 'location:Bayesian','location:40min','location:60min'])
sens_locActivity = pd.DataFrame(columns=['seed','locationRecognition','activityRecognition','overall'])
for s,bandwidth in enumerate(bwSet):
    print(s, end='______ time:{tt}sec\n'.format(tt = time.time()-start_time))
    # ____________________ random ACTIVITY sample as the training set from ALL activities
    activityData = pd.DataFrame()
    activityData['Duration(hour)'] = activityDurations['duration(sec)']/3600
    activityData['start(hour)'] = activityStarts['start_time(sec)']/3600
    activityData['type'] = activityStarts['type']
    seed = 101
    random.seed(seed)
    train_activityIndex = random.sample(range(len(activityData)),round(.01*len(activityData))) #random number from uniform dist.
    nactivity = len(train_activityIndex)
    test_activityIndex = np.delete(range(len(activityData)),train_activityIndex)
    train_activityData = activityData.loc[train_activityIndex]
    test_activityData = activityData.loc[test_activityIndex]
    # ____________________ selecting random trip sample as the training set from ALL trips
    tripData = pd.DataFrame()
    tripData['Duration(hour)'] = tripDurations['duration(sec)']/3600
    tripData['start(hour)'] = tripStarts['start_time(sec)']/3600
    # tripData['type'] = None
    random.seed(seed)
    train_tripIndex = random.sample(range(len(tripData)),round(.01*len(tripData))) #random number from uniform dist.
    ntrip = len(train_tripIndex)
    test_tripIndex = np.delete(range(len(tripData)),train_tripIndex)
    train_tripData = tripData.loc[train_tripIndex]
    test_tripData = tripData.loc[test_tripIndex]
    # ____________________ random WORK sample as the training set from ALL activities
    workData = pd.DataFrame()
    workData['Duration(hour)'] = workDurations['duration(sec)']/3600
    workData['start(hour)'] = workStarts['start_time(sec)']/3600
    random.seed(seed)
    train_workIndex = random.sample(range(len(workData)),round(.01*len(workData))) #random number from uniform dist.
    nWork = len(train_workIndex)
    test_workIndex = np.delete(range(len(workData)),train_workIndex)
    train_workData = workData.loc[train_workIndex]
    test_workData = workData.loc[test_workIndex]
    # ____________________ random HOME sample as the training set from ALL activities
    homeData = pd.DataFrame()
    homeData['Duration(hour)'] = (homeDurations['duration(sec)'])/3600
    homeData['start(hour)'] = (homeStarts['start_time(sec)'])/3600
    random.seed(seed)
    train_homeIndex = random.sample(range(len(homeData)),round(.01*len(homeData))) #random number from uniform dist.
    nHome = len(train_homeIndex)
    test_homeIndex = np.delete(range(len(homeData)),train_homeIndex)
    train_homeData = homeData.loc[train_homeIndex]
    test_homeData = homeData.loc[test_homeIndex]
    # ____________________ random OTHER sample as the training set from ALL activities
    otherData = pd.DataFrame()
    otherData['Duration(hour)'] = otherDurations['duration(sec)']/3600
    otherData['start(hour)'] = otherStarts['start_time(sec)']/3600
    random.seed(seed)
    train_otherIndex = random.sample(range(len(otherData)),round(.01*len(otherData))) #random number from uniform dist.
    nOther = len(train_otherIndex)
    test_otherIndex = np.delete(range(len(otherData)),train_otherIndex)
    train_otherData = otherData.loc[train_otherIndex]
    test_otherData = otherData.loc[test_otherIndex]

    # ___________________________ Probability calculations_________________________________________
    #***************************************************************************
    prior_activity = nactivity/(nactivity + ntrip)
    prior_trip = ntrip/(nactivity + ntrip)
    test_activityData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)'],bw_method=bandwidth).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)'],bw_method=bandwidth).pdf(test_activityData['start(hour)']))
    test_activityData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)'],bw_method=bandwidth).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)'],bw_method=bandwidth).pdf(test_activityData['start(hour)']))
    test_activityData['algorithm select activity'] = 0
    p=0
    for p in test_activityData.index:
        if (test_activityData['Log prob. of being activity']>=test_activityData['Log prob. of being trip'])[p] :
            test_activityData['algorithm select activity'][p] = 1
    # print(sum(test_activityData['algorithm select activity'])/len(test_activityData['algorithm select activity']))
    #***************************************************************************
    test_tripData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)'],bw_method=bandwidth).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)'],bw_method=bandwidth).pdf(test_tripData['start(hour)']))
    test_tripData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)'],bw_method=bandwidth).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)'],bw_method=bandwidth).pdf(test_tripData['start(hour)']))
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
    test_data['home'] = np.log(prior_home)+np.log( gaussian_kde(train_homeData['Duration(hour)'],bw_method=bandwidth).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_homeData['start(hour)'],bw_method=bandwidth).pdf(test_data['start(hour)']))
    test_data['work'] = np.log(prior_work)+np.log( gaussian_kde(train_workData['Duration(hour)'],bw_method=bandwidth).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_workData['start(hour)'],bw_method=bandwidth).pdf(test_data['start(hour)']))
    test_data['other'] = np.log(prior_other)+np.log( gaussian_kde(train_otherData['Duration(hour)'],bw_method=bandwidth).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_otherData['start(hour)'],bw_method=bandwidth).pdf(test_data['start(hour)']))
    test_data['predictedActivity'] = test_data[['home','work','other']].idxmax(axis=1)
    ####### adding time limit for location type identification #########
    test_tripData['satisfying time limit'] = test_tripData['Duration(hour)']<.6666 #40 min as the time limit
    test_tripData['satisfying time limit(1h)'] = test_tripData['Duration(hour)']<1 #60 min as the time limit
    test_activityData['satisfying time limit'] = test_activityData['Duration(hour)']>=.6666 #40 min as the time limit
    test_activityData['satisfying time limit(1h)'] = test_activityData['Duration(hour)']>=1 #60 min as the time limit
    # print("____________________trip,seed:{ss} _____________________".format(ss=seed))
    # print(sum(test_tripData['algorithm select trip'])/len(test_tripData['satisfying time limit']))#performance of Bayesian in trip identification
    # print(sum(test_tripData['satisfying time limit'])/len(test_tripData['satisfying time limit']))#performance of time limit in trip identification
    # print(sum(test_tripData['satisfying time limit(1h)'])/len(test_tripData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
    # print("____________________stay,seed:{ss}  _____________________".format(ss=seed))
    # print(sum(test_activityData['satisfying time limit'])/len(test_activityData['satisfying time limit']))#performance of time limit in trip identification
    # print(sum(test_activityData['satisfying time limit(1h)'])/len(test_activityData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
    # print(sum(test_activityData['algorithm select activity'])/len(test_activityData['satisfying time limit']))#performance of Bayesian in trip identification
    # print("____________________location,seed:{ss}  _____________________".format(ss=seed))
    # print((sum(test_activityData['satisfying time limit'])+sum(test_tripData['satisfying time limit']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
    # print((sum(test_activityData['satisfying time limit(1h)'])+sum(test_tripData['satisfying time limit(1h)']))/(len(test_tripData['satisfying time limit(1h)'])+len(test_activityData['satisfying time limit(1h)'])))
    # print((sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
    # ___________________________________________________________________________________________________________________________________________________________________________________
    # sensitivityTable.loc[s,'bw'] = seed
    # # print("____________________trip,seed:{ss} _____________________".format(ss=seed))
    # sensitivityTable.loc[s,'pass-by:Bayesian']=(sum(test_tripData['algorithm select trip'])/len(test_tripData['satisfying time limit']))#performance of Bayesian in trip identification
    # sensitivityTable.loc[s,'pass-by:40min']=(sum(test_tripData['satisfying time limit'])/len(test_tripData['satisfying time limit']))#performance of time limit in trip identification
    # sensitivityTable.loc[s,'pass-by:60min']=(sum(test_tripData['satisfying time limit(1h)'])/len(test_tripData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
    # # print("____________________stay,seed:{ss}  _____________________".format(ss=seed))
    # sensitivityTable.loc[s,'stay:40min']=(sum(test_activityData['satisfying time limit'])/len(test_activityData['satisfying time limit']))#performance of time limit in trip identification
    # sensitivityTable.loc[s,'stay:60min']=(sum(test_activityData['satisfying time limit(1h)'])/len(test_activityData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
    # sensitivityTable.loc[s,'stay:Bayesian']=(sum(test_activityData['algorithm select activity'])/len(test_activityData['satisfying time limit']))#performance of Bayesian in trip identification
    # # print("____________________location,seed:{ss}  _____________________".format(ss=seed))
    # sensitivityTable.loc[s,'location:40min']=((sum(test_activityData['satisfying time limit'])+sum(test_tripData['satisfying time limit']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
    # sensitivityTable.loc[s,'location:60min']=((sum(test_activityData['satisfying time limit(1h)'])+sum(test_tripData['satisfying time limit(1h)']))/(len(test_tripData['satisfying time limit(1h)'])+len(test_activityData['satisfying time limit(1h)'])))
    # sensitivityTable.loc[s,'location:Bayesian']=((sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
    sens_locActivity.loc[s,'bw'] = bandwidth
    sens_locActivity.loc[s,'locationRecognition'] = (sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData)+len(test_activityData))
    sens_locActivity.loc[s,'stayRecognition'] = sum(test_activityData['algorithm select activity'])/len(test_activityData)
    sens_locActivity.loc[s,'tripRecognition'] = sum(test_tripData['algorithm select trip'])/len(test_tripData)
    sens_locActivity.loc[s,'activityRecognition'] = (len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data))
    sens_locActivity.loc[s,'overall'] = (sum(test_tripData['algorithm select trip'])+len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data)+len(test_tripData))

# sensitivityTable.to_excel('D:/progress meeting/15July2020(Hans&Adam)/onePercentLocationDetectionSensitivity.xlsx', header=True)
sens_locActivity.to_excel('D:/progress meeting/15July2020(Hans&Adam)/bwTest_onePercentLocationActivityDetectionSensitivity.xlsx',
                              header=True)
# ***********************************************************************************************************************
