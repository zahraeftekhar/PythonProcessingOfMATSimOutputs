
######################################### importing XML file plan ######################################################
from lxml import etree
parser = etree.XMLParser(ns_clean=True, collect_ids=False)
itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
                               "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans_Nogeneric(all allowed).xml").getroot().findall('person')
######################################### deriving activity duration and traveling duration from plan files ###########################################
import numpy as np
import pandas as pd
import time
import datetime
import statistics as st
from duration import(to_seconds, to_tuple)
from matplotlib import pyplot as plt
from decimal import *
from shapely.geometry import Point
import random
from scipy.stats import gaussian_kde
# m=0
start_time = time.time()
# homeDurations = []
# workDurations = []
# homeStarts = []
# workStarts = []
# otherDurations = []
# otherStarts = []
# tripStarts = []
# tripDurations = []
sensitivityTable = pd.DataFrame(columns=['seed','pass-by:Bayesian','pass-by:40min','pass-by:60min', 'stay:Bayesian','stay:40min','stay:60min', 'location:Bayesian','location:40min','location:60min'])
sens_locActivity = pd.DataFrame(columns=['seed','locationRecognition','activityRecognition','overall'])
# _______________test seed based on user sampling _________________________________
# seedSet = range(101,151)
# seed = 101
# s=0
# for s, seed in enumerate(seedSet):
#     print(s, end='______ time:{tt}sec\n'.format(tt=time.time() - start_time))
#     homeDurations = []
#     workDurations = []
#     homeStarts = []
#     workStarts = []
#     otherDurations = []
#     otherStarts = []
#     tripStarts = []
#     tripDurations = []
#     random.seed(seed)
#     indices = random.sample(range(len(itemlistExperienced)),round(.01*len(itemlistExperienced))) #1% sampling of users
#     for index,m in enumerate(indices):#indices
#         end = to_seconds(itemlistExperienced[m].find('plan/activity').get('end_time'),
#                          strict=False) + 24 * 3600  # end of first activity in the next day
#         firstActivity = itemlistExperienced[m].find('plan/activity').get('type')
#         otherIndices = []
#         for n in range(len(itemlistExperienced[m].xpath('plan/activity'))):
#             if itemlistExperienced[m].xpath('plan/activity[attribute::type]')[n].attrib['type'] != 'home'\
#                     and itemlistExperienced[m].xpath('plan/activity[attribute::type]')[n].attrib['type'] != 'work':
#                 otherIndices+=[int(n)]
#         for p in range(len(itemlistExperienced[m].xpath('plan/leg'))):
#             tripStarts += [to_seconds( itemlistExperienced[m].xpath('plan/leg')[p].get('dep_time'), strict=False).__int__()]
#             tripDurations += [to_seconds(pd.Timedelta(itemlistExperienced[m].xpath('plan/leg')[p].get('trav_time')), strict=False).__int__()]
#
#         if firstActivity == 'home':
#             home = itemlistExperienced[m].findall('plan/activity[@type="home"]')[1:]
#             work = itemlistExperienced[m].findall('plan/activity[@type="work"]')
#             other = [itemlistExperienced[m].findall('plan/activity')[q] for q in otherIndices]
#         elif firstActivity == 'work':
#             home = itemlistExperienced[m].findall('plan/activity[@type="home"]')
#             work = itemlistExperienced[m].findall('plan/activity[@type="work"]')[1:]
#             other = [itemlistExperienced[m].findall('plan/activity')[q] for q in otherIndices]
#         else:
#             print(itemlistExperienced[m].get('id'),[m])
#             home = itemlistExperienced[m].findall('plan/activity[@type="home"]')
#             work = itemlistExperienced[m].findall('plan/activity[@type="work"]')
#             other = [itemlistExperienced[m].findall('plan/activity')[q] for q in otherIndices[1:]]
#
#         j=0
#         while j < len(home):
#             homeStarts += [to_seconds( home[j].get('start_time'), strict=False).__int__()]
#             homeDurations += [to_seconds((end if home[j].get('end_time') is None else  home[j].get('end_time')), strict=False).__int__() - \
#                               to_seconds(home[j].get('start_time') , strict=False).__int__()]
#             j+=1
#         k=0
#         while k < len(work):
#             workStarts += [ to_seconds(work[k].get('start_time'), strict=False).__int__()]
#             workDurations += [to_seconds((end if work[k].get(
#                 'end_time') is None else work[k].get('end_time')), strict=False).__int__() - \
#                              to_seconds(work[k].get('start_time'), strict=False).__int__()]
#             k += 1
#         l = 0
#         while l < len(other):
#             otherStarts += [to_seconds(other[l].get('start_time'), strict=False).__int__()]
#             otherDurations += [to_seconds((end if other[l].get(
#                 'end_time') is None else other[l].get('end_time')), strict=False).__int__() - \
#                               to_seconds(other[l].get('start_time'), strict=False).__int__()]
#             l += 1
#     homeDurations = pd.DataFrame(homeDurations, columns=['duration(sec)'])
#     workDurations = pd.DataFrame(workDurations, columns=['duration(sec)'])
#     otherDurations = pd.DataFrame(otherDurations, columns=['duration(sec)'])
#     tripDurations = pd.DataFrame(tripDurations, columns=['duration(sec)'])
#     workStarts = pd.DataFrame(workStarts, columns=['start_time(sec)'])
#     homeStarts = pd.DataFrame(homeStarts, columns=['start_time(sec)'])
#     otherStarts = pd.DataFrame(otherStarts, columns=['start_time(sec)'])
#     tripStarts = pd.DataFrame(tripStarts, columns=['start_time(sec)'])
#     activityTypesHome = pd.DataFrame(index=range(len(homeStarts)))
#     activityTypesWork = pd.DataFrame(index=range(len(workStarts)))
#     activityTypesOther = pd.DataFrame(index=range(len(otherStarts)))
#     activityTypesHome['type'] = 'home'
#     activityTypesWork['type'] = 'work'
#     activityTypesOther['type']='other'
#     activityTypes = pd.concat([activityTypesHome['type'],activityTypesWork['type'],activityTypesOther['type']],axis=0,ignore_index=True)
#     activityTypes = pd.DataFrame(activityTypes, columns=['type'])
#     activityDurations = pd.concat([homeDurations['duration(sec)'],workDurations['duration(sec)'],otherDurations['duration(sec)']],axis=0,ignore_index=True)
#     activityDurations = pd.DataFrame(activityDurations, columns = ['duration(sec)'])
#     activityStarts = pd.concat([homeStarts['start_time(sec)'],workStarts['start_time(sec)'],otherStarts['start_time(sec)']],axis=0,ignore_index=True)
#     activityStarts = pd.DataFrame(activityStarts , columns=['start_time(sec)'])
#     activityStarts['type'] = activityTypes['type']
#     # ________________________________________________________________________________________
#     # ****************************************************************************************
#     # ____________________ random USER sample as the training set from ALL users
#     train_activityData = pd.DataFrame()
#     train_activityData['Duration(hour)'] = activityDurations['duration(sec)'] / 3600
#     train_activityData['start(hour)'] = activityStarts['start_time(sec)'] / 3600
#     train_activityData['type'] = activityStarts['type']
#     # ____________________ selecting random trip sample as the training set from ALL trips
#     train_tripData = pd.DataFrame()
#     train_tripData['Duration(hour)'] = tripDurations['duration(sec)'] / 3600
#     train_tripData['start(hour)'] = tripStarts['start_time(sec)'] / 3600
#     # ____________________ random WORK sample as the training set from ALL activities
#     train_workData = pd.DataFrame()
#     train_workData['Duration(hour)'] = workDurations['duration(sec)'] / 3600
#     train_workData['start(hour)'] = workStarts['start_time(sec)'] / 3600
#     # ____________________ random HOME sample as the training set from ALL activities
#     train_homeData = pd.DataFrame()
#     train_homeData['Duration(hour)'] = (homeDurations['duration(sec)']) / 3600
#     train_homeData['start(hour)'] = (homeStarts['start_time(sec)']) / 3600
#     # ____________________ random OTHER sample as the training set from ALL activities
#     train_otherData = pd.DataFrame()
#     train_otherData['Duration(hour)'] = otherDurations['duration(sec)'] / 3600
#     train_otherData['start(hour)'] = otherStarts['start_time(sec)'] / 3600
#     # ****************************************************************************************
#     # ______________ importing required data _________________________
#     tripStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.tripStarts.csv")
#     activityStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.activityStarts.csv")
#     tripDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.tripDurations.CSV")
#     activityDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.activityDurations.CSV")
#     homeStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.homeStarts.CSV")
#     workStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.workStarts.CSV")
#     otherStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.otherStarts.CSV")
#     homeDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.homeDurations.CSV")
#     workDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.workDurations.CSV")
#     otherDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.otherDurations.CSV")
#     # ________________________________________________________________
#     test_activityData = pd.DataFrame()
#     test_activityData['Duration(hour)'] = activityDurations['duration(sec)']/3600
#     test_activityData['start(hour)'] = activityStarts['start_time(sec)']/3600
#     test_activityData['type'] = activityStarts['type']
#     nactivity = len(train_activityData)
#     # ____________________ selecting random trip sample as the training set from ALL trips
#     test_tripData = pd.DataFrame()
#     test_tripData['Duration(hour)'] = tripDurations['duration(sec)']/3600
#     test_tripData['start(hour)'] = tripStarts['start_time(sec)']/3600
#     ntrip = len(train_tripData)
#     # ____________________ random WORK sample as the training set from ALL activities
#     test_workData = pd.DataFrame()
#     test_workData['Duration(hour)'] = workDurations['duration(sec)']/3600
#     test_workData['start(hour)'] = workStarts['start_time(sec)']/3600
#     nWork = len(train_workData)
#     # ____________________ random HOME sample as the training set from ALL activities
#     test_homeData = pd.DataFrame()
#     test_homeData['Duration(hour)'] = (homeDurations['duration(sec)'])/3600
#     test_homeData['start(hour)'] = (homeStarts['start_time(sec)'])/3600
#     nHome = len(train_homeData)
#     # ____________________ random OTHER sample as the training set from ALL activities
#     test_otherData = pd.DataFrame()
#     test_otherData['Duration(hour)'] = otherDurations['duration(sec)']/3600
#     test_otherData['start(hour)'] = otherStarts['start_time(sec)']/3600
#     nOther = len(train_otherData)
#     # ___________________________ Probability calculations_________________________________________
#     #***************************************************************************
#     prior_activity = nactivity/(nactivity + ntrip)
#     prior_trip = ntrip/(nactivity + ntrip)
#     test_activityData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)'],bw_method=0.1).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)'],bw_method=0.1).pdf(test_activityData['start(hour)']))
#     test_activityData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)'],bw_method=0.1).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)'],bw_method=0.1).pdf(test_activityData['start(hour)']))
#     test_activityData['algorithm select activity'] = 0
#     p=0
#     for p in test_activityData.index:
#         if (test_activityData['Log prob. of being activity']>=test_activityData['Log prob. of being trip'])[p] :
#             test_activityData['algorithm select activity'][p] = 1
#     # print(sum(test_activityData['algorithm select activity'])/len(test_activityData['algorithm select activity']))
#     #***************************************************************************
#     test_tripData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)'],bw_method=0.1).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)'],bw_method=0.1).pdf(test_tripData['start(hour)']))
#     test_tripData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)'],bw_method=0.1).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)'],bw_method=0.1).pdf(test_tripData['start(hour)']))
#     test_tripData['algorithm select trip'] = 0
#     p=0
#     for p in test_tripData.index:
#         if (test_activityData['Log prob. of being activity']<=test_tripData['Log prob. of being trip'])[p] :
#             test_tripData['algorithm select trip'][p] = 1
#     # print(sum(test_tripData['algorithm select trip'])/len(test_tripData['algorithm select trip']))
#     #***************************************************************************
#     test_data = test_activityData[ test_activityData['algorithm select activity']>0]
#     prior_home = nHome/(nHome + nWork + nOther)
#     prior_work = nWork/(nHome + nWork + nOther)
#     prior_other = nOther/(nHome + nWork + nOther)
#     test_data['home'] = np.log(prior_home)+np.log( gaussian_kde(train_homeData['Duration(hour)']).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_homeData['start(hour)']).pdf(test_data['start(hour)']))
#     test_data['work'] = np.log(prior_work)+np.log( gaussian_kde(train_workData['Duration(hour)']).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_workData['start(hour)']).pdf(test_data['start(hour)']))
#     test_data['other'] = np.log(prior_other)+np.log( gaussian_kde(train_otherData['Duration(hour)']).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_otherData['start(hour)']).pdf(test_data['start(hour)']))
#     test_data['predictedActivity'] = test_data[['home','work','other']].idxmax(axis=1)
#     ####### adding time limit for location type identification #########
#     test_tripData['satisfying time limit'] = test_tripData['Duration(hour)']<.6666 #40 min as the time limit
#     test_tripData['satisfying time limit(1h)'] = test_tripData['Duration(hour)']<1 #60 min as the time limit
#     test_activityData['satisfying time limit'] = test_activityData['Duration(hour)']>=.6666 #40 min as the time limit
#     test_activityData['satisfying time limit(1h)'] = test_activityData['Duration(hour)']>=1 #60 min as the time limit
# # __________________________________________________________________________________________________________________________________________________________________________________
#     sensitivityTable.loc[s,'seed'] = seed
#     # print("____________________trip,seed:{ss} _____________________".format(ss=seed))
#     sensitivityTable.loc[s,'pass-by:Bayesian']=(sum(test_tripData['algorithm select trip'])/len(test_tripData['satisfying time limit']))#performance of Bayesian in trip identification
#     sensitivityTable.loc[s,'pass-by:40min']=(sum(test_tripData['satisfying time limit'])/len(test_tripData['satisfying time limit']))#performance of time limit in trip identification
#     sensitivityTable.loc[s,'pass-by:60min']=(sum(test_tripData['satisfying time limit(1h)'])/len(test_tripData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
#     # print("____________________stay,seed:{ss}  _____________________".format(ss=seed))
#     sensitivityTable.loc[s,'stay:40min']=(sum(test_activityData['satisfying time limit'])/len(test_activityData['satisfying time limit']))#performance of time limit in trip identification
#     sensitivityTable.loc[s,'stay:60min']=(sum(test_activityData['satisfying time limit(1h)'])/len(test_activityData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
#     sensitivityTable.loc[s,'stay:Bayesian']=(sum(test_activityData['algorithm select activity'])/len(test_activityData['satisfying time limit']))#performance of Bayesian in trip identification
#     # print("____________________location,seed:{ss}  _____________________".format(ss=seed))
#     sensitivityTable.loc[s,'location:40min']=((sum(test_activityData['satisfying time limit'])+sum(test_tripData['satisfying time limit']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
#     sensitivityTable.loc[s,'location:60min']=((sum(test_activityData['satisfying time limit(1h)'])+sum(test_tripData['satisfying time limit(1h)']))/(len(test_tripData['satisfying time limit(1h)'])+len(test_activityData['satisfying time limit(1h)'])))
#     sensitivityTable.loc[s,'location:Bayesian']=((sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
#     sens_locActivity.loc[s,'seed'] = seed
#     sens_locActivity.loc[s,'locationRecognition'] = (sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData)+len(test_activityData))
#     sens_locActivity.loc[s,'activityRecognition'] = (len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data))
#     sens_locActivity.loc[s,'overall'] = (sum(test_tripData['algorithm select trip'])+len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data)+len(test_tripData))
#
# sensitivityTable.to_excel('D:/progress meeting/15July2020(Hans&Adam)/userSampling_onePercentLocationDetectionSensitivity.xlsx', header=True)
# sens_locActivity.to_excel('D:/progress meeting/15July2020(Hans&Adam)/userSampling_onePercentLocationActivityDetectionSensitivity.xlsx',
#                               header=True)
# _______________________________test bw based on user sampling _________________________
bwSet = np.arange(.05,3.05,0.05)
# for s, bandwidth in enumerate(bwSet):
#     print(s, end='______ time:{tt}sec\n'.format(tt=time.time() - start_time))
#     homeDurations = []
#     workDurations = []
#     homeStarts = []
#     workStarts = []
#     otherDurations = []
#     otherStarts = []
#     tripStarts = []
#     tripDurations = []
#     seed = 101
#     random.seed(seed)
#     indices = random.sample(range(len(itemlistExperienced)),round(.01*len(itemlistExperienced))) #1% sampling of users
#     for index,m in enumerate(indices):#indices
#         end = to_seconds(itemlistExperienced[m].find('plan/activity').get('end_time'),
#                          strict=False) + 24 * 3600  # end of first activity in the next day
#         firstActivity = itemlistExperienced[m].find('plan/activity').get('type')
#         otherIndices = []
#         for n in range(len(itemlistExperienced[m].xpath('plan/activity'))):
#             if itemlistExperienced[m].xpath('plan/activity[attribute::type]')[n].attrib['type'] != 'home'\
#                     and itemlistExperienced[m].xpath('plan/activity[attribute::type]')[n].attrib['type'] != 'work':
#                 otherIndices+=[int(n)]
#         for p in range(len(itemlistExperienced[m].xpath('plan/leg'))):
#             tripStarts += [to_seconds( itemlistExperienced[m].xpath('plan/leg')[p].get('dep_time'), strict=False).__int__()]
#             tripDurations += [to_seconds(pd.Timedelta(itemlistExperienced[m].xpath('plan/leg')[p].get('trav_time')), strict=False).__int__()]
#
#         if firstActivity == 'home':
#             home = itemlistExperienced[m].findall('plan/activity[@type="home"]')[1:]
#             work = itemlistExperienced[m].findall('plan/activity[@type="work"]')
#             other = [itemlistExperienced[m].findall('plan/activity')[q] for q in otherIndices]
#         elif firstActivity == 'work':
#             home = itemlistExperienced[m].findall('plan/activity[@type="home"]')
#             work = itemlistExperienced[m].findall('plan/activity[@type="work"]')[1:]
#             other = [itemlistExperienced[m].findall('plan/activity')[q] for q in otherIndices]
#         else:
#             print(itemlistExperienced[m].get('id'),[m])
#             home = itemlistExperienced[m].findall('plan/activity[@type="home"]')
#             work = itemlistExperienced[m].findall('plan/activity[@type="work"]')
#             other = [itemlistExperienced[m].findall('plan/activity')[q] for q in otherIndices[1:]]
#
#         j=0
#         while j < len(home):
#             homeStarts += [to_seconds( home[j].get('start_time'), strict=False).__int__()]
#             homeDurations += [to_seconds((end if home[j].get('end_time') is None else  home[j].get('end_time')), strict=False).__int__() - \
#                               to_seconds(home[j].get('start_time') , strict=False).__int__()]
#             j+=1
#         k=0
#         while k < len(work):
#             workStarts += [ to_seconds(work[k].get('start_time'), strict=False).__int__()]
#             workDurations += [to_seconds((end if work[k].get(
#                 'end_time') is None else work[k].get('end_time')), strict=False).__int__() - \
#                              to_seconds(work[k].get('start_time'), strict=False).__int__()]
#             k += 1
#         l = 0
#         while l < len(other):
#             otherStarts += [to_seconds(other[l].get('start_time'), strict=False).__int__()]
#             otherDurations += [to_seconds((end if other[l].get(
#                 'end_time') is None else other[l].get('end_time')), strict=False).__int__() - \
#                               to_seconds(other[l].get('start_time'), strict=False).__int__()]
#             l += 1
#     homeDurations = pd.DataFrame(homeDurations, columns=['duration(sec)'])
#     workDurations = pd.DataFrame(workDurations, columns=['duration(sec)'])
#     otherDurations = pd.DataFrame(otherDurations, columns=['duration(sec)'])
#     tripDurations = pd.DataFrame(tripDurations, columns=['duration(sec)'])
#     workStarts = pd.DataFrame(workStarts, columns=['start_time(sec)'])
#     homeStarts = pd.DataFrame(homeStarts, columns=['start_time(sec)'])
#     otherStarts = pd.DataFrame(otherStarts, columns=['start_time(sec)'])
#     tripStarts = pd.DataFrame(tripStarts, columns=['start_time(sec)'])
#     activityTypesHome = pd.DataFrame(index=range(len(homeStarts)))
#     activityTypesWork = pd.DataFrame(index=range(len(workStarts)))
#     activityTypesOther = pd.DataFrame(index=range(len(otherStarts)))
#     activityTypesHome['type'] = 'home'
#     activityTypesWork['type'] = 'work'
#     activityTypesOther['type']='other'
#     activityTypes = pd.concat([activityTypesHome['type'],activityTypesWork['type'],activityTypesOther['type']],axis=0,ignore_index=True)
#     activityTypes = pd.DataFrame(activityTypes, columns=['type'])
#     activityDurations = pd.concat([homeDurations['duration(sec)'],workDurations['duration(sec)'],otherDurations['duration(sec)']],axis=0,ignore_index=True)
#     activityDurations = pd.DataFrame(activityDurations, columns = ['duration(sec)'])
#     activityStarts = pd.concat([homeStarts['start_time(sec)'],workStarts['start_time(sec)'],otherStarts['start_time(sec)']],axis=0,ignore_index=True)
#     activityStarts = pd.DataFrame(activityStarts , columns=['start_time(sec)'])
#     activityStarts['type'] = activityTypes['type']
#     # ________________________________________________________________________________________
#     # ****************************************************************************************
#     # ____________________ random USER sample as the training set from ALL users
#     train_activityData = pd.DataFrame()
#     train_activityData['Duration(hour)'] = activityDurations['duration(sec)'] / 3600
#     train_activityData['start(hour)'] = activityStarts['start_time(sec)'] / 3600
#     train_activityData['type'] = activityStarts['type']
#     # ____________________ selecting random trip sample as the training set from ALL trips
#     train_tripData = pd.DataFrame()
#     train_tripData['Duration(hour)'] = tripDurations['duration(sec)'] / 3600
#     train_tripData['start(hour)'] = tripStarts['start_time(sec)'] / 3600
#     # ____________________ random WORK sample as the training set from ALL activities
#     train_workData = pd.DataFrame()
#     train_workData['Duration(hour)'] = workDurations['duration(sec)'] / 3600
#     train_workData['start(hour)'] = workStarts['start_time(sec)'] / 3600
#     # ____________________ random HOME sample as the training set from ALL activities
#     train_homeData = pd.DataFrame()
#     train_homeData['Duration(hour)'] = (homeDurations['duration(sec)']) / 3600
#     train_homeData['start(hour)'] = (homeStarts['start_time(sec)']) / 3600
#     # ____________________ random OTHER sample as the training set from ALL activities
#     train_otherData = pd.DataFrame()
#     train_otherData['Duration(hour)'] = otherDurations['duration(sec)'] / 3600
#     train_otherData['start(hour)'] = otherStarts['start_time(sec)'] / 3600
#     # ****************************************************************************************
#     # ______________ importing required data _________________________
#     tripStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.tripStarts.csv")
#     activityStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.activityStarts.csv")
#     tripDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.tripDurations.CSV")
#     activityDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.activityDurations.CSV")
#     homeStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.homeStarts.CSV")
#     workStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.workStarts.CSV")
#     otherStarts=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.otherStarts.CSV")
#     homeDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.homeDurations.CSV")
#     workDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.workDurations.CSV")
#     otherDurations=pd.read_csv(
#         "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.otherDurations.CSV")
#     # ________________________________________________________________
#     test_activityData = pd.DataFrame()
#     test_activityData['Duration(hour)'] = activityDurations['duration(sec)']/3600
#     test_activityData['start(hour)'] = activityStarts['start_time(sec)']/3600
#     test_activityData['type'] = activityStarts['type']
#     nactivity = len(train_activityData)
#     # ____________________ selecting random trip sample as the training set from ALL trips
#     test_tripData = pd.DataFrame()
#     test_tripData['Duration(hour)'] = tripDurations['duration(sec)']/3600
#     test_tripData['start(hour)'] = tripStarts['start_time(sec)']/3600
#     ntrip = len(train_tripData)
#     # ____________________ random WORK sample as the training set from ALL activities
#     test_workData = pd.DataFrame()
#     test_workData['Duration(hour)'] = workDurations['duration(sec)']/3600
#     test_workData['start(hour)'] = workStarts['start_time(sec)']/3600
#     nWork = len(train_workData)
#     # ____________________ random HOME sample as the training set from ALL activities
#     test_homeData = pd.DataFrame()
#     test_homeData['Duration(hour)'] = (homeDurations['duration(sec)'])/3600
#     test_homeData['start(hour)'] = (homeStarts['start_time(sec)'])/3600
#     nHome = len(train_homeData)
#     # ____________________ random OTHER sample as the training set from ALL activities
#     test_otherData = pd.DataFrame()
#     test_otherData['Duration(hour)'] = otherDurations['duration(sec)']/3600
#     test_otherData['start(hour)'] = otherStarts['start_time(sec)']/3600
#     nOther = len(train_otherData)
#     # ___________________________ Probability calculations_________________________________________
#     #***************************************************************************
#     prior_activity = nactivity/(nactivity + ntrip)
#     prior_trip = ntrip/(nactivity + ntrip)
#     test_activityData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)'],bw_method=bandwidth).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)'],bw_method=bandwidth).pdf(test_activityData['start(hour)']))
#     test_activityData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)'],bw_method=bandwidth).pdf(test_activityData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)'],bw_method=bandwidth).pdf(test_activityData['start(hour)']))
#     test_activityData['algorithm select activity'] = 0
#     p=0
#     for p in test_activityData.index:
#         if (test_activityData['Log prob. of being activity']>=test_activityData['Log prob. of being trip'])[p] :
#             test_activityData['algorithm select activity'][p] = 1
#     # print(sum(test_activityData['algorithm select activity'])/len(test_activityData['algorithm select activity']))
#     #***************************************************************************
#     test_tripData['Log prob. of being activity'] = np.log(prior_activity)+np.log( gaussian_kde(train_activityData['Duration(hour)'],bw_method=bandwidth).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_activityData['start(hour)'],bw_method=bandwidth).pdf(test_tripData['start(hour)']))
#     test_tripData['Log prob. of being trip'] = np.log(prior_trip)+np.log( gaussian_kde(train_tripData['Duration(hour)'],bw_method=bandwidth).pdf(test_tripData['Duration(hour)']))+np.log( gaussian_kde(train_tripData['start(hour)'],bw_method=bandwidth).pdf(test_tripData['start(hour)']))
#     test_tripData['algorithm select trip'] = 0
#     p=0
#     for p in test_tripData.index:
#         if (test_activityData['Log prob. of being activity']<=test_tripData['Log prob. of being trip'])[p] :
#             test_tripData['algorithm select trip'][p] = 1
#     # print(sum(test_tripData['algorithm select trip'])/len(test_tripData['algorithm select trip']))
#     #***************************************************************************
#     test_data = test_activityData[ test_activityData['algorithm select activity']>0]
#     prior_home = nHome/(nHome + nWork + nOther)
#     prior_work = nWork/(nHome + nWork + nOther)
#     prior_other = nOther/(nHome + nWork + nOther)
#     test_data['home'] = np.log(prior_home)+np.log( gaussian_kde(train_homeData['Duration(hour)'],bw_method=bandwidth).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_homeData['start(hour)'],bw_method=bandwidth).pdf(test_data['start(hour)']))
#     test_data['work'] = np.log(prior_work)+np.log( gaussian_kde(train_workData['Duration(hour)'],bw_method=bandwidth).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_workData['start(hour)'],bw_method=bandwidth).pdf(test_data['start(hour)']))
#     test_data['other'] = np.log(prior_other)+np.log( gaussian_kde(train_otherData['Duration(hour)'],bw_method=bandwidth).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_otherData['start(hour)'],bw_method=bandwidth).pdf(test_data['start(hour)']))
#     test_data['predictedActivity'] = test_data[['home','work','other']].idxmax(axis=1)
#     ####### adding time limit for location type identification #########
#     test_tripData['satisfying time limit'] = test_tripData['Duration(hour)']<.6666 #40 min as the time limit
#     test_tripData['satisfying time limit(1h)'] = test_tripData['Duration(hour)']<1 #60 min as the time limit
#     test_activityData['satisfying time limit'] = test_activityData['Duration(hour)']>=.6666 #40 min as the time limit
#     test_activityData['satisfying time limit(1h)'] = test_activityData['Duration(hour)']>=1 #60 min as the time limit
# # __________________________________________________________________________________________________________________________________________________________________________________
# #     sensitivityTable.loc[s,'seed'] = seed
# #     # print("____________________trip,seed:{ss} _____________________".format(ss=seed))
# #     sensitivityTable.loc[s,'pass-by:Bayesian']=(sum(test_tripData['algorithm select trip'])/len(test_tripData['satisfying time limit']))#performance of Bayesian in trip identification
# #     sensitivityTable.loc[s,'pass-by:40min']=(sum(test_tripData['satisfying time limit'])/len(test_tripData['satisfying time limit']))#performance of time limit in trip identification
# #     sensitivityTable.loc[s,'pass-by:60min']=(sum(test_tripData['satisfying time limit(1h)'])/len(test_tripData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
# #     # print("____________________stay,seed:{ss}  _____________________".format(ss=seed))
# #     sensitivityTable.loc[s,'stay:40min']=(sum(test_activityData['satisfying time limit'])/len(test_activityData['satisfying time limit']))#performance of time limit in trip identification
# #     sensitivityTable.loc[s,'stay:60min']=(sum(test_activityData['satisfying time limit(1h)'])/len(test_activityData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
# #     sensitivityTable.loc[s,'stay:Bayesian']=(sum(test_activityData['algorithm select activity'])/len(test_activityData['satisfying time limit']))#performance of Bayesian in trip identification
# #     # print("____________________location,seed:{ss}  _____________________".format(ss=seed))
# #     sensitivityTable.loc[s,'location:40min']=((sum(test_activityData['satisfying time limit'])+sum(test_tripData['satisfying time limit']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
# #     sensitivityTable.loc[s,'location:60min']=((sum(test_activityData['satisfying time limit(1h)'])+sum(test_tripData['satisfying time limit(1h)']))/(len(test_tripData['satisfying time limit(1h)'])+len(test_activityData['satisfying time limit(1h)'])))
# #     sensitivityTable.loc[s,'location:Bayesian']=((sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
#     sens_locActivity.loc[s,'seed'] = seed
#     sens_locActivity.loc[s,'bw'] = bandwidth
#     sens_locActivity.loc[s,'locationRecognition'] = (sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData)+len(test_activityData))
#     sens_locActivity.loc[s,'stayRecognition'] = sum(test_activityData['algorithm select activity'])/len(test_activityData)
#     sens_locActivity.loc[s,'tripRecognition'] = sum(test_tripData['algorithm select trip'])/len(test_tripData)
#     sens_locActivity.loc[s,'activityRecognition'] = (len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data))
#     sens_locActivity.loc[s,'overall'] = (sum(test_tripData['algorithm select trip'])+len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data)+len(test_tripData))
# sensitivityTable.to_excel('D:/progress meeting/15July2020(Hans&Adam)/bwTest_userSampling_onePercentLocationDetectionSensitivity.xlsx', header=True)
# sens_locActivity.to_excel('D:/progress meeting/15July2020(Hans&Adam)/bwTest_userSampling_onePercentLocationActivityDetectionSensitivity.xlsx',header=True)

# ******************test for best values of bw=0.1 and 2.35
homeDurations = []
workDurations = []
homeStarts = []
workStarts = []
otherDurations = []
otherStarts = []
tripStarts = []
tripDurations = []
seed = 101
random.seed(seed)
indices = random.sample(range(len(itemlistExperienced)),round(.01*len(itemlistExperienced))) #1% sampling of users
s=0
bandwidth = 0.1
bandwidth2 = 0.3
for index,m in enumerate(indices):#indices
    end = to_seconds(itemlistExperienced[m].find('plan/activity').get('end_time'),
                     strict=False) + 24 * 3600  # end of first activity in the next day
    firstActivity = itemlistExperienced[m].find('plan/activity').get('type')
    otherIndices = []
    for n in range(len(itemlistExperienced[m].xpath('plan/activity'))):
        if itemlistExperienced[m].xpath('plan/activity[attribute::type]')[n].attrib['type'] != 'home'\
                and itemlistExperienced[m].xpath('plan/activity[attribute::type]')[n].attrib['type'] != 'work':
            otherIndices+=[int(n)]
    for p in range(len(itemlistExperienced[m].xpath('plan/leg'))):
        tripStarts += [to_seconds( itemlistExperienced[m].xpath('plan/leg')[p].get('dep_time'), strict=False).__int__()]
        tripDurations += [to_seconds(pd.Timedelta(itemlistExperienced[m].xpath('plan/leg')[p].get('trav_time')), strict=False).__int__()]

    if firstActivity == 'home':
        home = itemlistExperienced[m].findall('plan/activity[@type="home"]')[1:]
        work = itemlistExperienced[m].findall('plan/activity[@type="work"]')
        other = [itemlistExperienced[m].findall('plan/activity')[q] for q in otherIndices]
    elif firstActivity == 'work':
        home = itemlistExperienced[m].findall('plan/activity[@type="home"]')
        work = itemlistExperienced[m].findall('plan/activity[@type="work"]')[1:]
        other = [itemlistExperienced[m].findall('plan/activity')[q] for q in otherIndices]
    else:
        print(itemlistExperienced[m].get('id'),[m])
        home = itemlistExperienced[m].findall('plan/activity[@type="home"]')
        work = itemlistExperienced[m].findall('plan/activity[@type="work"]')
        other = [itemlistExperienced[m].findall('plan/activity')[q] for q in otherIndices[1:]]

    j=0
    while j < len(home):
        homeStarts += [to_seconds( home[j].get('start_time'), strict=False).__int__()]
        homeDurations += [to_seconds((end if home[j].get('end_time') is None else  home[j].get('end_time')), strict=False).__int__() - \
                          to_seconds(home[j].get('start_time') , strict=False).__int__()]
        j+=1
    k=0
    while k < len(work):
        workStarts += [ to_seconds(work[k].get('start_time'), strict=False).__int__()]
        workDurations += [to_seconds((end if work[k].get(
            'end_time') is None else work[k].get('end_time')), strict=False).__int__() - \
                         to_seconds(work[k].get('start_time'), strict=False).__int__()]
        k += 1
    l = 0
    while l < len(other):
        otherStarts += [to_seconds(other[l].get('start_time'), strict=False).__int__()]
        otherDurations += [to_seconds((end if other[l].get(
            'end_time') is None else other[l].get('end_time')), strict=False).__int__() - \
                          to_seconds(other[l].get('start_time'), strict=False).__int__()]
        l += 1
homeDurations = pd.DataFrame(homeDurations, columns=['duration(sec)'])
workDurations = pd.DataFrame(workDurations, columns=['duration(sec)'])
otherDurations = pd.DataFrame(otherDurations, columns=['duration(sec)'])
tripDurations = pd.DataFrame(tripDurations, columns=['duration(sec)'])
workStarts = pd.DataFrame(workStarts, columns=['start_time(sec)'])
homeStarts = pd.DataFrame(homeStarts, columns=['start_time(sec)'])
otherStarts = pd.DataFrame(otherStarts, columns=['start_time(sec)'])
tripStarts = pd.DataFrame(tripStarts, columns=['start_time(sec)'])
activityTypesHome = pd.DataFrame(index=range(len(homeStarts)))
activityTypesWork = pd.DataFrame(index=range(len(workStarts)))
activityTypesOther = pd.DataFrame(index=range(len(otherStarts)))
activityTypesHome['type'] = 'home'
activityTypesWork['type'] = 'work'
activityTypesOther['type']='other'
activityTypes = pd.concat([activityTypesHome['type'],activityTypesWork['type'],activityTypesOther['type']],axis=0,ignore_index=True)
activityTypes = pd.DataFrame(activityTypes, columns=['type'])
activityDurations = pd.concat([homeDurations['duration(sec)'],workDurations['duration(sec)'],otherDurations['duration(sec)']],axis=0,ignore_index=True)
activityDurations = pd.DataFrame(activityDurations, columns = ['duration(sec)'])
activityStarts = pd.concat([homeStarts['start_time(sec)'],workStarts['start_time(sec)'],otherStarts['start_time(sec)']],axis=0,ignore_index=True)
activityStarts = pd.DataFrame(activityStarts , columns=['start_time(sec)'])
activityStarts['type'] = activityTypes['type']
# ________________________________________________________________________________________
# ****************************************************************************************
# ____________________ random USER sample as the training set from ALL users
train_activityData = pd.DataFrame()
train_activityData['Duration(hour)'] = activityDurations['duration(sec)'] / 3600
train_activityData['start(hour)'] = activityStarts['start_time(sec)'] / 3600
train_activityData['type'] = activityStarts['type']
# ____________________ selecting random trip sample as the training set from ALL trips
train_tripData = pd.DataFrame()
train_tripData['Duration(hour)'] = tripDurations['duration(sec)'] / 3600
train_tripData['start(hour)'] = tripStarts['start_time(sec)'] / 3600
# ____________________ random WORK sample as the training set from ALL activities
train_workData = pd.DataFrame()
train_workData['Duration(hour)'] = workDurations['duration(sec)'] / 3600
train_workData['start(hour)'] = workStarts['start_time(sec)'] / 3600
# ____________________ random HOME sample as the training set from ALL activities
train_homeData = pd.DataFrame()
train_homeData['Duration(hour)'] = (homeDurations['duration(sec)']) / 3600
train_homeData['start(hour)'] = (homeStarts['start_time(sec)']) / 3600
# ____________________ random OTHER sample as the training set from ALL activities
train_otherData = pd.DataFrame()
train_otherData['Duration(hour)'] = otherDurations['duration(sec)'] / 3600
train_otherData['start(hour)'] = otherStarts['start_time(sec)'] / 3600
# ****************************************************************************************
# ______________ importing required data _________________________
tripStarts=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.tripStarts.csv")
activityStarts=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.activityStarts.csv")
tripDurations=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.tripDurations.CSV")
activityDurations=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.activityDurations.CSV")
homeStarts=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.homeStarts.CSV")
workStarts=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.workStarts.CSV")
otherStarts=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.otherStarts.CSV")
homeDurations=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.homeDurations.CSV")
workDurations=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.workDurations.CSV")
otherDurations=pd.read_csv(
    "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.otherDurations.CSV")
# ________________________________________________________________
test_activityData = pd.DataFrame()
test_activityData['Duration(hour)'] = activityDurations['duration(sec)']/3600
test_activityData['start(hour)'] = activityStarts['start_time(sec)']/3600
test_activityData['type'] = activityStarts['type']
nactivity = len(train_activityData)
# ____________________ selecting random trip sample as the training set from ALL trips
test_tripData = pd.DataFrame()
test_tripData['Duration(hour)'] = tripDurations['duration(sec)']/3600
test_tripData['start(hour)'] = tripStarts['start_time(sec)']/3600
ntrip = len(train_tripData)
# ____________________ random WORK sample as the training set from ALL activities
test_workData = pd.DataFrame()
test_workData['Duration(hour)'] = workDurations['duration(sec)']/3600
test_workData['start(hour)'] = workStarts['start_time(sec)']/3600
nWork = len(train_workData)
# ____________________ random HOME sample as the training set from ALL activities
test_homeData = pd.DataFrame()
test_homeData['Duration(hour)'] = (homeDurations['duration(sec)'])/3600
test_homeData['start(hour)'] = (homeStarts['start_time(sec)'])/3600
nHome = len(train_homeData)
# ____________________ random OTHER sample as the training set from ALL activities
test_otherData = pd.DataFrame()
test_otherData['Duration(hour)'] = otherDurations['duration(sec)']/3600
test_otherData['start(hour)'] = otherStarts['start_time(sec)']/3600
nOther = len(train_otherData)
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
test_data['home'] = np.log(prior_home)+np.log( gaussian_kde(train_homeData['Duration(hour)'],bw_method=bandwidth2).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_homeData['start(hour)'],bw_method=bandwidth2).pdf(test_data['start(hour)']))
test_data['work'] = np.log(prior_work)+np.log( gaussian_kde(train_workData['Duration(hour)'],bw_method=bandwidth2).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_workData['start(hour)'],bw_method=bandwidth2).pdf(test_data['start(hour)']))
test_data['other'] = np.log(prior_other)+np.log( gaussian_kde(train_otherData['Duration(hour)'],bw_method=bandwidth2).pdf(test_data['Duration(hour)']))+np.log( gaussian_kde(train_otherData['start(hour)'],bw_method=bandwidth2).pdf(test_data['start(hour)']))
test_data['predictedActivity'] = test_data[['home','work','other']].idxmax(axis=1)
####### adding time limit for location type identification #########
test_tripData['satisfying time limit'] = test_tripData['Duration(hour)']<.6666 #40 min as the time limit
test_tripData['satisfying time limit(1h)'] = test_tripData['Duration(hour)']<1 #60 min as the time limit
test_activityData['satisfying time limit'] = test_activityData['Duration(hour)']>=.6666 #40 min as the time limit
test_activityData['satisfying time limit(1h)'] = test_activityData['Duration(hour)']>=1 #60 min as the time limit
# __________________________________________________________________________________________________________________________________________________________________________________
#     sensitivityTable.loc[s,'seed'] = seed
#     # print("____________________trip,seed:{ss} _____________________".format(ss=seed))
#     sensitivityTable.loc[s,'pass-by:Bayesian']=(sum(test_tripData['algorithm select trip'])/len(test_tripData['satisfying time limit']))#performance of Bayesian in trip identification
#     sensitivityTable.loc[s,'pass-by:40min']=(sum(test_tripData['satisfying time limit'])/len(test_tripData['satisfying time limit']))#performance of time limit in trip identification
#     sensitivityTable.loc[s,'pass-by:60min']=(sum(test_tripData['satisfying time limit(1h)'])/len(test_tripData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
#     # print("____________________stay,seed:{ss}  _____________________".format(ss=seed))
#     sensitivityTable.loc[s,'stay:40min']=(sum(test_activityData['satisfying time limit'])/len(test_activityData['satisfying time limit']))#performance of time limit in trip identification
#     sensitivityTable.loc[s,'stay:60min']=(sum(test_activityData['satisfying time limit(1h)'])/len(test_activityData['satisfying time limit(1h)']))#performance of time limit of 60min in trip identification
#     sensitivityTable.loc[s,'stay:Bayesian']=(sum(test_activityData['algorithm select activity'])/len(test_activityData['satisfying time limit']))#performance of Bayesian in trip identification
#     # print("____________________location,seed:{ss}  _____________________".format(ss=seed))
#     sensitivityTable.loc[s,'location:40min']=((sum(test_activityData['satisfying time limit'])+sum(test_tripData['satisfying time limit']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
#     sensitivityTable.loc[s,'location:60min']=((sum(test_activityData['satisfying time limit(1h)'])+sum(test_tripData['satisfying time limit(1h)']))/(len(test_tripData['satisfying time limit(1h)'])+len(test_activityData['satisfying time limit(1h)'])))
#     sensitivityTable.loc[s,'location:Bayesian']=((sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData['satisfying time limit'])+len(test_activityData['satisfying time limit'])))
# sens_locActivity.loc[s,'seed'] = seed
# sens_locActivity.loc[s,'bw'] = bandwidth
sens_locActivity.loc[s,'locationRecognition'] = (sum(test_tripData['algorithm select trip'])+sum(test_activityData['algorithm select activity']))/(len(test_tripData)+len(test_activityData))
sens_locActivity.loc[s,'stayRecognition'] = sum(test_activityData['algorithm select activity'])/len(test_activityData)
sens_locActivity.loc[s,'tripRecognition'] = sum(test_tripData['algorithm select trip'])/len(test_tripData)
sens_locActivity.loc[s,'activityRecognition'] = (len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data))
sens_locActivity.loc[s,'overall'] = (sum(test_tripData['algorithm select trip'])+len(test_data[test_data['predictedActivity']==test_data['type']]))/(len(test_data)+len(test_tripData))
print(sens_locActivity['activityRecognition'])
print(sens_locActivity['overall'])
