
######################################### importing XML file plan ######################################################
from lxml import etree
parser = etree.XMLParser(ns_clean=True, collect_ids=False)
# itemlistPlan = etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                         "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml").getroot().findall('person')
itemlistExperienced= etree.parse("D:/ax/gis/output_base/PlanWithOnlyCar_again_NoGeneric_NoZeroDuationActivity.xml").getroot().findall('person')
######################################### deriving activity duration and traveling duration from plan files ###########################################
import numpy as np
import pandas as pd
import time
import datetime
import statistics as st
from duration import(to_seconds, to_tuple)
from matplotlib import pyplot as plt
from decimal import *
# person = itemlistPlan[0]
m=0
start_time = time.time()
activityDurations = []
activityDurations_total = []
activityAverageDurations = []
travelDuration = []
# negs = []
for m, person in enumerate(itemlistExperienced): # seconds for 1000 itemlist
    # activityListPlan = itemlistPlan[m].findall('plan/activity')
    activityListExperienced = itemlistExperienced[m].findall('plan/activity')
    if len(activityListExperienced)>2:
        durations = [to_seconds(activityListExperienced[0].get('end_time'), strict= False).__int__()- to_seconds(activityListExperienced[-1].get('start_time'), strict= False).__int__()]
        #todo please enter simulation begining time for the fist activity duration
        travelDur = [to_seconds(activityListExperienced[1].get('start_time'), strict= False).__int__()-
                     to_seconds(activityListExperienced[0].get('end_time'), strict= False).__int__()]
        j=1
        while j < len(activityListExperienced) - 1:
            durations = durations + [to_seconds(activityListExperienced[j].get('end_time'), strict= False).__int__() -
                                     to_seconds(activityListExperienced[j].get('start_time'),strict= False).__int__()]
            travelDur += [to_seconds(activityListExperienced[j+1].get('start_time'), strict= False).__int__()-
                     to_seconds(activityListExperienced[j].get('end_time'), strict= False).__int__()]
            j+=1

        # durations = durations + [to_seconds('25:59:59', strict=False).__int__() -
        #                          to_seconds(activityListExperienced[j].get('start_time'), strict=False).__int__()]
        # todo please enter simulation finish time for the last activity duration
        # if(min(durations)<0):
        #     negs += [m, itemlistExperienced[m].get('id') ]
        activityAverageDurations += [st.mean(durations)]
        # activityDurations = activityDurations + [durations]
        activityDurations_total += durations
        travelDuration += travelDur
travelDuration = pd.DataFrame(travelDuration, columns=['TravelingDuration(min)'])
activityDurations_total = pd.DataFrame(activityDurations_total, columns=['duration(sec)'])
fontDictAxis = {'family':'serif',
                'style':'normal',
                'size':'10',
                'color':  'black',
                'weight': 'normal',
                }
fontDictLabel = {'family':'serif',
                'style':'normal',
                'size':'14',
                'color':  'black',
                'weight': 'normal',
                }
fontDictTitle = {'family':'serif',
                'style':'normal',
                'size':'14',
                'color':  'black',
                'weight': 'bold',
                }
# _________________________________________ Bar plot of activity durations _____________________________________________
plt.figure()
activityDurations_total['bins_cut(5)'] = pd.cut(activityDurations_total['duration(sec)'], bins =
[0 ,300, 600,900,1200,1500,1800,2100,2400,2700, 3600,86400], include_lowest = True, labels=['AD>5|TD<5','AD>10|TD<10'
    ,'AD>15|TD<15','AD>20|TD<20', 'AD>25|TD<25', 'AD>30|TD<30', 'AD>35|TD<35','AD>40|TD<40','AD>45|TD<45', 'AD>60|TD<60','...'] )
activityDurations_total['bins_cut(5)'].value_counts(sort=False, normalize=True).cumsum().plot(kind = 'bar')
plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0)
plt.gca().patch.set_facecolor('0.8')
for index,data in enumerate(activityDurations_total['bins_cut(5)'].value_counts(sort=False, normalize=True).cumsum()):
    plt.text(x=index-.1 , y =data+.01 , s="{0:.1f}%".format(data*100) , fontdict=fontDictAxis)
plt.xticks(rotation = 0, horizontalalignment = 'center')
plt.xlabel('activity duration ranges(minutes)', fontdict = fontDictLabel)
plt.ylabel('proportion of agents', fontdict = fontDictLabel)
plt.title('bar plot of activity duration range frequency.', fontdict = fontDictTitle)
plt.gca().set_axisbelow(True)
plt.setp(plt.gca().get_xticklabels(), fontsize = 12)
plt.tight_layout()



# ______________________________________________________________________________________________________________________
# _________________________________________ Bar plot of traveling durations ____________________________________________
plt.figure()
travelDuration['bins_cut(9)'] = pd.cut(travelDuration['TravelingDuration(min)'], bins =
[0 ,300, 600,900,1200,1500,1800,2100,2400,2700, 3600,86400], include_lowest = True, labels=['AD>5|TD<5','AD>10|TD<10'
    ,'AD>15|TD<15','AD>20|TD<20', 'AD>25|TD<25', 'AD>30|TD<30', 'AD>35|TD<35','AD>40|TD<40','AD>45|TD<45', 'AD>60|TD<60','...'] )
travelDuration['bins_cut(9)'].value_counts(sort=False, normalize=True).cumsum().plot(kind = 'bar')
plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0)
plt.gca().patch.set_facecolor('0.8')
for index,data in enumerate(travelDuration['bins_cut(9)'].value_counts(sort=False, normalize=True).cumsum()):
    plt.text(x=index-.1 , y =data+.01 , s="{0:.1f}%".format(data*100) , fontdict=fontDictAxis)
plt.xticks(rotation = 0, horizontalalignment = 'center')
plt.xlabel('traveling duration ranges(minutes)', fontdict = fontDictLabel)
plt.ylabel('proportion of agents', fontdict = fontDictLabel)
plt.title('bar plot of traveling duration range frequency.', fontdict = fontDictTitle)
plt.gca().set_axisbelow(True)
plt.setp(plt.gca().get_xticklabels(), fontsize = 12)
plt.tight_layout()
# ______________________________________________________________________________________________________________________
# _________________________******* subploting  Bar plot of traveling durations *********________________________________
plt.figure()
testData1 = 1-pd.DataFrame(activityDurations_total['bins_cut(5)'].value_counts(sort=False, normalize=True).cumsum())
testData1['categories'] = testData1.index
testData2 = pd.DataFrame(travelDuration['bins_cut(9)'].value_counts(sort=False, normalize=True).cumsum())
testData2['categories'] = testData2.index
testData1['traveling_duration'] = testData2['bins_cut(9)']
testData1 = testData1.rename(columns={'bins_cut(5)':'activity_duration'})
# X = testData1['categories']
# Y = testData1['traveling_duration']
# Y2 = testData1['activity_duration']
# Ysum = Y+Y2
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()
# new_tick_locations = np.array(['>5', '>10', '>15', '>20','>25', '>30', '>45', '>60', '>24our'])
# testData1.traveling_duration.plot(kind='bar',color = 'red', ax=ax1)
# testData1.activity_duration.plot(kind='bar',color = 'blue', ax=ax2)
# ax2.set_xticklabels(new_tick_locations)
# ax1.set_xlabel('traveling_duration')
# ax2.set_xlabel('activity_duration')
# ax1.plot(X,Y, kind='bar')
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticklabels(new_tick_locations)
# ax2.set_xlabel(r"activity duration range")
# ax2.plot(X, Y2)
# plt.grid(True, color='lightgray', linestyle='-', linewidth=2, zorder = 0)
# plt.gca().patch.set_facecolor('0.8')
# plt.gca().set_facecolor('xkcd:salmon')
# plt.show()
testData1.plot.bar(figsize = (12, 10))
plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0)
# plt.figure(figsize=(9, 4.5))
plt.gca().patch.set_facecolor('0.8')
for index in range(len(testData1['activity_duration'])-1):
    plt.text(x=index-.25 , y =testData1['activity_duration'][index]+.01 , s="{0:.1f}%".
             format(testData1['activity_duration'][index]*100) , fontdict=fontDictAxis)
    plt.text(x=index , y =testData1['traveling_duration'][index]+.01 , s="{0:.1f}%".
             format(testData1['traveling_duration'][index]*100) , fontdict=fontDictAxis)
plt.xlabel('activity/traveling duration ranges(minutes)', fontdict = fontDictLabel)
plt.xticks(rotation = 30, horizontalalignment = 'center')
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.ylabel('proportion of all activities/travelings', fontdict = fontDictLabel)
plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0, which='both')
plt.gca().patch.set_facecolor('0.8')
plt.gca().set_axisbelow(True)
plt.savefig("D:/ax/gis/plots/kde_test/activityTrip_durationRange.png",dpi = 150)
plt.savefig("D:/ax/gis/plots/kde_test/activityTrip_durationRange.pdf",dpi = 150)
# ______________________________________________________________________________________________________________________
# _________________________________________ Box plot of activity durations _____________________________________________
plt.figure()
data = [activityDurations_total['duration(sec)']/60,travelDuration['TravelingDuration(min)']/60 ]
fig7, ax7 = plt.subplots()
ax7.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
            notch= False,
            capprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            flierprops=dict(markerfacecolor = 'r', markeredgecolor='black', marker = 's'),
            medianprops=dict(color='darkblue'), vert=True, showfliers=True, labels=['activity','traveling'])
plt.ylim(0,600)
plt.ylabel('duration(min)', fontsize = 12)
plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0, which='both')
plt.gca().patch.set_facecolor('0.8')
plt.gca().set_axisbelow(True)




plt.boxplot(activityDurations_total['duration(sec)']/3600, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
            notch= False,
            capprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            flierprops=dict(markerfacecolor = 'r', markeredgecolor='black', marker = 's'),
            medianprops=dict(color='darkblue'), vert=True, showfliers=False)
plt.ylabel('activity duration (hours)', fontdict = fontDictLabel)
plt.title('box plot of activity duration of agents.', fontdict = fontDictTitle)
plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0, which='both')
plt.gca().patch.set_facecolor('0.8')
plt.gca().set_axisbelow(True)
# ______________________________________________________________________________________________________________________
# _________________________________________ Box plot of traveling durations ____________________________________________
plt.figure()
plt.boxplot(travelDuration['TravelingDuration(min)']/60, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
            notch= False,
            capprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            flierprops=dict(markerfacecolor = 'r', markeredgecolor='black', marker = 's'),
            medianprops=dict(color='darkblue'), vert=True, showfliers=False)
plt.ylabel('traveling duration (hours)', fontdict = fontDictLabel)
plt.title('box plot of traveling duration of agents.', fontdict = fontDictTitle)
plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0, which='both')
plt.gca().patch.set_facecolor('0.8')
plt.gca().set_axisbelow(True)
# _____________________________________________BOXPLOT of random 50 seeds____________________________________________________________
userSampling = pd.read_excel('D:/ax/gis/plots/kde_test/usersamplingLocActivity.xlsx', sheet_name='Sheet1')
# # plt.figure()
# plt.boxplot(userSampling.loc[25:50,'overall'], patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
#             notch= False,
#             capprops=dict(color='black'),
#             whiskerprops=dict(color='black'),
#             flierprops=dict(markerfacecolor = 'r', markeredgecolor='black', marker = 's'),
#             medianprops=dict(color='darkblue'), vert=True, showfliers=False)
# plt.ylabel('activity-travel detection accuracy', fontdict = fontDictLabel)
# plt.title('box plot of activity-travel detection.', fontdict = fontDictTitle)
# plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0, which='both')
# plt.gca().patch.set_facecolor('0.8')
# plt.gca().set_axisbelow(True)
# # plt.figure()
# plt.boxplot(userSampling.loc[0:25,'overall'], patch_artist=True, boxprops=dict(facecolor='red', color='black'),
#             notch= False,
#             capprops=dict(color='black'),
#             whiskerprops=dict(color='black'),
#             flierprops=dict(markerfacecolor = 'r', markeredgecolor='black', marker = 's'),
#             medianprops=dict(color='darkblue'), vert=True, showfliers=False)
# plt.ylabel('activity-travel detection accuracy', fontdict = fontDictLabel)
# plt.title('box plot of activity-travel detection.', fontdict = fontDictTitle)
# plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0, which='both')
# plt.gca().patch.set_facecolor('0.8')
# plt.gca().set_axisbelow(True)



