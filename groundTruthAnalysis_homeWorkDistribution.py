
######################################### importing XML file plan ######################################################
from lxml import etree
parser = etree.XMLParser(ns_clean=True, collect_ids=False)
# itemlistPlan = etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                         "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml").getroot().findall('person')
itemlist= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
                               "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml").getroot().findall('person')
# itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                                "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans.xml").getroot()
itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
                               "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans.xml").getroot().findall('person')
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
# m=0
start_time = time.time()
homeloc = []
workloc = []
# homeDurations = []
# workDurations = []
######################TEST###############################################################################
firstActivityType = []
start_time_all = []
end_time_all = []
# same1stlastActivity = []
# same2stlastActivity = []
# indices = set(range(len(itemlistExperienced)))-set(same1stlastActivity)
# for m in indices :
#     if itemlistExperienced[m].findall('plan/activity')[0].get('type') !=itemlistExperienced[m].findall('plan/activity')[-1].get('type'):
#         # print(m)
#         # same1stlastActivity+= [m]
        # same2stlastActivity += [itemlistExperienced[m].get('id')]
        # same2stlastActivity+= [m]
        # np.delete(range(len(itemlistExperienced)), same1stlastActivity)
    # firstActivityType += [itemlistExperienced[m].find('plan/activity').get('type')] # result: array(['business', 'home', 'leisure', 'sozializing', 'touring'], dtype='<U11')

    # start_time_all += [itemlistExperienced[m].findall('plan/activity')[-1].get('start_time')] # result: max=25:44:18, min=05:43:12
    # end_time_all += [itemlistExperienced[m].findall('plan/activity')[0].get('end_time')] # result: max=23:26:56, min=02:20:48

# start_time_all_D = pd.DataFrame(start_time_all)
# end_time_all_D = pd.DataFrame(end_time_all)
##########################################################################################################
# testVar = itemlistExperienced.findall('person')
same1stlastActivity = []
for m in range(len(itemlistExperienced)) : #removing users that the fist activity type and the last one is not the same(only 10)
    if itemlistExperienced[m].findall('plan/activity')[0].get('type') !=itemlistExperienced[m].findall('plan/activity')[-1].get('type'):
        same1stlastActivity+= [m]
deducted_usersNum = len(same1stlastActivity)
homeDurations = []
workDurations = []
homeStarts = []
workStarts = []
otherDurations = []
otherStarts = []
indices = set(range(len(itemlistExperienced)))-set(same1stlastActivity)
m=8
for m in indices:
    # end =to_seconds( itemlistExperienced[m].find('plan/activity').get('end_time'),strict=False)+24*3600#end of first activity in the next day
    end = to_seconds(itemlistExperienced[m].find('plan/activity').get('end_time'),
                     strict=False) + 24 * 3600  # end of first activity in the next day
    # firstActivity = itemlistExperienced[m].find('plan/activity').get('type')
    firstActivity = itemlistExperienced[m].find('plan/activity').get('type')
    # itemlistExperienced[m].xpath('plan/activity[attribute::type]')[0].attrib['type'] != 'home'
    otherIndices = []
    for n in range(len(itemlistExperienced[m].xpath('plan/activity'))):
        if itemlistExperienced[m].xpath('plan/activity[attribute::type]')[n].attrib['type'] != 'home'\
                and itemlistExperienced[m].xpath('plan/activity[attribute::type]')[n].attrib['type'] != 'work':
            otherIndices+=[int(n)]

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
    # homeDurations += [to_seconds(home[0].get('end_time'), strict= False).__int__()-
    #                   to_seconds('02:00:00').__int__()] #todo please enter simulation begining time for the fist activity duration
    # workDuration += [to_seconds(work[0].get('end_time'), strict= False).__int__()-
    #                  to_seconds(work[0].get('start_time'), strict= False).__int__()]

    j=0
    while j < len(home):
        if to_seconds( home[j].get('start_time'), strict=False).__int__()==7200:
            print([itemlistExperienced[m].get('id')])
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

    # __________________________________________________________________________________________
    homeloc += [Point(float(itemlist[m].find('plan/activity[@type="home"]').get('x')),
                     float(itemlist[m].find('plan/activity[@type="home"]').get('y'))) if itemlist[m].find('plan/activity[@type="home"]')!= None else None]
    workloc += [Point(float(itemlist[m].find('plan/activity[@type="work"]').get('x')),
                     float(itemlist[m].find('plan/activity[@type="work"]').get('y'))) if itemlist[m].find('plan/activity[@type="work"]') != None else None]

print(time.time() - start_time)
homeDurations = pd.DataFrame(homeDurations, columns=['duration(sec)'])
workDurations = pd.DataFrame(workDurations, columns=['duration(sec)'])
otherDurations = pd.DataFrame(otherDurations, columns=['duration(sec)'])
workStarts = pd.DataFrame(workStarts, columns=['start_time(sec)'])
homeStarts = pd.DataFrame(homeStarts, columns=['start_time(sec)'])
otherStarts = pd.DataFrame(otherStarts, columns=['start_time(sec)'])
homeloc = pd.DataFrame(homeloc, columns=['lon-lat'])
workloc = pd.DataFrame(workloc, columns=['lon-lat'])
nTotalActivity = len(etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
                               "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans.xml").getroot().findall('person/plan/activity'))-deducted_usersNum
nHomeActivity = len(homeDurations)
nWorkActivity = len(workDurations)
nOtherActivity = len(otherDurations)
# fontDictAxis = {'family':'serif',
#                 'style':'normal',
#                 'size':'10',
#                 'color':  'black',
#                 'weight': 'normal',
#                 }
# fontDictLabel = {'family':'serif',
#                 'style':'normal',
#                 'size':'14',
#                 'color':  'black',
#                 'weight': 'normal',
#                 }
# fontDictTitle = {'family':'serif',
#                 'style':'normal',
#                 'size':'14',
#                 'color':  'black',
#                 'weight': 'bold',
#                 }
# # _________________________________________ Bar plot of homeStay durations _____________________________________________
# plt.figure()
# homeDurations['bins_cut(5min)'] = pd.cut(homeDurations['duration(sec)'],bins =288, include_lowest = True )
# homeDurations['bins_cut(5min)'].value_counts(sort=False, normalize=True).plot(kind = 'bar')
# plt.grid(True, color='w', linestyle='-', linewidth=2, zorder = 0)
# plt.gca().patch.set_facecolor('0.8')
# # for index,data in enumerate(homeDurations['bins_cut(5min)'].value_counts(sort=False, normalize=True)):
# #     plt.text(x=index-.1 , y =data+.01 , s="{0:.1f}%".format(data*100) , fontdict=fontDictAxis)
# plt.xticks(rotation = 0, horizontalalignment = 'center')
# plt.xlabel('home duration ranges(0 to 24 hours)', fontdict = fontDictLabel)
# plt.ylabel('proportion of agents', fontdict = fontDictLabel)
# plt.title('bar plot of home stay duration range frequency.', fontdict = fontDictTitle)
# plt.gca().set_axisbelow(True)
# plt.setp(plt.gca().get_xticklabels(), fontsize = 12)
# plt.tight_layout()
# #___________________________________ best fit _____________________________________#
# import matplotlib.pyplot as plt
# import scipy
# import scipy.stats
# from scipy.stats import gamma
# import numpy as np
# from scipy.stats import gaussian_kde
# plt.style.use('ggplot')
# y = scipy.int_((homeDurations['duration(sec)'][homeDurations['duration(sec)']>0]))
# h = plt.hist(y, bins=np.linspace(0,86400, 288), density= True)
# dist_names = ['norm', 'halflogistic', 'halfnorm']
# x = np.linspace(0,86400,288)
# #!! empty histogram #todo: fix this
# a = 1.99
# fit_alpha, fit_loc, fit_beta = gamma.fit(y, floc=0)
# pdf_fitted = gamma.pdf(x, fit_alpha, fit_loc, fit_beta)
# plt.plot(x, pdf_fitted, label='gamma')
# plt.xlim(0, len(homeDurations['duration(sec)']))
# plt.legend(loc='upper right')
# # dist_name = dist_names[3]
# for dist_name in dist_names:
#     dist = getattr(scipy.stats, dist_name)
#     param = dist.fit(y)
#     pdf_fitted = dist.pdf(x, loc=param[0], scale=param[1]) # s=param[1]
#     plt.plot(x,pdf_fitted, label=dist_name)
#     plt.xlim(0,len(homeDurations['duration(sec)']))
#     plt.legend(loc='upper right')
# plt.show()

