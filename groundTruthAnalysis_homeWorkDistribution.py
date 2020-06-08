
######################################### importing XML file plan ######################################################
from lxml import etree
parser = etree.XMLParser(ns_clean=True, collect_ids=False)
# itemlistPlan = etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                         "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml").getroot().findall('person')
itemlist= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
                               "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml").getroot().findall('person')
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
# activityDurations = []
# activityDurations_total = []
# activityAverageDurations = []
# travelDuration = []
homeloc = []
workloc = []
homeDurations = []
workDuration = []
######################TEST###############################################################################
firstActivityType = []
start_time_all = []
end_time_all = []
for m, person in enumerate(itemlistExperienced):\
    # firstActivityType += [itemlistExperienced[m].find('plan/activity').get('type')] # result: array(['business', 'home', 'leisure', 'sozializing', 'touring'], dtype='<U11')

    # start_time_all += [itemlistExperienced[m].findall('plan/activity')[-1].get('start_time')] # result: max=25:44:18, min=05:43:12
    end_time_all += [itemlistExperienced[m].findall('plan/activity')[0].get('end_time')] # result: max=23:26:56, min=02:20:48

# start_time_all_D = pd.DataFrame(start_time_all)
# end_time_all_D = pd.DataFrame(end_time_all)
##########################################################################################################
homeDurations = []
workDuration = []
m=0
for m, person in enumerate(itemlist):
    home = itemlistExperienced[m].findall('plan/activity[@type="home"]')
    work = itemlistExperienced[m].findall('plan/activity[@type="work"]')
    # homeDurations += [to_seconds(home[0].get('end_time'), strict= False).__int__()-
    #                   to_seconds('02:00:00').__int__()] #todo please enter simulation begining time for the fist activity duration
    # workDuration += [to_seconds(work[0].get('end_time'), strict= False).__int__()-
    #                  to_seconds(work[0].get('start_time'), strict= False).__int__()]

    j=0
    while j < len(home):
        homeDurations += [to_seconds(('26:00:00' if home[j].get(
            'end_time') is None else  ##todo please enter simulation end time for the last activity duration
                                      home[j].get('end_time')), strict=False).__int__() - \
                          to_seconds(('02:00:00' if home[j].find('plan/activity').get(
                              'start_time') is None else  ##todo please enter simulation begining time for the fist activity duration
                                      home[j].find('plan/activity').get('start_time')), strict=False).__int__()]
        j+=1
    k=0
    while k < len(work):
        workDuration += [to_seconds(('26:00:00' if work[k].get(
            'end_time') is None else  ##todo please enter simulation end time for the last activity duration
                                     work[k].get('end_time')), strict=False).__int__() - \
                         to_seconds(('02:00:00' if work[k].find('plan/activity').get(
                             'start_time') is None else  ##todo please enter simulation begining time for the fist activity duration
                                     work[k].find('plan/activity').get('start_time')), strict=False).__int__()]
        k += 1

    # __________________________________________________________________________________________
    homeloc += Point(float(itemlist[m].findall('plan/activity[@type="home"]')[0].get('x')),
                     float(itemlist[m].findall('plan/activity[@type="home"]')[0].get('y')))
    workloc += Point(float(itemlist[m].findall('plan/activity[@type="work"]')[0].get('x')),
                     float(itemlist[m].findall('plan/activity[@type="work"]')[0].get('y')))
