import pandas as pd
import numpy as np
import time
import datetime
######################################### importing XML file plan ######################################################
from xml.dom import minidom
from lxml import etree
start_time = time.time()
tree= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim_directory/matsim-example-project/scenarios/" \
              "example_zahra/Amsterdam/original files/AlbatrossAgentsCleaned_Stable.xml")
root = tree.getroot()
minusDuration = []
for m, person in enumerate(root.findall('person')):
    # if m%1000==0:
    #     print('{per} percent ______ time: {tt} min'.format(per=100*round(m/len(root.findall('person'))),tt=round((time.time() - start_time)/60)))
    activityList = person.findall('plan/activity')
    for n in range(1,len(activityList)-1):
        if activityList[n].get("type")!="pt interaction":
            if pd.to_timedelta(activityList[n].get('end_time')) - pd.to_timedelta(
                    activityList[n].get('start_time')) > pd.to_timedelta('0 days 00:00:00'):
                        activityList[n].set('duration',"{kk}".format(kk=(pd.to_timedelta(activityList[n].get('end_time'))-pd.to_timedelta(activityList[n].get('start_time'))))[-8:])
            if pd.to_timedelta(activityList[n].get('end_time'))-pd.to_timedelta(activityList[n].get('start_time'))<pd.to_timedelta('0 days 00:00:00'):
                minusDuration += [pd.to_timedelta(activityList[n].get('end_time'))-pd.to_timedelta(activityList[n].get('start_time'))]
            del activityList[n].attrib["end_time"]
XMLFileName = "D:/ax/zeroDuration/PlanWithDuration.xml"
tree.write(str(XMLFileName), encoding="UTF-8", method="xml",
           xml_declaration=True)

testData = tree.getroot().findall('person/plan/activity')
