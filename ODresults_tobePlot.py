import pandas as pd
import numpy as np
import time
######################################### importing XML file plan ######################################################
from xml.dom import minidom
from lxml import etree
start_time = time.time()
# itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot_testMinimalDuration/ITERS/it.1/1.experienced_plans.xml").getroot().findall('person')
# itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim_directory/matsim-example-project/scenarios/" \
#               "example_zahra/Amsterdam/original files/AlbatrossAgentsCleaned_Stable.xml").getroot().findall('person')
itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim_directory/matsim-example-project/scenarios/" \
              "example_zahra/Amsterdam/original files/PlanWithOnlyCar_again_NoGeneric.xml").getroot().findall('person')
trueLocations = pd.DataFrame(columns=['VEHICLE','activityType','x','y'])
zeroDurationActyivityCheck = pd.DataFrame(columns=['VEHICLE','activityType','duration','start','end'])
for m, person in enumerate(itemlistExperienced[0:1000]):
    if m%100==0:
        print('{percentage} percent____{duration} sec'.format(percentage=m / len(itemlistExperienced)*100,
                                                              duration=time.time() - start_time))
    activityList = itemlistExperienced[m].findall('plan/activity')
    i=0

    for i in range(1,len(activityList)):
        zerocheck = pd.DataFrame(columns=['VEHICLE','activityType','duration','start','end'])
        zerocheck.loc[i, 'VEHICLE'] = person.get('id')
        if activityList[i].get('type')!='pt interaction':
            zerocheck.loc[i, 'activityType'] = activityList[i].get('type')
            if i==(len(activityList)-1):
                zerocheck.loc[i,'start'] = pd.to_timedelta(activityList[i].get('start_time'))
                zerocheck.loc[i, 'end'] = pd.to_timedelta(activityList[0].get('end_time'))
                zerocheck.loc[i, 'duration'] = pd.to_timedelta(24*3600,unit="s")+((zerocheck.loc[i, 'end']) - (zerocheck.loc[i, 'start']))
            else:
                zerocheck.loc[i,'start'] = pd.to_timedelta(activityList[i].get('start_time'))
                zerocheck.loc[i, 'end'] = pd.to_timedelta(activityList[i].get('end_time'))
                zerocheck.loc[i, 'duration'] = ((zerocheck.loc[i, 'end']) - (zerocheck.loc[i, 'start']))
            zeroDurationActyivityCheck = zeroDurationActyivityCheck.append(zerocheck)
    # zeroDurationActyivityCheck = zeroDurationActyivityCheck.append(zerocheck)
zeroDurationActyivityCheck = zeroDurationActyivityCheck.sort_values(by=['VEHICLE','start'])
zeroDurationActyivityCheck = zeroDurationActyivityCheck.reset_index(drop=True)
print(len(zeroDurationActyivityCheck[zeroDurationActyivityCheck['duration']<=pd.to_timedelta('0 days 00:00:00')]))
zeroIDs=[]
for i in range(len(zeroDurationActyivityCheck)):
    if zeroDurationActyivityCheck.loc[i,'duration']<=pd.to_timedelta('0 days 00:00:00'):
        zeroIDs+=[zeroDurationActyivityCheck.loc[i,'VEHICLE']]

zeroIDs = pd.DataFrame(zeroIDs)
print(zeroIDs.nunique()/(zeroDurationActyivityCheck['VEHICLE'].nunique()))