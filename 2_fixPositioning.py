import pandas as pd
from scipy.stats import gaussian_kde
from shapely.geometry import Point
import time
from math import floor
import numpy as np
from statistics import mean
from lxml import etree
start_time = time.time()
parser = etree.XMLParser(ns_clean=True, collect_ids=False)
itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim_directory/matsim-example-project/scenarios/" \
              "example_zahra/Amsterdam/original files/PlanWithOnlyCar_again_NoGeneric.xml").getroot().findall('person')
links = etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/output_network.xml").getroot().findall('links/link')
nodes = etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/output_network.xml").getroot().findall('nodes/node')
trueLocations = pd.DataFrame(columns=['VEHICLE','activityType','x','y'])
m=0
person = itemlistExperienced[m]
mmm=1
for m, person in enumerate(itemlistExperienced):
    if m == mmm * 100:
        print('{percentage} percent____{duration} sec'.format(percentage=m / len(itemlistExperienced)*100,
                                                              duration=time.time() - start_time))
        mmm += 1
    legList = itemlistExperienced[m].findall('plan/leg/route')
    activityList = itemlistExperienced[m].findall('plan/activity')
    i=0
    for i in range(len(legList)):
        trueloc = pd.DataFrame(columns=['VEHICLE','activityType','x','y'])
        trueloc.loc[0,'VEHICLE'] = person.get('id')
        trueloc.loc[0,'activityType'] = activityList[i+1].get('type')
        linkID = legList[i].get('end_link')
        for j in range(len(links)):
            if links[j].get('id')==linkID:
                nod = links[j].get('to')
                for k in range(len(nodes)):
                    if nodes[k].get('id')==nod:
                        trueloc.loc[0,'x'] = nodes[k].get('x')
                        trueloc.loc[0,'y'] = nodes[k].get('y')
        trueLocations = trueLocations.append(trueloc)
trueLocations.to_csv("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
                               "/Results_AlbatrossAgentsCleaned_Stable_30secSnapShot/ITERS/it.1/1.trueLocExperienced.csv", header=True, index=False)
#  43 min run time