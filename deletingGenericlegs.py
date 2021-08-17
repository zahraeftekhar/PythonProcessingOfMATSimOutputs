# from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
import time
import numpy as np
import pandas as pd
from lxml import etree
parser = etree.XMLParser(ns_clean=True, collect_ids=False)
itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
                               "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans.xml").getroot().findall('person')
forbiddenIDs = []
noGeneric = []
for m, person in enumerate(itemlistExperienced):
    legListExperienced = itemlistExperienced[m].findall('plan/leg/route')
    nn=0
    for n in range(len(legListExperienced)):
        if legListExperienced[n].get('type') == 'generic':
            nn+=1
    if nn==0 and itemlistExperienced[m].findall('plan/activity')[0].get('type') == \
            itemlistExperienced[m].findall('plan/activity')[-1].get('type'):
        noGeneric+=[person.get('id')]
#
# import xml.etree.ElementTree as ET
# import time
# _________________________________________________________________
# startTime = time.time()
# treeExperienced = ET.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                         "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans.xml")
# treePlan = ET.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                         "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml")
# itemE = treeExperienced.getroot().findall('person')
# itemP = treePlan.getroot().findall('person')
# rootExperienced = treeExperienced.getroot()
# rootPlan = treePlan.getroot()
# # child = rootPlan.findall('person')[0]
# # i=0
# for i, child in enumerate(itemP):
#     if child.get('id') not in noGeneric:
#         rootPlan.remove(child)
# rootPlan.remove(rootPlan[0])
# for i, child in enumerate(itemE):
#     if child.get('id') not in noGeneric:
#         rootExperienced.remove(child)
#
# #
# # ExperiencedFileName = "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans_Nogeneric(all allowed).xml"
# # PlanFileName = "C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans_Nogeneric(all allowed).xml"
# #
# # treeExperienced.write(str(ExperiencedFileName), encoding="UTF-8", method="xml",
# #            xml_declaration=True)
# #
# # treePlan.write(str(PlanFileName), encoding="UTF-8", method="xml",
# #            xml_declaration=True)
#
# print(time.time() - startTime)  # 67.54050326347351 seconds
# # ______________________________ correcting snap ___________________________________________
# snapData = pd.read_csv("D:/ax/gis/locationMappingToMezuroZones/gisOutputForPLU2.csv")
# snapDataModified = snapData.loc[snapData['VEHICLE'].isin(noGeneric),:]
# snapDataModified.to_csv("D:/ax/gis/locationMappingToMezuroZones/gisOutputForPLU2.csv", header=True,index = False)
# _____________________________correcting clusterData _______________________________________
# files = range(0,10)
# for j in files:
#     clusterData = pd.read_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=j))
#     clusterDataModified = clusterData.loc[clusterData['VEHICLE'].isin(noGeneric), :]
#     clusterDataModified.to_csv('D:/ax/gis/completePLUdata_30sec/clusterPLU_30sec/clusterPLU_30sec_{number}.CSV'.format(number=j), header=True,index= False)
# #_______________________________correcting anchorLocPLU___________________________________
files = range(0,10)
interval = 900
j=0
for j in files:
    anchorLoc = pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_{number}.CSV'.format(number=j, inter=interval))
    anchorLocModified = anchorLoc.loc[anchorLoc['VEHICLE'].isin(noGeneric), :]
    print(len(anchorLocModified['VEHICLE'].unique()))
    anchorLocModified.to_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_{number}.CSV'.format(number=j, inter=interval), header = True,index = False)
# # __________________________________________recreation of train data __________________________
# itemlistPlan = etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                         "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans_Nogeneric(all allowed).xml").getroot().findall('person')
# itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                                "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans_Nogeneric(all allowed).xml").getroot().findall('person')

kk= pd.DataFrame()
for j in files:
    kk=kk.append(pd.read_csv('D:/ax/gis/completePLUdata_{inter}sec/clusterPLU_{inter}sec/anchorLocsPLU_{inter}sec_{number}.CSV'.format(number=j, inter=interval)))
ids = kk['VEHICLE'].unique()
kk = kk.loc[kk['VEHICLE'].isin(noGeneric), :]
ids2  = kk['VEHICLE'].unique()