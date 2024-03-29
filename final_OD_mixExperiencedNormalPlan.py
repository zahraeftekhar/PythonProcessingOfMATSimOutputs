
################################################### CLASS ##############################################################
class LongLat:
    def __init__(self, *args):
        self.TAZ = 0
    def set_location(self, x, y):
        from shapely.geometry import Point
        self.location = Point(x, y)

    def changeCoordSys(self, initial: str = 'epsg:23031', final: str = 'epsg:28992'):
        from pyproj import Proj, transform
        from shapely.geometry import Point
        self.location = Point(transform(Proj(init=initial), Proj(init=final), self.location.x, self.location.y))

    def zoneMapping(self, onepolygon, polygonName):
        if (onepolygon.contains(self.location)):
            self.TAZ = polygonName


class TAZmap:
    def __init__(self): pass

    def set_map(self, value):
        import geopandas as gpd
        self.map = gpd.read_file(value)


######################################### geo-spacial related packages #################################################
import pandas as pd
# from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
import time
# building the zero OD Matrix based on the number of zones in the SHP
map_mzr = TAZmap()
map_mzr.set_map('C:/Users/zahraeftekhar/PycharmProjects/XMLparsing1/netherlandsSHP/mezuro_areas_2018.shp')
inputs = map_mzr.map.geometry
tazNames = map_mzr.map.iloc[:].mzr_id
zoneZero = pd.Series(0)
matrixRowColNames = tuple(zoneZero.append(tazNames))
odsize = map_mzr.map.__len__() + 1
del map_mzr
startTime_OD = "06:00:00"
endTime_OD = "09:00:00"
# m = 0
# person = itemlist[0]


######################################### importing XML file plan ######################################################
from xml.dom import minidom
from lxml import etree
parser = etree.XMLParser(ns_clean=True, collect_ids=False)
# parser = etree.XMLPullParser(tag = "person")
# context = etree.iterparse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
#                         "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml", tag= 'person')
itemlistPlan = etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
                        "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml").getroot().findall('person')
itemlistExperienced= etree.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples"
                               "/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans.xml").getroot().findall('person')
# for index, person in enumerate(itemlistPlan):
#     # activity = person.find('plan/activity').get('type')
#     start_time = itemlistExperienced[index].find('plan/activity').get('start_time')
#     # print(activity)
#     print(start_time)


######################################### deriving OD matrix from plan files ###########################################
import numpy as np
import pandas as pd
ODMatrix_df = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32), columns=matrixRowColNames,
                           index=matrixRowColNames)  # creating empty OD matrix
# person = itemlistPlan[0]
# m=0
start_time = time.time()
for m, person in enumerate(itemlistPlan): #172 seconds for 1000 itemlist
    activityListPlan = itemlistPlan[m].findall('plan/activity')
    activityListExperienced = itemlistExperienced[m].findall('plan/activity')
    j=0
    while j < len(activityListPlan) - 1:
        start_time1 = activityListExperienced[j].get('start_time')
        if j.__eq__(0):
            start_time1 = '03:00:00' #____________________PLEASE ENTER THE BEGINING TIME OF THE SIMULATION _____________________
        end_time1 = activityListExperienced[j].get('end_time')
        start_time2 = activityListExperienced[j + 1].get('start_time')
        endActivity = end_time1 # when using inconsistent timings : endActivity = min(end_time1, start_time2)
        startNewActivity = start_time2 # when using inconsistent timings : startNewActivity = max(end_time1, start_time2)
        if start_time1 <= startTime_OD < startNewActivity:
            if endTime_OD <= endActivity:
                break
            else:
                while endTime_OD > endActivity:
                    point1 = LongLat()
                    point1.set_location(x=float(activityListPlan[j].get('x')),
                                        y=float(activityListPlan[j].get('y')))
                    point1.changeCoordSys()
                    for k in range(len(tazNames)):
                        point1.zoneMapping(inputs[k], tazNames[k])
                    origin = point1.TAZ
                    point2 = LongLat()
                    point2.set_location(x=float(activityListPlan[j + 1].get('x')),
                                        y=float(activityListPlan[j + 1].get('y')))
                    point2.changeCoordSys()
                    for k in range(len(tazNames)):
                        point2.zoneMapping(inputs[k], tazNames[k])
                    destination = point2.TAZ
                    ODMatrix_df[origin][destination] = ODMatrix_df[origin][destination] + 1
                    j += 1
                    if j >= len(activityListPlan) - 1: break
                    end_time1 = activityListExperienced[j].get('end_time')
                    # start_time2 = activityList.item(j + 1).getAttribute('start_time') # when using inconsistent timings
                    endActivity = end_time1 # when using inconsistent timings :endActivity = min(end_time1, start_time2)
                break
            # continue
        j += 1
     # print(np.sum(np.sum(ODMatrix_df, axis=0)))

print(time.time() - start_time)
# elapsedTime = time.time() - start_time
# print(time.time() - start_time)
TXTFileName = "D:/ax/OD6_9.CSV"
ODMatrix_df.to_csv(TXTFileName, index=True, header=True)
################ FINAL decision: this code fraction works for OD estimation from ground truth .XML data

################ Elapsed time: 5552.5 seconds





