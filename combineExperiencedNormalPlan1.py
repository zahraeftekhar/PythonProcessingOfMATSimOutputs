
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
xmlPlan = minidom.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.plans.xml")
itemlistPlan = xmlPlan.getElementsByTagName('person')
del xmlPlan
# importing XML file experienced plan
from xml.dom import minidom
xmlExperienced = minidom.parse("C:/Users/zahraeftekhar/eclipse-workspace/matsim-code-examples/Results_PlanWithOnlyCar_30secSnapShot/ITERS/it.1/1.experienced_plans.xml")
itemlistExperienced = xmlExperienced.getElementsByTagName('person')
del xmlExperienced


######################################### deriving OD matrix from plan files ###########################################
def ODmatrixEstimate(m):
    start_time = time.time()
    import numpy as np
    import pandas as pd
    global itemlistPlan, itemlistExperienced, startTime_OD, endTime_OD, odsize, matrixRowColNames, inputs, tazNames
    activityListPlan = itemlistPlan[m].getElementsByTagName('activity')
    activityListExperienced = itemlistExperienced[m].getElementsByTagName('activity')
    ODMatrix_df = pd.DataFrame(np.zeros((odsize, odsize), dtype=np.int32), columns=matrixRowColNames,
                              index=matrixRowColNames)  # creating empty OD matrix
    # print(m)
    j = 0
    while j < len(activityListPlan) - 1:
        start_time1 = activityListExperienced.item(j).getAttribute('start_time')
        if j.__eq__(0):
            start_time1 = '03:00:00' #____________________PLEASE ENTER THE BEGINING TIME OF THE SIMULATION _____________________
        end_time1 = activityListExperienced.item(j).getAttribute('end_time')
        start_time2 = activityListExperienced.item(j + 1).getAttribute('start_time')
        endActivity = end_time1 # when using inconsistent timings : endActivity = min(end_time1, start_time2)
        startNewActivity = start_time2 # when using inconsistent timings : startNewActivity = max(end_time1, start_time2)
        if start_time1 <= startTime_OD < startNewActivity:
            if endTime_OD <= endActivity:
                break
            else:
                while endTime_OD > endActivity:
                    point1 = LongLat()
                    point1.set_location(x=float(activityListPlan.item(j).getAttribute('x')),
                                        y=float(activityListPlan.item(j).getAttribute('y')))
                    point1.changeCoordSys()
                    for k in range(len(tazNames)):
                        point1.zoneMapping(inputs[k], tazNames[k])
                    origin = point1.TAZ
                    point2 = LongLat()
                    point2.set_location(x=float(activityListPlan.item(j + 1).getAttribute('x')),
                                        y=float(activityListPlan.item(j + 1).getAttribute('y')))
                    point2.changeCoordSys()
                    for k in range(len(tazNames)):
                        point2.zoneMapping(inputs[k], tazNames[k])
                    destination = point2.TAZ
                    ODMatrix_df[origin][destination] = ODMatrix_df[origin][destination] + 1
                    j += 1
                    if j >= len(activityListPlan) - 1: break
                    end_time1 = activityListExperienced.item(j).getAttribute('end_time')
                    # start_time2 = activityList.item(j + 1).getAttribute('start_time') # when using inconsistent timings
                    endActivity = end_time1 # when using inconsistent timings :endActivity = min(end_time1, start_time2)
                break
            # continue
        j += 1
    print(np.sum(np.sum(ODMatrix_df, axis=0)))
    print(time.time() - start_time)
    return ODMatrix_df